"""LoRA fine-tune Qwen2.5-1.5B-Instruct on data/generated/train.jsonl (+ train_extra.jsonl if present) (runs on CUDA, MPS or CPU).

  python train.py                         # saves the adapter to out/lora
  python train.py --epochs 12 --patience 2   # early stopping on held-out validation examples

Loss is computed only on the assistant turn; the system and user turns are masked with -100.
Early stopping: --val-per-label examples of every fallacy are held out of training (never from eval.jsonl).
After each epoch the loss on the "Fallacy: <name>" line of those examples is measured; training stops after
--patience epochs without improvement and the best epoch's adapter is kept. --val-per-label 0 disables this
and trains for exactly --epochs.
"""
import argparse
import json
import random
from pathlib import Path

import torch
import chat_utils
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).parent
TRAIN_FILES = [ROOT / "data" / "generated" / f for f in ("train.jsonl", "train_extra.jsonl", "train_none.jsonl", "train_real.jsonl", "distill_real.jsonl")]
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LEN = 512


def encode(tok, messages):
    """Tokenise one chat; labels are -100 everywhere except the assistant reply."""
    prompt, reply = chat_utils.train_pair(tok, messages)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(reply, add_special_tokens=False)["input_ids"]
    ids = (p + r)[:MAX_LEN]
    labels = ([-100] * len(p) + r)[:MAX_LEN]
    return ids, labels


def encode_label_only(tok, messages):
    """Like encode, but only the 'Fallacy: <name>' line counts, which is what the score depends on."""
    ids, labels = encode(tok, messages)
    first_line = tok(messages[-1]["content"].split("\n")[0] + "\n", add_special_tokens=False)["input_ids"]
    start = next(i for i, l in enumerate(labels) if l != -100)
    return ids, [l if start <= i < start + len(first_line) else -100 for i, l in enumerate(labels)]


@torch.no_grad()
def label_loss(model, val, pad_id, device, dtype):
    model.eval()
    total, count = 0.0, 0
    for s in range(0, len(val), 16):
        ids, labels, mask = (t.to(device) for t in collate(val[s:s + 16], pad_id))
        with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            logits = model(input_ids=ids, attention_mask=mask).logits
        ce = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels[:, 1:],
                                               ignore_index=-100, reduction="sum")
        total += ce.item()
        count += (labels[:, 1:] != -100).sum().item()
    model.train()
    return total / max(count, 1)


def collate(batch, pad_id):
    n = max(len(i) for i, _ in batch)
    ids = torch.tensor([i + [pad_id] * (n - len(i)) for i, _ in batch])
    labels = torch.tensor([l + [-100] * (n - len(l)) for _, l in batch])
    mask = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in batch])
    return ids, labels, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "out" / "lora"))
    ap.add_argument("--epochs", type=int, default=3, help="maximum number of epochs")
    ap.add_argument("--patience", type=int, default=2, help="stop after this many epochs without validation improvement")
    ap.add_argument("--val-per-label", type=int, default=1, help="held-out examples per fallacy; 0 = no validation or early stopping")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (timing probe: see s/step before a long run)")
    ap.add_argument("--base", default=BASE, help="base model; the adapter only works with the base it was trained on")
    ap.add_argument("--four-bit", action="store_true", help="load the base model in 4-bit (QLoRA); needed for 7B on a 16 GB GPU")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok, model, dtype, device = chat_utils.load_model(args.base, args.four_bit)
    print(f"device={device} dtype={dtype} base={args.base}")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    rows = [json.loads(l)["messages"] for f in TRAIN_FILES if f.exists()
            for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(rows)} training examples")
    val_rows = []
    if args.val_per_label:
        by_label = {}
        for m in rows:
            by_label.setdefault(m[-1]["content"].split("\n")[0], []).append(m)
        rng = random.Random(args.seed)
        for group in by_label.values():
            rng.shuffle(group)
            val_rows += group[:args.val_per_label]
        held = {id(m) for m in val_rows}
        rows = [m for m in rows if id(m) not in held]
        print(f"{len(rows)} train / {len(val_rows)} validation examples")
    data = [encode(tok, m) for m in rows]
    val = [encode_label_only(tok, m) for m in val_rows]
    steps_per_epoch = (len(data) + args.batch - 1) // args.batch
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps_per_epoch * args.epochs), steps_per_epoch * args.epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda" and dtype == torch.float16)

    model.train()
    best, best_epoch, bad = float("inf"), 0, 0
    done = 0
    bar = tqdm(total=steps_per_epoch * args.epochs, unit="step", desc="train", dynamic_ncols=True)
    for epoch in range(args.epochs):
        random.shuffle(data)
        for s in range(0, len(data), args.batch):
            ids, labels, mask = (t.to(device) for t in collate(data[s:s + args.batch], tok.pad_token_id))
            with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            bar.update(1)
            done += 1
            if args.max_steps and done >= args.max_steps:
                break
            bar.set_postfix(epoch=epoch + 1, loss=f"{loss.item():.3f}")
        if args.max_steps and done >= args.max_steps:
            model.save_pretrained(args.out)
            print(f"stopped after {done} steps (--max-steps); see the s/step in the progress bar")
            break
        if not val:
            model.save_pretrained(args.out)  # checkpoint after every epoch, so a dropped session loses little
            continue
        vloss = label_loss(model, val, tok.pad_token_id, device, dtype)
        improved = vloss < best - 1e-4
        tqdm.write(f"epoch {epoch + 1}: validation label loss {vloss:.4f} (best {min(best, vloss):.4f})"
                   f"{' -> saved' if improved else ''}")
        if improved:
            best, best_epoch, bad = vloss, epoch + 1, 0
            model.save_pretrained(args.out)
        else:
            bad += 1
            if bad >= args.patience:
                tqdm.write(f"early stop: no improvement for {bad} epochs")
                break
    bar.close()
    print(f"saved adapter to {args.out}" + (f" (best epoch {best_epoch}, validation label loss {best:.4f})" if val else ""))


if __name__ == "__main__":
    main()
