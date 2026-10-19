"""Top-k accuracy: rank every known fallacy name by the model's probability and check where the gold label lands.

  python topk.py --adapter out/lora_B --run B            # writes data/topk_B.json
  python topk.py --run baseline --hint                   # untrained base model (format hint added, as in evaluate.py --hint)

The model is scored on the full answer prefix "Fallacy: <name>\n" for each of the candidate names, so the ranking is
the model's own probability for that exact string. Candidates are the fallacy names found in train.jsonl and eval.jsonl.
Top-1 should therefore match the strict greedy score from evaluate.py (up to rare ties).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import BASE, EVAL, FORMAT_HINT, norm

ROOT = Path(__file__).parent
GEN = ROOT / "data" / "generated"
MAX_K = 30
CHUNK = 40


def label_of(row):
    return row["messages"][2]["content"].split("\n")[0].replace("Fallacy: ", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--run", required=True)
    ap.add_argument("--hint", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    names = {}
    for f in (GEN / "train.jsonl", GEN / "train_extra.jsonl", EVAL):
        if f.exists():
            for l in f.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    n = label_of(json.loads(l))
                    names.setdefault(norm(n), n)  # one candidate per normalised name
    cands = sorted(names.values())
    cand_ids = [tok(f"Fallacy: {n}\n", add_special_tokens=False)["input_ids"] for n in cands]
    L = max(map(len, cand_ids))
    ids = torch.tensor([c + [tok.pad_token_id] * (L - len(c)) for c in cand_ids], device=device)
    cmask = torch.tensor([[1] * len(c) + [0] * (L - len(c)) for c in cand_ids], device=device)
    print(f"{len(cands)} candidate names, longest {L} tokens", flush=True)

    ranks, t0 = [], time.time()
    for row in tqdm(rows, unit="ex", desc=f"topk {args.run}", dynamic_ncols=True):
        system, user, _ = (m["content"] for m in row["messages"])
        msgs = [{"role": "system", "content": system + (FORMAT_HINT if args.hint else "")}, {"role": "user", "content": user}]
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
        score = torch.empty(len(cands), device=device)
        with torch.no_grad():
            for c0 in range(0, len(cands), CHUNK):  # chunked so the vocab-sized logits fit in memory
                c1 = min(c0 + CHUNK, len(cands))
                out = model(**p, use_cache=True)
                first = torch.log_softmax(out.logits[0, -1].float(), -1)  # distribution over the first candidate token
                cache = out.past_key_values
                cache.batch_repeat_interleave(c1 - c0)
                n_prompt = p["input_ids"].shape[1]
                attn = torch.cat([torch.ones(c1 - c0, n_prompt, device=device, dtype=cmask.dtype), cmask[c0:c1]], 1)
                logits = model(input_ids=ids[c0:c1], attention_mask=attn, past_key_values=cache, use_cache=True).logits
                lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(2, ids[c0:c1, 1:, None])[..., 0]  # tokens 1..L-1
                score[c0:c1] = first[ids[c0:c1, 0]] + (lp * cmask[c0:c1, 1:]).sum(1)
        order = score.argsort(descending=True).tolist()
        gold = norm(label_of(row))
        ranks.append(next(r for r, i in enumerate(order, 1) if norm(cands[i]) == gold))
    print(f"{time.time() - t0:.0f}s")

    acc = {k: sum(r <= k for r in ranks) / len(ranks) for k in range(1, MAX_K + 1)}
    print(" ".join(f"top{k}={100 * acc[k]:.1f}%" for k in (1, 2, 3, 5, 10, 20)))
    (ROOT / "data" / f"topk_{args.run}.json").write_text(json.dumps({"run": args.run, "n": len(ranks), "ranks": ranks, "acc": acc}), encoding="utf-8")


if __name__ == "__main__":
    main()
