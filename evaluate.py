"""Score a model on data/generated/eval.jsonl: does it name the right fallacy?

  python evaluate.py                          # base model, no fine-tuning (the baseline)
  python evaluate.py --adapter out/lora       # base model + LoRA adapter (after stage 4)

Strict score  = first line of the answer is 'Fallacy: <name>' and the name matches the label.
Lenient score = the label's name appears anywhere in the first 32 generated tokens.
Names are compared case-insensitively with punctuation and a trailing 'fallacy' removed.
"""
import argparse
import collections
import json
import re
import time
from pathlib import Path

import torch
import chat_utils
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
EVAL = ROOT / "data" / "generated" / "eval.jsonl"
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW = 32  # enough for "Fallacy: <name>"; the explanation is not scored
FORMAT_HINT = ("\nReply in exactly this format: first line 'Fallacy: <name of the fallacy>', "
               "then a short explanation of 1-3 sentences.")


def norm(name: str) -> str:
    name = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    name = re.sub(r"\bfallacy\b", " ", name)
    return " ".join(name.split())


def first_line_name(answer: str) -> str:
    line = answer.strip().split("\n")[0]
    return re.sub(r"^\W*fallacy\s*:\s*", "", line, flags=re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="path to a LoRA adapter")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--hint", action="store_true", help="add the output-format hint to the system prompt (baseline only)")
    ap.add_argument("--out", default="baseline_preds.jsonl")
    ap.add_argument("--eval-file", default=str(EVAL), help="eval set; use data/generated/eval_none.jsonl for the no-fallacy set")
    ap.add_argument("--base", default=BASE, help="base model; must be the one the adapter was trained on")
    ap.add_argument("--four-bit", action="store_true", help="load the base model in 4-bit (use for 7B on a 16 GB GPU)")
    args = ap.parse_args()

    tok, model, dtype, device = chat_utils.load_model(args.base, args.four_bit)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in Path(args.eval_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    # One prompt at a time: batched generation with left padding returns "!!!!" on Apple's MPS backend.
    results, t0 = [], time.time()
    bar = tqdm(total=len(rows), unit="ex", desc="eval", dynamic_ncols=True)
    for row in rows:
        system, user, gold = (m["content"] for m in row["messages"])
        label = gold.split("\n")[0].replace("Fallacy: ", "").strip()
        msgs = [{"role": "system", "content": system + (FORMAT_HINT if args.hint else "")},
                {"role": "user", "content": user}]
        ids = chat_utils.prompt_ids(tok, msgs, device)
        t_gen = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False)
        new = out[0, ids["input_ids"].shape[1]:]
        answer = tok.decode(new, skip_special_tokens=True)
        results.append({"label": label, "argument": user, "answer": answer,
                        "strict": norm(first_line_name(answer)) == norm(label),
                        "lenient": norm(label) in norm(answer)})
        bar.update(1)
        bar.set_postfix(strict=sum(r["strict"] for r in results), lenient=sum(r["lenient"] for r in results),
                        tok_s=f"{len(new) / max(time.time() - t_gen, 1e-9):.1f}")
    bar.close()

    n = len(results)
    print(f"\nModel: {args.base}{' + ' + args.adapter if args.adapter else ' (no fine-tuning)'}")
    print(f"Strict  (first line names the fallacy): {sum(r['strict'] for r in results)}/{n} = {100 * sum(r['strict'] for r in results) / n:.1f}%")
    print(f"Lenient (label appears in answer):      {sum(r['lenient'] for r in results)}/{n} = {100 * sum(r['lenient'] for r in results) / n:.1f}%")
    said_none = sum(norm(first_line_name(r["answer"])) == "none" for r in results)
    if all(norm(r["label"]) == "none" for r in results):
        print(f"No-fallacy set: said 'None' on {said_none}/{n} = {100 * said_none / n:.1f}%  (false alarms: {n - said_none}/{n} = {100 * (n - said_none) / n:.1f}%)")
    else:
        print(f"Said 'None' on a real fallacy:          {said_none}/{n}")
    fmt = sum(1 for r in results if r["answer"].strip().lower().startswith("fallacy:"))
    print(f"Followed 'Fallacy: <name>' format:      {fmt}/{n}")
    (ROOT / "data" / args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")


if __name__ == "__main__":
    main()
