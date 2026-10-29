"""Score every fallacy name (plus None) for every text in one or more sets, and save the raw scores.

  python score_candidates.py --adapter out/lora_H --run H \
      --set eval=data/generated/eval.jsonl --set cocolofa_test=external/cocolofa_test.jsonl

A set is either a chat-format file (train/eval style, gold = the answer's fallacy name) or an external file with
{"text", "gold": [names that count as correct]} per line (see build_external.py).
Writes data/scores_<run>_<set>.json = {"cands": [...], "items": [{"id", "text", "gold", "scores": [...]}]}.
`scores` is the log-probability of the exact answer prefix "Fallacy: <name>\n" for each candidate; calibration
(temperature, thresholds) is fitted offline from these, see calibrate.py.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import chat_utils
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import BASE, norm

ROOT = Path(__file__).parent
GEN = ROOT / "data" / "generated"
SYSTEM_MSG = "You are a logical-fallacy expert. Identify the fallacy in the argument and explain it."
CHUNK = 62  # 186 candidates = 3 chunks; keeps the vocab-sized logits small


def label_of(row):
    return row["messages"][-1]["content"].split("\n")[0].replace("Fallacy: ", "").strip()


def read_set(path):
    items = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        if "messages" in r:
            items.append({"id": i, "text": r["messages"][1]["content"], "gold": [label_of(r)]})
        else:
            items.append({"id": r.get("id", i), "text": r["text"], "gold": r["gold"]})
    return items


def candidate_names():
    names = {}
    v6 = ROOT / "data" / "v6" / "train_v6.jsonl"
    if v6.exists():  # v6 run: the label set is whatever the v6 training file contains
        for line in v6.read_text(encoding="utf-8").splitlines():
            if line.strip():
                n = label_of(json.loads(line))
                names.setdefault(norm(n), n)
        return sorted(names.values())
    for f in ("train.jsonl", "train_extra.jsonl", "train_none.jsonl", "train_real.jsonl", "eval.jsonl"):
        if not (GEN / f).exists():
            continue
        for line in (GEN / f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                n = label_of(json.loads(line))
                names.setdefault(norm(n), n)
    return sorted(names.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", action="append", required=True, help="NAME=PATH, repeatable")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--four-bit", action="store_true", help="load the base model in 4-bit (7B+ models on a 16 GB GPU)")
    ap.add_argument("--chunk", type=int, default=0, help="candidates scored per forward pass; 0 = all at once on a CUDA GPU, 62 otherwise")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    tok, model, dtype, device = chat_utils.load_model(args.base, args.four_bit)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    cands = candidate_names()
    chunk = args.chunk or (len(cands) if device == "cuda" else CHUNK)  # one pass over all names is fastest; chunks keep the vocab-sized logits small
    cand_ids = [tok(f"Fallacy: {n}\n", add_special_tokens=False)["input_ids"] for n in cands]
    L = max(map(len, cand_ids))
    ids = torch.tensor([c + [tok.pad_token_id] * (L - len(c)) for c in cand_ids], device=device)
    cmask = torch.tensor([[1] * len(c) + [0] * (L - len(c)) for c in cand_ids], device=device)
    known = {norm(n) for n in cands}
    print(f"{len(cands)} candidates ('None' included: {'none' in known})", flush=True)

    for spec in args.set:
        name, path = spec.split("=", 1)
        items = read_set(path)[:args.limit]
        missing = sorted({g for it in items for g in it["gold"] if norm(g) not in known})
        if missing:
            print(f"WARNING {name}: gold names that are not candidates: {missing}")
        t0 = time.time()
        for it in tqdm(items, unit="ex", desc=f"{args.run}/{name}", dynamic_ncols=True):
            msgs = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": it["text"]}]
            p = chat_utils.prompt_ids(tok, msgs, device)
            score = torch.empty(len(cands), device=device)
            with torch.no_grad():
                for c0 in range(0, len(cands), chunk):
                    c1 = min(c0 + chunk, len(cands))
                    out = model(**p, use_cache=True)
                    first = torch.log_softmax(out.logits[0, -1].float(), -1)
                    cache = out.past_key_values
                    cache.batch_repeat_interleave(c1 - c0)
                    attn = torch.cat([torch.ones(c1 - c0, p["input_ids"].shape[1], device=device, dtype=cmask.dtype), cmask[c0:c1]], 1)
                    logits = model(input_ids=ids[c0:c1], attention_mask=attn, past_key_values=cache, use_cache=True).logits
                    lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(2, ids[c0:c1, 1:, None])[..., 0]
                    score[c0:c1] = first[ids[c0:c1, 0]] + (lp * cmask[c0:c1, 1:]).sum(1)
            it["scores"] = [round(s, 4) for s in score.tolist()]
        (ROOT / "data" / f"scores_{args.run}_{name}.json").write_text(json.dumps({"cands": cands, "items": items}, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: {len(items)} items in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
