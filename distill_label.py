"""Option B: label REAL comments with an LLM and keep only the labels we can trust, to train our own model on real-style text.

Sources (both stay local: external data, licences unclear or share-alike -- do not upload to the Hub or GitHub):
  cocolofa_train      CoCoLoFa train split, top-level comments. Keep a comment when the LLM's answer agrees with the CoCoLoFa label.
  mafalda_unlabeled   MAFALDA non-annotated texts (Reddit, debates). No labels: keep a text when two different LLMs give the same answer.
Output: data/generated/distill_real.jsonl (chat format: answer `Fallacy: <name>` + a one-sentence explanation) and distill_raw.jsonl.

  python distill_label.py                                  # dry run: counts and token estimates, spends nothing
  python distill_label.py --yes --n-cocolofa 1500 --n-mafalda 1500
  python distill_label.py --price-in 2.5 --price-cached 0.25 --price-out 10     # USD per 1M tokens, to turn tokens into dollars
"""
import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

import build_dataset as bd
import llm_baseline as lb
from build_external import COCO
from evaluate import norm

ROOT = Path(__file__).parent
EXT = ROOT / "external"
OUT = bd.GEN_DIR
SYSTEM_MSG = "You are a logical-fallacy expert. Identify the fallacy in the argument and explain it."
MAX_WORDS, MIN_WORDS = 160, 12
# measured on the 766-call baseline: ~1.1k uncached + ~6.9k cached input tokens per call (prompt caching), ~0.08k output
TOK_PER_CALL = {"uncached": 1100, "cached": 6900, "out_label": 150, "out_vote": 30}
COCO_ALL = dict(COCO) | {"appeal to tradition": ["Appeal to Tradition"]}

SYSTEM = lb.SYSTEM + "\n- Also write `why`: ONE sentence that starts exactly with 'This is a <Name>:' / 'This is an <Name>:' (using your `best` name) and says what the commenter did wrong, or, for None, starts with 'This is not a fallacy:' and says why there is nothing to flag."
lock = threading.Lock()
usage = {"calls": 0, "in": 0, "cached": 0, "out": 0}


def ask(text, model):
    for attempt in range(5):
        try:
            r = bd.client.chat.completions.create(model=model, response_format=lb.SCHEMA, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
            with lock:
                u = r.usage
                usage["calls"] += 1
                usage["in"] += u.prompt_tokens
                usage["cached"] += getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
                usage["out"] += u.completion_tokens
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            if attempt == 4 or "no credits" in str(e):
                raise
            time.sleep(2 ** attempt)


def clean(text):
    text = re.sub(r"\bTITLE:\s*", "", text)
    text = re.sub(r"\s*\bPOST:\s*", "\n\n", text)
    return text.strip()


def ok_length(t):
    return MIN_WORDS <= len(t.split()) <= MAX_WORDS


def cocolofa_rows(n, seed):
    comments = [c for a in json.loads((EXT / "cocolofa" / "train.json").read_text(encoding="utf-8")) for c in a["comments"] if not c.get("respond_to")]
    rows = [{"source": "cocolofa", "id": c["id"], "text": c["comment"].strip(), "gold": COCO_ALL.get(c["fallacy"], ["None"] if c["fallacy"] == "none" else None)} for c in comments]
    rows = [r for r in rows if r["gold"] and ok_length(r["text"])]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def mafalda_rows(n, seed):
    blocked = set()  # never train on texts that are in our test sets
    for f in ("mafalda_test.jsonl", "cocolofa_test.jsonl", "cocolofa_calib.jsonl"):
        blocked |= {json.loads(l)["text"].strip()[:100] for l in (EXT / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    gold = {json.loads(l)["text"].strip()[:100] for l in (EXT / "mafalda" / "gold_standard_dataset.jsonl").read_text(encoding="utf-8").splitlines()}
    rows = []
    for i, line in enumerate((EXT / "mafalda" / "non_annotated_dataset.jsonl").read_text(encoding="utf-8").splitlines()):
        t = clean(json.loads(line)["text"])
        if ok_length(t) and t[:100] not in blocked | gold and clean(t)[:100] not in blocked | gold:
            rows.append({"source": "mafalda", "id": i, "text": t, "gold": None})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cocolofa", type=int, default=1500)
    ap.add_argument("--n-mafalda", type=int, default=1500)
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--vote-model", default="gpt-5.4-mini")
    ap.add_argument("--max-none-share", type=float, default=0.35, help="cap on the share of None answers in the output")
    ap.add_argument("--per-label-cap", type=int, default=120)
    ap.add_argument("--yes", action="store_true", help="actually call the API (default: dry run)")
    ap.add_argument("--price-in", type=float), ap.add_argument("--price-cached", type=float), ap.add_argument("--price-out", type=float)
    ap.add_argument("--price-vote-in", type=float), ap.add_argument("--price-vote-cached", type=float), ap.add_argument("--price-vote-out", type=float)
    args = ap.parse_args()

    co, ma = cocolofa_rows(args.n_cocolofa, 0), mafalda_rows(args.n_mafalda, 0)
    calls_main, calls_vote = len(co) + len(ma), len(ma)
    est = lambda calls, out: dict(uncached=calls * TOK_PER_CALL["uncached"], cached=calls * TOK_PER_CALL["cached"], out=calls * out)
    e1, e2 = est(calls_main, TOK_PER_CALL["out_label"]), est(calls_vote, TOK_PER_CALL["out_vote"])
    print(f"texts: CoCoLoFa {len(co)}, MAFALDA {len(ma)}")
    print(f"{args.model:12} {calls_main:5} calls: ~{e1['uncached'] / 1e6:.2f}M uncached + {e1['cached'] / 1e6:.2f}M cached input tokens, ~{e1['out'] / 1e6:.2f}M output tokens")
    print(f"{args.vote_model:12} {calls_vote:5} calls: ~{e2['uncached'] / 1e6:.2f}M uncached + {e2['cached'] / 1e6:.2f}M cached input tokens, ~{e2['out'] / 1e6:.2f}M output tokens")
    if args.price_in is not None:
        usd = lambda e, pi, pc, po: (e["uncached"] * pi + e["cached"] * pc + e["out"] * po) / 1e6
        total = usd(e1, args.price_in, args.price_cached, args.price_out)
        if args.price_vote_in is not None:
            total += usd(e2, args.price_vote_in, args.price_vote_cached, args.price_vote_out)
        print(f"estimated cost at the prices you gave: ${total:,.2f}")
    if not args.yes:
        print("dry run: nothing was sent. Re-run with --yes to label.")
        return

    raw_path = OUT / "distill_raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    rows = co + ma
    with ThreadPoolExecutor(8) as pool:
        main_ans = list(tqdm(pool.map(lambda r: ask(r["text"], args.model), rows), total=len(rows), desc=f"label ({args.model})", dynamic_ncols=True))
    vote_idx = [i for i, r in enumerate(rows) if r["source"] == "mafalda"]
    with ThreadPoolExecutor(8) as pool:
        vote_ans = dict(zip(vote_idx, tqdm(pool.map(lambda i: ask(rows[i]["text"], args.vote_model), vote_idx), total=len(vote_idx), desc=f"vote ({args.vote_model})", dynamic_ncols=True)))

    kept, stat = [], {"cocolofa": [0, 0], "mafalda": [0, 0]}
    with raw_path.open("a", encoding="utf-8") as f:
        for i, (r, a) in enumerate(zip(rows, main_ans)):
            if r["source"] == "cocolofa":
                keep = norm(a["best"]) in {norm(g) for g in r["gold"]}
            else:
                keep = norm(a["best"]) == norm(vote_ans[i]["best"])
            stat[r["source"]][1] += 1
            stat[r["source"]][0] += keep
            f.write(json.dumps({**r, "answer": a, "vote": vote_ans.get(i), "keep": keep}, ensure_ascii=False) + "\n")
            if keep:
                kept.append((r, a))
    for s, (k, n) in stat.items():
        print(f"{s}: kept {k}/{n} = {100 * k / max(n, 1):.0f}% (LLM agrees with {'the CoCoLoFa label' if s == 'cocolofa' else 'a second LLM'})")

    # balance: cap None and every label so a handful of common answers cannot swamp the set
    rng = random.Random(1)
    rng.shuffle(kept)
    per, out = {}, []
    for r, a in kept:
        per[a["best"]] = per.get(a["best"], 0) + 1
        if per[a["best"]] <= args.per_label_cap:
            out.append((r, a))
    nones = [x for x in out if norm(x[1]["best"]) == "none"]
    cap = int(args.max_none_share / (1 - args.max_none_share) * (len(out) - len(nones)))
    drop = {id(x) for x in nones[cap:]}
    out = [x for x in out if id(x) not in drop]
    chat = [{"messages": [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": r["text"]},
                          {"role": "assistant", "content": f"Fallacy: {a['best']}\n{a['why'].strip()}"}]} for r, a in out]
    (OUT / "distill_real.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chat), encoding="utf-8")
    labels = {}
    for r, a in out:
        labels[a["best"]] = labels.get(a["best"], 0) + 1
    print(f"wrote {len(chat)} examples to distill_real.jsonl ({len(labels)} labels; most common: {sorted(labels.items(), key=lambda kv: -kv[1])[:8]})")
    print(f"tokens used: in {usage['in']:,} (cached {usage['cached']:,}), out {usage['out']:,}, calls {usage['calls']}")


if __name__ == "__main__":
    main()
