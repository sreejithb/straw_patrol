"""Add more TRAIN examples per fallacy without touching eval.jsonl.

Reuses the generate -> auto-check -> judge steps from build_dataset.py, but:
  * uses fresh topic/format assignments (round numbers start at ROUND_OFFSET),
  * rejects anything that duplicates or heavily overlaps an existing train or eval argument,
  * writes only data/generated/train_extra.jsonl (chat format) and train_extra_raw.jsonl (with judge verdicts).
train.py can then be pointed at train.jsonl + train_extra.jsonl.

Usage:
  python extend_dataset.py --names "Appeal to Fear,Accident"   # pilot
  python extend_dataset.py --top-up 12                         # every label up to 12 train examples
"""
import argparse
import collections
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import build_dataset as bd

ROUND_OFFSET = 10
OVERLAP_NGRAM = 8
raw_lock = threading.Lock()
stats = {"kept": 0, "generated": 0}


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def label_of(row):
    return row["messages"][2]["content"].split("\n")[0].replace("Fallacy: ", "").strip()


def run_one(fallacy, need, book_grams, existing_keys, existing_grams, out_path, bar, chat_path):
    seen = set(existing_keys)
    kept, rows = [], []
    for r in range(bd.MAX_ROUNDS):
        items = bd.generate(fallacy, 8, ROUND_OFFSET + r)
        reasons = []
        for it in items:
            reason = bd.auto_check(it, fallacy, book_grams, seen)
            if not reason and bd.ngrams(it["argument"], OVERLAP_NGRAM) & existing_grams:
                reason = "overlaps existing example"
            reasons.append(reason)
        with ThreadPoolExecutor(4) as pool:
            verdicts = list(pool.map(lambda x: None if x[1] else bd.judge(x[0], fallacy), zip(items, reasons)))
        for it, reason, v in zip(items, reasons, verdicts):
            keep = not reason and bd.passes(v)
            rows.append({**it, "keep": keep, "reject_reason": reason or ("" if keep else v["issues"] or "judge"), "judge": v})
            if keep and len(kept) < need:
                kept.append(it)
        if len(kept) >= need:
            break
    with raw_lock:  # write per label, so an interrupted run keeps what it finished
        with out_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with chat_path.open("a", encoding="utf-8") as f:
            for it in kept:
                f.write(json.dumps(bd.to_chat(it), ensure_ascii=False) + "\n")
        stats["kept"] += len(kept)
        stats["generated"] += len(rows)
        bar.update(1)
        bar.set_postfix(kept=stats["kept"], generated=stats["generated"])
    return len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", help="comma-separated exact fallacy names (pilot)")
    ap.add_argument("--top-up", type=int, default=12, help="bring each label to this many train examples")
    args = ap.parse_args()

    train, ev = load(bd.GEN_DIR / "train.jsonl"), load(bd.GEN_DIR / "eval.jsonl")
    counts = collections.Counter(label_of(r) for r in train)
    existing = [r["messages"][1]["content"] for r in train + ev]
    keys = {" ".join(bd.re.findall(r"[a-z]+", a.lower()))[:80] for a in existing}
    grams = set().union(*(bd.ngrams(a, OVERLAP_NGRAM) for a in existing))
    book_grams = set()
    for f in bd.load_fallacies():
        for ex in f["examples"]:
            book_grams |= bd.ngrams(ex["text"], bd.COPY_NGRAM)

    by_name = {f["name"]: f for f in bd.load_fallacies()}
    wanted = [n.strip() for n in args.names.split(",")] if args.names else list(counts)
    jobs = [(by_name[n], args.top_up - counts[n]) for n in wanted if n in by_name and counts[n] < args.top_up]
    print(f"{len(jobs)} labels, {sum(n for _, n in jobs)} examples wanted")

    out_path = bd.GEN_DIR / "train_extra_raw.jsonl"
    chat_path = bd.GEN_DIR / "train_extra.jsonl"
    if not args.names:
        chat_path.write_text("", encoding="utf-8")  # full run starts a fresh file
    bar = tqdm(total=len(jobs), unit="label", desc="extend", dynamic_ncols=True)
    with ThreadPoolExecutor(8) as pool:
        total = sum(pool.map(lambda j: run_one(j[0], j[1], book_grams, keys, grams, out_path, bar, chat_path), jobs))
    bar.close()
    print(f"wrote {total} examples to {chat_path}")


if __name__ == "__main__":
    main()
