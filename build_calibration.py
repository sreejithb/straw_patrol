"""Build a fresh, held-out calibration set (never used for training or for the eval sets).

  fallacy part: every one of the 185 labels gets at least 1 new example, plus random extras up to N_FALLACY in total
  none part:    ~N_NONE new 'Fallacy: None' texts, spread over the same kinds as build_nofallacy.py
Both use the same generate -> auto checks -> judge pipeline as the training data, with fresh topic/format draws
(ROUND_OFFSET 300+) and a dedupe/overlap check against every existing train and eval example.

Writes data/generated/calib_fallacy.jsonl and calib_none.jsonl (chat format) and calib_*_raw.jsonl (with verdicts).
"""
import collections
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import build_dataset as bd
import build_nofallacy as bn

N_FALLACY, N_NONE = 250, 100
ROUND_OFFSET = 300
OVERLAP_NGRAM = 8
lock = threading.Lock()


def load(name):
    p = bd.GEN_DIR / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def key(text):
    return " ".join(bd.re.findall(r"[a-z]+", text.lower()))[:80]


def run_label(fallacy, need, book_grams, keys, grams, bar, stats, out, raw_path):
    seen, kept, rows = set(keys), [], []
    for r in range(bd.MAX_ROUNDS):
        items = bd.generate(fallacy, max(4, need * 3), ROUND_OFFSET + r)
        reasons = []
        for it in items:
            reason = bd.auto_check(it, fallacy, book_grams, seen)
            if not reason and bd.ngrams(it["argument"], OVERLAP_NGRAM) & grams:
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
    with lock:
        with raw_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.extend(bd.to_chat(it) for it in kept)
        stats["kept"] += len(kept)
        stats["generated"] += len(rows)
        bar.update(1)
        bar.set_postfix(kept=stats["kept"], generated=stats["generated"])


def main() -> None:
    existing = [r["messages"][1]["content"] for f in ("train.jsonl", "train_extra.jsonl", "eval.jsonl", "train_none.jsonl", "eval_none.jsonl")
                for r in load(f)]
    keys = {key(a) for a in existing}
    grams = set().union(*(bd.ngrams(a, OVERLAP_NGRAM) for a in existing))
    book_grams = set()
    for f in bd.load_fallacies():
        for ex in f["examples"]:
            book_grams |= bd.ngrams(ex["text"], bd.COPY_NGRAM)

    # ---- fallacy part: at least one per trained label, extras chosen at random
    labels = sorted({r["messages"][2]["content"].split("\n")[0][len("Fallacy: "):].strip() for f in ("train.jsonl", "train_extra.jsonl") for r in load(f)})
    by_name = {f["name"]: f for f in bd.load_fallacies()}
    labels = [n for n in labels if n in by_name]
    rng = random.Random(7)
    need = collections.Counter(labels)
    for n in rng.sample(labels, N_FALLACY - len(labels)):
        need[n] += 1
    raw_f = bd.GEN_DIR / "calib_fallacy_raw.jsonl"
    raw_f.write_text("", encoding="utf-8")
    out_f, stats = [], {"kept": 0, "generated": 0}
    bar = tqdm(total=len(labels), unit="label", desc="calib fallacy", dynamic_ncols=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda n: run_label(by_name[n], need[n], book_grams, keys, grams, bar, stats, out_f, raw_f), labels))
    bar.close()
    (bd.GEN_DIR / "calib_fallacy.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out_f), encoding="utf-8")
    print(f"calib_fallacy: {len(out_f)} examples for {len(labels)} labels", flush=True)

    # ---- none part: same kinds as the training data, fresh draws, deduped against everything that exists
    bn.ROUND_OFFSET = ROUND_OFFSET
    bn.RAW = bd.GEN_DIR / "calib_none_raw.jsonl"
    bn.RAW.write_text("", encoding="utf-8")
    bn.EXISTING.update(keys)
    total = sum(u[3] for u in bn.UNITS)
    units = [(k, s, i, max(1, round(t * N_NONE / total))) for k, s, i, t in bn.UNITS]
    out_n, stats_n = [], {"kept": 0, "generated": 0}
    bar = tqdm(total=len(units), unit="unit", desc="calib none", dynamic_ncols=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda u: bn.run_unit(u, bar, stats_n, out_n, 3, bn.PER_CALL), units))
    bar.close()
    (bd.GEN_DIR / "calib_none.jsonl").write_text("".join(json.dumps(bn.to_chat(r), ensure_ascii=False) + "\n" for r in out_n), encoding="utf-8")
    print(f"calib_none: {len(out_n)} examples")


if __name__ == "__main__":
    main()
