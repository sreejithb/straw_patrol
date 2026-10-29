"""Span test: does classifying the labelled span beat classifying the whole MAFALDA text?  (local analysis of saved scores)

  python compare_spans.py J Jspan
  run 1 = scores of the WHOLE-text run (data/scores_<run>_test_mafalda.json), run 2 = scores of the SPAN run (data/scores_<run2>_spans.json)
For every span the answer key is that span's own label(s). 'whole' = the model's answer for the full parent text, 'span' = its answer for the span alone.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import calibrate as c
from compare_runs import ALIAS, tkey

ROOT = Path(__file__).parent
DATA, EXT = ROOT / "data", ROOT / "external"


def best_and_accepts(cands, it, strict):
    names = [c.norm(x) for x in cands]
    best = names[int(np.argmax(it["scores"]))]
    gold = {c.norm(g) for g in it["gold"]}
    if not strict:
        for g in list(gold):
            gold |= ALIAS.get(g, set())
    return best, gold


def main(whole_run, span_run):
    texts = [json.loads(l)["text"] for l in (EXT / "mafalda" / "gold_standard_dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    wd = json.loads((DATA / f"scores_{whole_run}_test_mafalda.json").read_text(encoding="utf-8"))
    whole = {tkey(i["text"]): i for i in wd["items"]}
    sd = json.loads((DATA / f"scores_{span_run}_spans.json").read_text(encoding="utf-8"))
    spans = [json.loads(l) for l in (EXT / "mafalda_spans.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {i["id"]: i for i in sd["items"]}
    n_spans_in_parent = defaultdict(int)
    for s in spans:
        n_spans_in_parent[s["parent"]] += 1
    rows = []
    for s in spans:
        w = whole.get(tkey(texts[s["parent"]]))
        if w is None or s["id"] not in by_id:
            continue
        key = {"gold": s["gold"]}
        bs, acc_s = best_and_accepts(sd["cands"], {**by_id[s["id"]], **key}, True)
        bw, acc_w = best_and_accepts(wd["cands"], {**w, **key}, True)
        _, gs = best_and_accepts(sd["cands"], {**by_id[s["id"]], **key}, False)
        rows.append(dict(span=bs in acc_s, whole=bw in acc_w, span_g=bs in gs, whole_g=bw in gs, span_none=bs == "none", whole_none=bw == "none",
                         words=s["words"], multi=n_spans_in_parent[s["parent"]] > 1, orig=s["orig"]))
    n = len(rows)
    print(f"{n} spans scored (of {len(spans)})\n")
    f = lambda k, sel=rows: 100 * np.mean([r[k] for r in sel])
    print(f"strict hit    : span-only {f('span'):5.1f}%   whole text {f('whole'):5.1f}%   (same answer key: the span's own label)")
    print(f"group-aware   : span-only {f('span_g'):5.1f}%   whole text {f('whole_g'):5.1f}%")
    print(f"said None     : span-only {f('span_none'):5.1f}%   whole text {f('whole_none'):5.1f}%")
    both = sum(r["span"] and r["whole"] for r in rows); only_s = sum(r["span"] and not r["whole"] for r in rows)
    only_w = sum(r["whole"] and not r["span"] for r in rows); neither = n - both - only_s - only_w
    print(f"\nper span: both right {both}, only the span run right {only_s}, only the whole-text run right {only_w}, neither {neither}")
    print("\nby span length (words):")
    for name, lo, hi in (("up to 15", 0, 15), ("16-30", 16, 30), ("over 30", 31, 999)):
        sel = [r for r in rows if lo <= r["words"] <= hi]
        if sel:
            print(f"  {name:10} n={len(sel):3}   span {f('span', sel):5.1f}%   whole {f('whole', sel):5.1f}%")
    print("\nby how many labelled spans the parent text has:")
    for name, flag in (("one span", False), ("two or more", True)):
        sel = [r for r in rows if r["multi"] == flag]
        if sel:
            print(f"  {name:12} n={len(sel):3}   span {f('span', sel):5.1f}%   whole {f('whole', sel):5.1f}%")
    print("\nby MAFALDA type (spans with that type, n >= 8):")
    bt = defaultdict(list)
    for r in rows:
        for o in set(r["orig"]):
            bt[o.lower()].append(r)
    for o, sel in sorted(bt.items(), key=lambda kv: -len(kv[1])):
        if len(sel) >= 8:
            print(f"  {o:28} n={len(sel):3}   span {f('span', sel):5.1f}%   whole {f('whole', sel):5.1f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
