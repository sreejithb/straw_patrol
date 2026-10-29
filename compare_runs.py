"""Compare saved score files of two runs on identical test items: strict and group-aware top-1, None recall, false alarms, calibration.

  python compare_runs.py I J
Strict = the model's best name is in the item's answer key. Group-aware = also accepts the names in data/alias_groups.json for the key's names.
Items are matched by text, so runs scored on differently packaged files are still compared on the same items.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

import calibrate as c

DATA = Path(__file__).parent / "data"
V6 = DATA / "v6"
ALIAS = {c.norm(g): {c.norm(a) for a in alts} for g, alts in json.loads((DATA / "alias_groups.json").read_text(encoding="utf-8")).items()}
# score files of runs scored before v6 (H, I) and of v6 runs (J ...)
OLD = {"test_generated": ["eval", "eval_none", "eval_real"], "test_cocolofa": ["cocolofa_test", "cocolofa_tradition_test"], "test_mafalda": ["mafalda_test"],
       "calib_generated": ["calib_fallacy", "calib_none"], "calib_cocolofa": ["cocolofa_calib", "cocolofa_tradition_calib"]}


def tkey(t):
    return " ".join(re.findall(r"[a-z0-9']+", t.lower()))[:200]


def load(run, name):
    files = [f"scores_{run}_{name}.json"] if (DATA / f"scores_{run}_{name}.json").exists() else [f"scores_{run}_{n}.json" for n in OLD[name]]
    cands, items = None, {}
    for f in files:
        d = json.loads((DATA / f).read_text(encoding="utf-8"))
        cands = cands or d["cands"]
        for it in d["items"]:
            items[tkey(it["text"])] = it
    return cands, items


def subset_of():
    m = {}
    for p in (V6 / "test").glob("*.jsonl"):
        for l in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            m[tkey(r["text"])] = (r["source"], r["subset"], r["label"] == "None")
    return m


def evaluate(cands, it):
    names = [c.norm(x) for x in cands]
    s = np.array(it["scores"])
    best = names[int(s.argmax())]
    gold = {c.norm(g) for g in it["gold"]}
    acc = set(gold)
    for g in gold:
        acc |= ALIAS.get(g, set())
    return best in gold, best in acc, best == "none"


def main(runs):
    sub = subset_of()
    cache = {r: {n: load(r, n) for n in ("test_generated", "test_cocolofa", "test_mafalda")} for r in runs}
    keys = {n: set.intersection(*[set(cache[r][n][1]) for r in runs]) for n in ("test_generated", "test_cocolofa", "test_mafalda")}
    groups = {}
    for n in keys:
        for k in keys[n]:
            src, subset, is_none = sub[k]
            tag = f"{src}/{subset}" if src == "generated" else src
            groups.setdefault((n, tag, "None" if is_none else "fallacy"), []).append(k)
    print(f"{'test set (items matched by text)':44}{'n':>5}" + "".join(f"   {r}: strict  group" for r in runs))
    order = sorted(groups, key=lambda g: (g[0], g[1], g[2]))
    for g in order:
        n, tag, kind = g
        row = f"{tag + ' / ' + kind:44}{len(groups[g]):5}"
        for r in runs:
            cands, items = cache[r][n]
            ev = [evaluate(cands, items[k]) for k in groups[g]]
            if kind == "None":
                row += f"   {r}: {100 * np.mean([e[2] for e in ev]):5.0f}%  (None recall)"
            else:
                row += f"   {r}: {100 * np.mean([e[0] for e in ev]):5.1f}% {100 * np.mean([e[1] for e in ev]):5.1f}%"
        print(row)
    print("\nfalse-alarm rate on fallacy items (said None) and on None items (named a fallacy):")
    for n in keys:
        for kind in ("fallacy", "None"):
            ks = [k for g in order if g[0] == n and g[2] == kind for k in groups[g]]
            row = f"  {n:16}{kind:8}n={len(ks):4}"
            for r in runs:
                cands, items = cache[r][n]
                said = np.mean([evaluate(cands, items[k])[2] for k in ks])
                row += f"   {r}: said None {100 * said:4.0f}%"
            print(row)
    # ---- calibration: one temperature fitted on the pooled calibration data (generated + CoCoLoFa dev)
    print("\ncalibration (temperature fitted on generated calib + CoCoLoFa calib pooled; ECE = confidence vs hit-rate gap, lower is better):")
    for r in runs:
        cal = [load(r, n) for n in ("calib_generated", "calib_cocolofa")]
        cands = cal[0][0]
        names = [c.norm(x) for x in cands]
        mat = lambda items: (np.array([i["scores"] for i in items.values()]), np.array([[n in {c.norm(g) for g in i["gold"]} for n in names] for i in items.values()]))
        S = np.vstack([mat(x[1])[0] for x in cal]); G = np.vstack([mat(x[1])[1] for x in cal])
        T = c.fit_T(S, G)
        print(f"  run {r}: temperature {T:.2f}")
        for n in ("test_generated", "test_cocolofa", "test_mafalda"):
            cands_t, items = cache[r][n]
            S2, G2 = mat({k: items[k] for k in keys[n]})
            e1, conf1, hit = c.ece(S2, G2, 1.0)
            e2, conf2, _ = c.ece(S2, G2, T)
            print(f"    {n:16} hit-rate {100 * hit:5.1f}%   T=1: confidence {conf1:.2f} ECE {e1:.3f}   T={T:.2f}: confidence {conf2:.2f} ECE {e2:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
