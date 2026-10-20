"""Fit a softmax temperature on saved scores and report calibration + prediction-set statistics.

  python calibrate.py                      # compares temperatures fitted on generated vs real-style calibration data
Reads data/scores_H_<set>.json (see score_candidates.py). A text counts as a hit when the model's name is in its `gold` list.
"""
import json
import math
import re
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
RUN = "H"


def norm(n):
    n = re.sub(r"[^a-z0-9 ]", " ", n.lower())
    return " ".join(re.sub(r"\bfallacy\b", " ", n).split())


def load(*sets):
    S, G = [], []
    for s in sets:
        d = json.loads((DATA / f"scores_{RUN}_{s}.json").read_text(encoding="utf-8"))
        names = [norm(c) for c in d["cands"]]
        for it in d["items"]:
            S.append(it["scores"])
            g = {norm(x) for x in it["gold"]}
            G.append([n in g for n in names])
    return np.array(S, dtype=np.float64), np.array(G, dtype=bool)


def probs(S, T):
    z = S / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(S, G, T):
    return float(-np.log(np.maximum((probs(S, T) * G).sum(1), 1e-12)).mean())


def fit_T(S, G):
    grid = np.exp(np.linspace(math.log(0.5), math.log(40), 120))
    return float(min(grid, key=lambda T: nll(S, G, T)))


def ece(S, G, T, bins=10):
    p = probs(S, T)
    top = p.argmax(1)
    conf = p.max(1)
    hit = G[np.arange(len(G)), top]
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total), float(conf.mean()), float(hit.mean())


def pred_sets(S, G, T, q):
    p = probs(S, T)
    order = np.argsort(-p, 1)
    sizes, covered, single_hit, singles = [], 0, 0, 0
    for i in range(len(p)):
        cum = np.cumsum(p[i, order[i]])
        k = int(np.searchsorted(cum, q) + 1)
        k = min(k, 5)  # never show more than 5 names
        chosen = order[i, :k]
        sizes.append(k)
        covered += G[i, chosen].any()
        if k == 1:
            singles += 1
            single_hit += G[i, chosen[0]]
    n = len(p)
    return covered / n, float(np.mean(sizes)), singles / n, (single_hit / singles if singles else float("nan"))


if __name__ == "__main__":
    fits = {"none (T=1)": None, "generated calib": load("calib_fallacy", "calib_none"), "CoCoLoFa calib": load("cocolofa_calib")}
    tests = {"generated eval": load("eval", "eval_none"), "CoCoLoFa test": load("cocolofa_test"), "MAFALDA": load("mafalda_test")}
    Ts = {k: (1.0 if v is None else fit_T(*v)) for k, v in fits.items()}
    print("temperature fitted on each calibration set:", {k: round(v, 2) for k, v in Ts.items()})
    for name, (S, G) in tests.items():
        print(f"\n=== {name} (n={len(S)})   top-1 hit rate {100 * G[np.arange(len(G)), S.argmax(1)].mean():.1f}%")
        print(f"  {'temperature from':18}{'T':>6}{'ECE':>8}{'mean conf':>11}{'NLL':>8}")
        for k, T in Ts.items():
            e, c, h = ece(S, G, T)
            print(f"  {k:18}{T:6.2f}{e:8.3f}{c:11.3f}{nll(S, G, T):8.3f}")
