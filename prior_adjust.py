"""Test a 'common fallacies' prior: add LAMBDA to the score of every well-known fallacy (and to None).

The list COMMON was written by judgment before looking at any result. LAMBDA is the only fitted number: it is chosen on the
CoCoLoFa calibration slice and then checked on data it never saw (CoCoLoFa test, MAFALDA, the generated sets).
A control repeats the test with RANDOM name lists of the same size: if the real list is not clearly better than random lists,
the gain comes from the benchmark's label mix, not from favouring common fallacies.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import calibrate as c  # noqa: E402

COMMON = """Abusive Ad Hominem|Circumstantial Ad Hominem|Appeal to false authority|Appeal to Common Belief|Appeal to Consequences|
Appeal to Fear|Appeal to Nature|Appeal to Novelty|Appeal to Pity|Appeal to anger|Appeal to emotion|Appeal to ridicule|
Argument from Ignorance|Causal Reductionism|Cherry Picking|Circular Reasoning|Cum Hoc Ergo Propter Hoc|Post hoc|
Complex question fallacy|Double Standard|Fallacy of Division|Fallacy of composition|False Equivalence|False dilemma|
Gambler's fallacy|Genetic Fallacy|Guilt by Association|Hasty Generalization|Moving the Goalposts|No True Scotsman|Red Herring|
Relative privation|Shifting the Burden of Proof|Slippery Slope|Straw Man Fallacy|Tu Quoque|Weak analogy|Ad Hoc Rescue|Moralistic fallacy""".replace("\n", "").split("|")

SETS = {"CoCoLoFa calib (fit)": ("cocolofa_calib",), "CoCoLoFa test": ("cocolofa_test",), "MAFALDA": ("mafalda_test",),
        "generated eval": ("eval", "eval_none"), "generated calib": ("calib_fallacy", "calib_none")}


def top1(S, G, bonus):
    return float(G[np.arange(len(G)), (S + bonus).argmax(1)].mean())


def top3(S, G, bonus):
    idx = np.argsort(-(S + bonus), 1)[:, :3]
    return float(np.take_along_axis(G, idx, 1).any(1).mean())


def none_split(S, G, bonus, none_col):
    is_none = G[:, none_col]
    pred = (S + bonus).argmax(1)
    return float((pred[is_none] == none_col).mean()), float(G[np.arange(len(G)), pred][~is_none].mean())


if __name__ == "__main__":
    import json
    cands = json.loads((c.DATA / "scores_H_eval.json").read_text(encoding="utf-8"))["cands"]
    norm_c = [c.norm(n) for n in cands]
    none_col = norm_c.index("none")
    idx = {n: i for i, n in enumerate(cands)}
    missing = [n for n in COMMON if n not in idx]
    assert not missing, missing
    mask = np.zeros(len(cands))
    mask[[idx[n] for n in COMMON]] = 1
    mask[none_col] = 1  # None is not penalised: only obscure names are
    data = {k: c.load(*v) for k, v in SETS.items()}

    print(f"{len(COMMON)} common names (+ None get the bonus); the other {len(cands) - len(COMMON) - 1} names do not.\n")
    print("lambda sweep, top-1 hit rate (%):")
    lams = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
    print(f"  {'':22}" + "".join(f"{l:>7}" for l in lams))
    for k, (S, G) in data.items():
        print(f"  {k:22}" + "".join(f"{100 * top1(S, G, l * mask):7.1f}" for l in lams))
    S0, G0 = data["CoCoLoFa calib (fit)"]
    best = max(lams, key=lambda l: (round(top1(S0, G0, l * mask), 4), -l))
    print(f"\nlambda chosen on the CoCoLoFa calibration slice: {best}\n")

    print(f"{'set':22}{'top-1 before':>13}{'top-1 after':>13}{'top-3 before':>14}{'top-3 after':>13}{'None recall b/a':>18}{'fallacy hit b/a':>18}")
    for k, (S, G) in data.items():
        nb, fb = none_split(S, G, 0 * mask, none_col)
        na, fa = none_split(S, G, best * mask, none_col)
        print(f"{k:22}{100 * top1(S, G, 0 * mask):12.1f}%{100 * top1(S, G, best * mask):12.1f}%{100 * top3(S, G, 0 * mask):13.1f}%"
              f"{100 * top3(S, G, best * mask):12.1f}%{f'{100 * nb:.0f}% / {100 * na:.0f}%':>18}{f'{100 * fb:.0f}% / {100 * fa:.0f}%':>18}")

    print(f"\nCONTROL: {best} added to 200 random lists of {len(COMMON)} names (None also gets it), top-1 hit rate %:")
    rng = random.Random(0)
    others = [i for i in range(len(cands)) if i != none_col]
    for k in ("CoCoLoFa test", "MAFALDA", "generated eval"):
        S, G = data[k]
        vals = []
        for _ in range(200):
            m = np.zeros(len(cands))
            m[rng.sample(others, len(COMMON))] = 1
            m[none_col] = 1
            vals.append(100 * top1(S, G, best * m))
        real = 100 * top1(S, G, best * mask)
        print(f"  {k:16} real list {real:5.1f}   random lists: mean {np.mean(vals):5.1f}  sd {np.std(vals):4.1f}  max {max(vals):5.1f}   real list beats {100 * np.mean([real > v for v in vals]):.0f}% of random lists")
