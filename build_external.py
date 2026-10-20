"""Turn the external datasets (kept local in external/, never uploaded) into scoring files with OUR label names.

  external/cocolofa_calib.jsonl   15 comments per mappable type + 60 'none', from CoCoLoFa dev   (calibration)
  external/cocolofa_test.jsonl    the same sizes, from CoCoLoFa test                              (final real-style test)
  external/mafalda_test.jsonl     MAFALDA gold standard (200 real texts), mapped                  (final real-text check)

Record: {"source", "id", "text", "gold": [names that count as correct], "orig": original label(s)}.  gold == ["None"] means no fallacy.
A prediction is a hit when it is in `gold`; top-k is a hit when any gold name is in the top k.

Sources: CoCoLoFa (Crowd-AI-Lab/cocolofa, Yang et al.), MAFALDA (Helwe et al., CC-BY-SA). No license is stated for CoCoLoFa:
keep the files local, use them for evaluation and calibration only, and cite the papers.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
EXT = ROOT / "external"
AUTH = ["Appeal to false authority", "Blind Authority Fallacy", "Anonymous Authority"]
COCO = {
    "appeal to authority": AUTH, "appeal to majority": ["Appeal to Common Belief"], "appeal to nature": ["Appeal to Nature"],
    "appeal to worse problems": ["Relative privation"], "false dilemma": ["False dilemma"],
    "hasty generalization": ["Hasty Generalization"], "slippery slope": ["Slippery Slope"],
    # "appeal to tradition": no counterpart among our labels -> not used
}
MAFALDA = {
    "hasty generalization": ["Hasty Generalization"], "causal oversimplification": ["Causal Reductionism", "Insignificant Cause"],
    "appeal to ridicule": ["Appeal to ridicule"], "false dilemma": ["False dilemma"],
    "ad hominem": ["Abusive Ad Hominem", "Circumstantial Ad Hominem"], "ad populum": ["Appeal to Common Belief"],
    "false causality": ["Post hoc", "Cum Hoc Ergo Propter Hoc", "False Effect"], "straw man": ["Straw Man Fallacy"],
    "false analogy": ["Weak analogy", "Faulty Comparison", "Extended Analogy"], "slippery slope": ["Slippery Slope"],
    "circular reasoning": ["Circular Reasoning", "Circular Definition"], "appeal to fear": ["Appeal to Fear"],
    "appeal to nature": ["Appeal to Nature"], "appeal to (false) authority": AUTH, "guilt by association": ["Guilt by Association"],
    "appeal to worse problems": ["Relative privation"], "appeal to anger": ["Appeal to anger"], "tu quoque": ["Tu Quoque"],
    "fallacy of division": ["Fallacy of Division"], "appeal to pity": ["Appeal to Pity"],
    # not mapped: equivocation (not one of our trained labels), appeal to tradition, appeal to positive emotion
}
ARTIFACTS = {"nothing", "to clean"}  # annotation leftovers in MAFALDA, not fallacy labels
PER_TYPE, N_NONE = 15, 60


def write(name, rows):
    (EXT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    kinds = {}
    for r in rows:
        kinds[r["gold"][0]] = kinds.get(r["gold"][0], 0) + 1
    print(f"{name}: {len(rows)} rows", dict(sorted(kinds.items(), key=lambda kv: -kv[1])[:4]), "...")


def cocolofa(split, name, seed):
    comments = [c for a in json.loads((EXT / "cocolofa" / f"{split}.json").read_text(encoding="utf-8")) for c in a["comments"]
                if not c.get("respond_to")]  # top-level comments only: replies lack the comment they answer
    rng = random.Random(seed)
    rows = []
    for label, gold in list(COCO.items()) + [("none", ["None"])]:
        pool = [c for c in comments if c["fallacy"] == label]
        for c in rng.sample(pool, N_NONE if label == "none" else PER_TYPE):
            rows.append({"source": "cocolofa", "id": c["id"], "text": c["comment"].strip(), "gold": gold, "orig": [label]})
    rng.shuffle(rows)
    write(name, rows)


def mafalda():
    rows, skipped = [], {"unmappable": 0, "artifact only": 0}
    for i, line in enumerate((EXT / "mafalda" / "gold_standard_dataset.jsonl").read_text(encoding="utf-8").splitlines()):
        r = json.loads(line)
        labels = json.loads(r["labels"]) if isinstance(r["labels"], str) else r["labels"]
        orig = [x[2] for x in labels]
        real = [o for o in orig if o.lower() not in ARTIFACTS]
        if not orig:
            gold = ["None"]
        elif not real:
            skipped["artifact only"] += 1
            continue
        else:
            gold = sorted({g for o in real for g in MAFALDA.get(o.lower(), [])})  # MAFALDA mixes upper and lower case
            if not gold:
                skipped["unmappable"] += 1
                continue
        rows.append({"source": "mafalda", "id": i, "text": r["text"].strip(), "gold": gold, "orig": orig})
    print("mafalda skipped:", skipped)
    write("mafalda_test.jsonl", rows)


def cocolofa_tradition(split, name, seed):
    """Appeal to Tradition (a label we only just added): kept apart so the older sets stay identical for comparisons."""
    comments = [c for a in json.loads((EXT / "cocolofa" / f"{split}.json").read_text(encoding="utf-8")) for c in a["comments"]
                if not c.get("respond_to") and c["fallacy"] == "appeal to tradition"]
    rows = [{"source": "cocolofa", "id": c["id"], "text": c["comment"].strip(), "gold": ["Appeal to Tradition"], "orig": ["appeal to tradition"]}
            for c in random.Random(seed).sample(comments, PER_TYPE)]
    write(name, rows)


if __name__ == "__main__":
    cocolofa_tradition("dev", "cocolofa_tradition_calib.jsonl", 3)
    cocolofa_tradition("test", "cocolofa_tradition_test.jsonl", 4)
    cocolofa("dev", "cocolofa_calib.jsonl", 1)
    cocolofa("test", "cocolofa_test.jsonl", 2)
    mafalda()
