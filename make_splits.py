"""Build the clean train / calib / test folder  data/v6/  from every source we have.

  python make_splits.py

  data/v6/train/{generated,cocolofa}.jsonl
  data/v6/calib/{generated,cocolofa}.jsonl
  data/v6/test/{generated,cocolofa,mafalda}.jsonl
  data/v6/MANIFEST.json   counts, label coverage, file hashes, the results of the leak checks
  data/v6/README.md       what is in each split and the rules

One record per line, the same shape for every source:
  {"id", "source", "subset", "split", "text", "label", "gold", "answer", "group"}
  label  our label name or "None"           gold    names that count as correct (several for the public data)
  answer the full assistant answer (with explanation) for generated data, null for public data (explanations come later)
  group  records sharing a group must never be split apart (a CoCoLoFa article)

Rules: the public datasets keep their OWN splits (CoCoLoFa train/dev/test are separated by news article; MAFALDA gold is test only);
a text appears in one split only; train records that overlap any calib/test text (identical, or a shared 8-word run) are dropped.
The public files are local only (licences): data/v6/ is gitignored except the manifest, README and this script.
"""
import collections
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).parent
GEN = ROOT / "data" / "generated"
EXT = ROOT / "external"
OUT = ROOT / "data" / "v6"
SEED, CAP_TYPE, CAP_NONE, NGRAM = 0, 150, 300, 8

# CoCoLoFa type -> (canonical label for training, names that count as correct when testing)
AUTH = ["Appeal to false authority", "Blind Authority Fallacy", "Anonymous Authority"]
COCO = {"appeal to authority": ("Appeal to false authority", AUTH), "appeal to majority": ("Appeal to Common Belief", ["Appeal to Common Belief"]),
        "appeal to nature": ("Appeal to Nature", ["Appeal to Nature"]), "appeal to worse problems": ("Relative privation", ["Relative privation"]),
        "false dilemma": ("False dilemma", ["False dilemma"]), "hasty generalization": ("Hasty Generalization", ["Hasty Generalization"]),
        "slippery slope": ("Slippery Slope", ["Slippery Slope"]), "appeal to tradition": ("Appeal to Tradition", ["Appeal to Tradition"]),
        "none": ("None", ["None"])}


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def key(t):
    return " ".join(re.findall(r"[a-z0-9']+", t.lower()))


def grams(t):
    w = re.findall(r"[a-z0-9']+", t.lower())
    return {tuple(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def rec(source, subset, split, rid, text, label, gold, answer=None, group=None):
    return {"id": f"{subset}:{rid}", "source": source, "subset": subset, "split": split, "text": text.strip(), "label": label,
            "gold": gold, "answer": answer, "group": group}


def generated(files, split):
    out = []
    for f in files:
        for i, r in enumerate(read(GEN / f)):
            m = r["messages"]
            label = m[2]["content"].split("\n")[0][len("Fallacy: "):].strip()
            out.append(rec("generated", f.replace(".jsonl", ""), split, i, m[1]["content"], label, [label], answer=m[2]["content"]))
    return out


def cocolofa_slice(files, split, news_of):
    out = []
    for f in files:
        for r in read(EXT / f):
            label = r["gold"][0] if r["gold"] == ["None"] or len(r["gold"]) == 1 and r["gold"][0] == "Appeal to Tradition" else None
            lab = next((c for c, (cn, g) in COCO.items() if g == r["gold"]), None)
            out.append(rec("cocolofa", f.replace(".jsonl", ""), split, r["id"], r["text"], COCO[lab][0] if lab else r["gold"][0], r["gold"],
                           group=f"cocolofa:{news_of.get(r['id'])}"))
    return out


def main() -> None:
    rng = random.Random(SEED)
    news_of = {}
    coco = {}
    for sp in ("train", "dev", "test"):
        coco[sp] = json.loads((EXT / "cocolofa" / f"{sp}.json").read_text(encoding="utf-8"))
        for a in coco[sp]:
            for c in a["comments"]:
                news_of[c["id"]] = a["id"]

    # ---- train: generated + CoCoLoFa official train split (top-level comments, capped per type)
    train = generated(["train.jsonl", "train_extra.jsonl", "train_none.jsonl", "train_real.jsonl"], "train")
    pool = collections.defaultdict(list)
    for a in coco["train"]:
        for c in a["comments"]:
            if not c.get("respond_to") and c["fallacy"] in COCO and 12 <= len(c["comment"].split()) <= 160:
                pool[c["fallacy"]].append((a["id"], c))
    for typ, items in sorted(pool.items()):
        rng.shuffle(items)
        for aid, c in items[:CAP_NONE if typ == "none" else CAP_TYPE]:
            cn, gold = COCO[typ]
            train.append(rec("cocolofa", "cocolofa_train", "train", c["id"], c["comment"], cn, gold, group=f"cocolofa:{aid}"))

    # ---- calib / test: the slices we already scored with runs H and I (so old and new numbers stay comparable)
    calib = generated(["calib_fallacy.jsonl", "calib_none.jsonl"], "calib") + cocolofa_slice(["cocolofa_calib.jsonl", "cocolofa_tradition_calib.jsonl"], "calib", news_of)
    test = generated(["eval.jsonl", "eval_none.jsonl", "eval_real.jsonl"], "test") + cocolofa_slice(["cocolofa_test.jsonl", "cocolofa_tradition_test.jsonl"], "test", news_of)
    for r in read(EXT / "mafalda_test.jsonl"):
        test.append(rec("mafalda", "mafalda_gold", "test", r["id"], r["text"], r["gold"][0], r["gold"]))

    # ---- leak checks: nothing in train may duplicate or overlap a calib/test text
    held_keys = {key(r["text"]) for r in calib + test}
    held_grams = set()
    for r in calib + test:
        held_grams |= grams(r["text"])
    kept, dropped = [], collections.Counter()
    for r in train:
        if key(r["text"]) in held_keys:
            dropped[(r["subset"], "identical text")] += 1
        elif grams(r["text"]) & held_grams:
            dropped[(r["subset"], f"shares an {NGRAM}-word run")] += 1
        else:
            kept.append(r)
    train = kept
    split_keys = {s: {key(r["text"]) for r in rs} for s, rs in (("train", train), ("calib", calib), ("test", test))}
    dup = {f"{a}&{b}": len(split_keys[a] & split_keys[b]) for a, b in (("train", "calib"), ("train", "test"), ("calib", "test"))}
    groups = {s: {r["group"] for r in rs if r["group"] and r["source"] == "cocolofa"} for s, rs in (("train", train), ("calib", calib), ("test", test))}
    shared = {f"{a}&{b}": len(groups[a] & groups[b]) for a, b in (("train", "calib"), ("train", "test"), ("calib", "test"))}
    assert not any(dup.values()), dup
    assert not any(shared.values()), shared

    # ---- write
    manifest = {"seed": SEED, "cocolofa_cap_per_type": CAP_TYPE, "cocolofa_cap_none": CAP_NONE, "splits": {}, "dropped_from_train": {f"{a}: {b}": n for (a, b), n in dropped.items()},
                "leak_checks": {"identical texts shared between splits": dup, "CoCoLoFa articles shared between splits": shared}}
    for name, rows in (("train", train), ("calib", calib), ("test", test)):
        manifest["splits"][name] = {}
        for src in sorted({r["source"] for r in rows}):
            part = [r for r in rows if r["source"] == src]
            p = OUT / name / f"{src}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
            n_none = sum(r["label"] == "None" for r in part)
            manifest["splits"][name][src] = {"total": len(part), "fallacy": len(part) - n_none, "none": n_none,
                                              "labels": len({r["label"] for r in part if r["label"] != "None"}),
                                              "subsets": dict(collections.Counter(r["subset"] for r in part)),
                                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    for name, srcs in manifest["splits"].items():
        print(f"\n{name}")
        for src, m in srcs.items():
            print(f"  {src:10}{m['total']:6}   fallacy {m['fallacy']:5}  None {m['none']:4}  labels {m['labels']:3}   {m['subsets']}")
        print(f"  {'total':10}{sum(m['total'] for m in srcs.values()):6}")
    print("\ndropped from train because of overlap with calib/test:", manifest["dropped_from_train"] or "none")
    print("leak checks (all must be 0):", manifest["leak_checks"])


README = """# data/v6: train / calib / test

| split | generated | CoCoLoFa | MAFALDA |
|---|---|---|---|
| train | all earlier generated training files | official train split, top-level comments, capped per type | not used |
| calib | generated calibration set | official dev split slice | not used |
| test  | generated eval sets | official test split slice | gold set (out-of-domain) |

Not in any split (reserved): the rest of CoCoLoFa train, MAFALDA's 9,545 unlabeled texts.

Rules
- Public datasets keep their own splits; a CoCoLoFa article never appears in two splits.
- A text appears in one split only; train records that duplicate or share an 8-word run with a calib/test text are dropped.
- CoCoLoFa types are mapped to one canonical label of ours for training (appeal to authority -> Appeal to false authority, appeal to majority -> Appeal to Common Belief,
  appeal to worse problems -> Relative privation, appeal to tradition -> Appeal to Tradition); at test time several names count as correct (`gold`).
- Public data have no explanation (`answer` is null): explanations are generated later, when the training file is built.
- Public-data files are local only (CoCoLoFa states no licence, MAFALDA is CC-BY-SA); only MANIFEST.json and this README are tracked in git.

Build: `python make_splits.py` (deterministic, seed 0).
"""

if __name__ == "__main__":
    main()
