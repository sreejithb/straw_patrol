"""Build train_real.jsonl / eval_real.jsonl from the raw files, without any API call.

Same as the end of build_realistic.py, plus one relaxation for the 'mention' unit: the judge rejected many texts only because they
did not NAME a fallacy, although it found no fallacy in them. Those that quote or reply to someone else's claim and rebut it
(a quote, an @-reply, 'saying ...') are real 'mention, not use' negatives, so they are accepted as subtype 'rebuttal'
(only when the judge found no fallacy and rated the text realistic with an accurate reason).
"""
import collections
import json
import random
import re

import build_dataset as bd
import build_nofallacy as bn
import build_realistic as br

PAT = re.compile(r"[“\"”>]|@\w+|\b(saying|says|said|claim|claims|argument|arguing|reply|point|accus|assum|implies|calling)\b", re.I)


def load(name):
    p = bd.GEN_DIR / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    labels = sorted({r["messages"][2]["content"].split("\n")[0][len("Fallacy: "):].strip() for f in ("train.jsonl", "train_extra.jsonl") for r in load(f)}
                    | {"Appeal to Tradition"})
    common = set(br.COMMON) | {"Appeal to Tradition"}
    need = {n: br.N_ALL + (br.N_COMMON if n in common else 0) for n in labels}
    taken, fallacy = collections.Counter(), []
    for row in load("real_fallacy_raw.jsonl"):
        if row["keep"] and taken[row["fallacy"]] < need.get(row["fallacy"], 0):
            taken[row["fallacy"]] += 1
            fallacy.append(("F/" + row["fallacy"], bd.to_chat(row)))
    none_rows = load("real_none_raw.jsonl")
    nones = [("N/%s/%s" % (r["kind"], r["subtype"]), bn.to_chat(r)) for r in none_rows if r["keep"]]
    relaxed = [r for r in none_rows if r["subtype"] == "named" and not r["keep"] and r["judge"] and not r["judge"]["contains_fallacy"]
               and r["judge"]["realistic"] and r["judge"]["reason_ok"] and PAT.search(r["text"])]
    nones += [("N/mention/rebuttal", bn.to_chat(r)) for r in relaxed]
    print(f"fallacy {len(fallacy)} (short of target by {sum(need.values()) - len(fallacy)}), none {len(nones)} (of which {len(relaxed)} relaxed 'rebuttal')")

    rng = random.Random(0)
    groups = collections.defaultdict(list)
    for g, c in fallacy + nones:
        groups[g].append(c)
    train, ev = [], []
    for g in groups.values():
        rng.shuffle(g)
        k = max(1, round(len(g) * br.HOLDOUT)) if len(g) >= 4 else 0
        ev += g[:k]
        train += g[k:]
    rng.shuffle(train)
    for name, part in (("train_real", train), ("eval_real", ev)):
        (bd.GEN_DIR / f"{name}.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in part), encoding="utf-8")
        n_none = sum(c["messages"][2]["content"].startswith("Fallacy: None") for c in part)
        print(f"wrote {len(part)} rows to {name}.jsonl ({n_none} None)")


if __name__ == "__main__":
    main()
