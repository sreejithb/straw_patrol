"""Re-score saved predictions with alias groups (no model needed).

  python rescore.py                                  # all data/lora_*_preds.jsonl, uses data/alias_groups.json
  python rescore.py --aliases data/alias_groups_draft.json A B

alias_groups.json maps a gold label to the other fallacy names that also count as correct for it.
Strict = exact name match (as in evaluate.py). Alias = strict, or the answer is in the gold label's alias list.
"""
import argparse
import json
from pathlib import Path

from evaluate import first_line_name, norm

DATA = Path(__file__).parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aliases", default=str(DATA / "alias_groups.json"))
    ap.add_argument("runs", nargs="*", help="run names such as A B (default: every data/lora_*_preds.jsonl)")
    args = ap.parse_args()

    aliases = {norm(g): {norm(a) for a in alts} for g, alts in json.loads(Path(args.aliases).read_text(encoding="utf-8")).items()}
    runs = args.runs or sorted(p.name[5:-12] for p in DATA.glob("lora_*_preds.jsonl"))
    print(f"aliases: {args.aliases} ({sum(len(v) for v in aliases.values())} pairs)")
    print(f"{'run':<8}{'strict':>14}{'with aliases':>16}")
    for run in runs:
        rows = [json.loads(l) for l in (DATA / f"lora_{run}_preds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len(rows)
        strict = sum(norm(first_line_name(r["answer"])) == norm(r["label"]) for r in rows)
        alias = sum(norm(first_line_name(r["answer"])) == norm(r["label"])
                    or norm(first_line_name(r["answer"])) in aliases.get(norm(r["label"]), ()) for r in rows)
        print(f"{run:<8}{f'{strict}/{n} {100 * strict / n:.1f}%':>14}{f'{alias}/{n} {100 * alias / n:.1f}%':>16}")


if __name__ == "__main__":
    main()
