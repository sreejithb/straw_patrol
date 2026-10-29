"""Cut the MAFALDA gold texts into their labelled spans, one scoring item per span (local only: MAFALDA text).

  python build_mafalda_spans.py          # writes external/mafalda_spans.jsonl
A span is the text between a label's start and end positions. Spans with the same positions but different labels are merged into one
item whose answer key holds all their names. The model sees ONLY the span text (no surrounding context).
"""
import json
from pathlib import Path

from build_external import MAFALDA

ROOT = Path(__file__).parent
EXT = ROOT / "external"
MAP = dict(MAFALDA) | {"appeal to tradition": ["Appeal to Tradition"]}  # the label we added later
ARTIFACTS = {"nothing", "to clean"}


def main() -> None:
    items, skipped = [], {"unmappable type": 0, "artifact label": 0, "too short": 0}
    for i, line in enumerate((EXT / "mafalda" / "gold_standard_dataset.jsonl").read_text(encoding="utf-8").splitlines()):
        r = json.loads(line)
        labels = json.loads(r["labels"]) if isinstance(r["labels"], str) else r["labels"]
        by_pos = {}
        for a, b, typ in labels:
            if typ.lower() in ARTIFACTS:
                skipped["artifact label"] += 1
                continue
            by_pos.setdefault((a, b), []).append(typ)
        for (a, b), types in sorted(by_pos.items()):
            gold = sorted({g for t in types for g in MAP.get(t.lower(), [])})
            text = r["text"][a:b].strip()
            if not gold:
                skipped["unmappable type"] += 1
            elif len(text.split()) < 3:
                skipped["too short"] += 1
            else:
                items.append({"source": "mafalda_span", "id": f"{i}:{a}-{b}", "text": text, "gold": gold, "orig": types, "parent": i, "words": len(text.split())})
    (EXT / "mafalda_spans.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    ws = sorted(x["words"] for x in items)
    print(f"{len(items)} span items from {len({x['parent'] for x in items})} texts; skipped {skipped}")
    print(f"words per span: median {ws[len(ws) // 2]}, 10th percentile {ws[len(ws) // 10]}, 90th percentile {ws[9 * len(ws) // 10]}")
    print(f"items with more than one name in the key (merged labels on one span): {sum(len(x['orig']) > 1 for x in items)}")
    for x in items[:3]:
        print(f"  [{', '.join(x['orig'])}] {x['text'][:150]!r}")


if __name__ == "__main__":
    main()
