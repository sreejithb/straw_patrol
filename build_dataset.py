"""Generate a synthetic fallacy dataset from fallacy definitions.

Pipeline:  fallacies.jsonl -> generate (LLM) -> auto checks -> judge (LLM) -> split -> train/eval jsonl

Only the fallacy name/aliases/definition/form are sent to the model, never the book's examples,
so the output is original text. Book examples are used locally to reject near-copies.

Usage:
  python build_dataset.py --names "Straw Man Fallacy,Gambler's fallacy" --per-fallacy 6
  python build_dataset.py --all
"""
import argparse
import collections
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import prompts

load_dotenv()

ROOT = Path(__file__).parent
FALLACIES = ROOT / "data" / "fallacies.jsonl"
GEN_DIR = ROOT / "data" / "generated"
GEN_MODEL = "gpt-5.4-mini"
JUDGE_MODEL = "gpt-5.4"
EVAL_FRACTION = 0.1
MIN_WORDS, MAX_WORDS = 8, 110
MAX_ROUNDS = 3
MAX_WORDS_OVERRIDE = {"gish gallop": 170}
MIN_KEPT = 4  # fallacies below this are reported and left out of the final dataset
COPY_NGRAM = 6  # reject an argument sharing a 6-word run with a book example

# Fallacies whose mechanism cannot be shown in plain text; skipped for a text-based bot.
EXCLUDE = {"accent fallacy"}

TOPICS = ["politics", "health", "technology", "sports", "relationships", "money", "school", "work",
          "food", "science", "parenting", "travel", "religion", "environment", "law", "entertainment",
          "history", "gaming", "housing", "pets"]
FORMATS = ["tweet-length post", "forum reply", "two-person dialogue (use A:/B: or uncommon names)",
           "single sentence in an op-ed", "text message to a friend", "one speaker mid-rant, no names",
           "reply that quotes and answers someone's claim", "workplace chat message"]

client = OpenAI()

GEN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "examples",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["examples"],
            "properties": {"examples": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["argument", "explanation"],
                "properties": {"argument": {"type": "string"}, "explanation": {"type": "string"}},
            }}},
        },
    },
}

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "required": ["analysis", "commits_target", "more_obvious_other_fallacy", "realistic",
                         "leaks_label", "explanation_ok", "speaker_commits", "templated", "ambiguous", "issues"],
            "properties": {
                "analysis": {"type": "string"},
                "commits_target": {"type": "boolean"},
                "more_obvious_other_fallacy": {"type": "string"},
                "realistic": {"type": "boolean"},
                "leaks_label": {"type": "boolean"},
                "explanation_ok": {"type": "boolean"},
                "speaker_commits": {"type": "boolean"},
                "templated": {"type": "boolean"},
                "ambiguous": {"type": "boolean"},
                "issues": {"type": "string"},
            },
        },
    },
}


def chat_json(model: str, system: str, user: str, schema: dict) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=schema,
    )
    return json.loads(resp.choices[0].message.content)


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def load_fallacies() -> list[dict]:
    return [json.loads(l) for l in FALLACIES.read_text(encoding="utf-8").splitlines() if l.strip()]


def guidance_line(name: str) -> str:
    g = prompts.GUIDANCE.get(name.lower())
    return f"Special guidance for this fallacy: {g}\n" if g else ""


def assignments(name: str, n: int, round_no: int = 0) -> str:
    rng = random.Random(f"{name}/{round_no}")  # deterministic per fallacy and round
    topics = rng.sample(TOPICS, n)
    formats = [FORMATS[i % len(FORMATS)] for i in rng.sample(range(len(FORMATS)), min(n, len(FORMATS)))]
    formats += [rng.choice(FORMATS) for _ in range(n - len(formats))]
    return "\n".join(f"{i + 1}. topic hint: {t}; format: {f}" for i, (t, f) in enumerate(zip(topics, formats)))


def generate(fallacy: dict, n: int, round_no: int = 0) -> list[dict]:
    system = prompts.GENERATE_SYSTEM.format(avoid="\n".join(f"- {a}" for a in prompts.AVOID))
    user = prompts.GENERATE_USER.format(
        name=fallacy["name"], aka=fallacy["aka"] or "n/a",
        description=fallacy["description"], logical_form=fallacy["logical_form"] or "n/a", n=n,
        assignments=assignments(fallacy["name"], n, round_no), guidance=guidance_line(fallacy["name"]))
    items = chat_json(GEN_MODEL, system, user, GEN_SCHEMA)["examples"]
    return [{"fallacy": fallacy["name"], **it} for it in items]


def auto_check(item: dict, fallacy: dict, book_grams: set, seen: set) -> str:
    """Return a rejection reason, or '' if the item passes."""
    arg = item["argument"].strip()
    words = len(arg.split())
    if not MIN_WORDS <= words <= MAX_WORDS_OVERRIDE.get(fallacy["name"].lower(), MAX_WORDS):
        return f"length {words}"
    if not re.match(r"this is an? ", item["explanation"].lower()):
        return "explanation prefix"
    if re.match(r"\s*(tweet|comment|reply|op-ed|post|text|message)\b[^:]{0,30}:", arg.lower()):
        return "scaffold label"
    if "fallacy" in arg.lower() or fallacy["name"].lower() in arg.lower():
        return "leaks label"
    g = ngrams(arg, COPY_NGRAM)
    if g & book_grams:
        return "copies book text"
    key = " ".join(re.findall(r"[a-z]+", arg.lower()))[:80]
    if key in seen:
        return "duplicate"
    seen.add(key)
    return ""


def judge(item: dict, fallacy: dict) -> dict:
    user = prompts.JUDGE_USER.format(name=fallacy["name"], description=fallacy["description"],
                                     guidance=guidance_line(fallacy["name"]), argument=item["argument"], explanation=item["explanation"])
    return chat_json(JUDGE_MODEL, prompts.JUDGE_SYSTEM, user, JUDGE_SCHEMA)


def passes(v: dict) -> bool:
    return (v["commits_target"] and v["realistic"] and not v["leaks_label"]
            and v["explanation_ok"] and v["speaker_commits"] and not v["templated"] and not v["ambiguous"] and not v["more_obvious_other_fallacy"].strip())


def process(fallacy: dict, n: int, target: int, book_grams: set) -> list[dict]:
    """Generate in rounds until `target` examples are kept (max MAX_ROUNDS)."""
    seen: set = set()
    out: list[dict] = []
    for round_no in range(MAX_ROUNDS):
        items = generate(fallacy, n, round_no)
        reasons = [auto_check(it, fallacy, book_grams, seen) for it in items]
        with ThreadPoolExecutor(4) as pool:
            verdicts = list(pool.map(lambda it_r: None if it_r[1] else judge(it_r[0], fallacy), zip(items, reasons)))
        for item, reason, verdict in zip(items, reasons, verdicts):
            keep = not reason and passes(verdict)
            out.append({**item, "keep": keep, "round": round_no,
                        "reject_reason": reason or ("" if keep else verdict["issues"] or "judge"),
                        "judge": verdict})
        if sum(r["keep"] for r in out) >= target:
            break
    print(f"  {fallacy['name']}: {sum(r['keep'] for r in out)}/{len(out)} kept", flush=True)
    return out


def to_chat(row: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": "You are a logical-fallacy expert. Identify the fallacy in the argument and explain it."},
        {"role": "user", "content": row["argument"]},
        {"role": "assistant", "content": f"Fallacy: {row['fallacy']}\n{row['explanation']}"},
    ]}


def split_and_write(rows: list[dict]) -> None:
    random.Random(0).shuffle(rows)
    n_eval = max(1, int(len(rows) * EVAL_FRACTION))
    for name, part in (("eval", rows[:n_eval]), ("train", rows[n_eval:])):
        path = GEN_DIR / f"{name}.jsonl"
        path.write_text("".join(json.dumps(to_chat(r), ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
        print(f"wrote {len(part)} rows to {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", help="comma-separated fallacy names (exact)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--per-fallacy", type=int, default=8, help="examples requested per round")
    ap.add_argument("--target", type=int, default=6, help="stop generating once this many are kept")
    ap.add_argument("--out", default="raw.jsonl")
    ap.add_argument("--merge", help="existing raw file: replace rows of the fallacies rerun here, then re-split everything")
    args = ap.parse_args()

    fallacies = [f for f in load_fallacies() if f["name"].lower() not in EXCLUDE]
    if args.names:
        wanted = {n.strip().lower() for n in args.names.split(",")}
        fallacies = [f for f in fallacies if f["name"].lower() in wanted]
        missing = wanted - {f["name"].lower() for f in fallacies}
        if missing:
            print(f"WARNING: no fallacy matched: {sorted(missing)}")
    elif not args.all:
        raise SystemExit("pass --names or --all")

    book_grams: set = set()
    for f in load_fallacies():
        for ex in f["examples"]:
            book_grams |= ngrams(ex["text"], COPY_NGRAM)

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda f: process(f, args.per_fallacy, args.target, book_grams), fallacies))

    rows = [r for res in results for r in res]
    (GEN_DIR / args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{sum(r['keep'] for r in rows)}/{len(rows)} kept in this run")

    if args.merge:
        rerun = {f["name"] for f in fallacies}
        old_rows = [json.loads(l) for l in (GEN_DIR / args.merge).read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in old_rows if r["fallacy"] not in rerun] + rows
        (GEN_DIR / args.merge).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    if args.all or args.merge:
        yield_by = collections.Counter()
        for r in rows:
            yield_by[r["fallacy"]] += r["keep"]
        low = {n: k for n, k in yield_by.items() if k < MIN_KEPT}
        if low:
            print(f"WARNING: {len(low)} fallacies below {MIN_KEPT} kept (left out of dataset): {low}")
        (GEN_DIR / "yield.json").write_text(json.dumps(dict(yield_by), indent=1, ensure_ascii=False), encoding="utf-8")
        split_and_write([r for r in rows if r["keep"] and r["fallacy"] not in low])

if __name__ == "__main__":
    main()
