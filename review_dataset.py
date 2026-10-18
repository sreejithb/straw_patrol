"""Second-pass quality review of the generated dataset (cheap classify -> strong rewrite of mismatches).

1. A cheap model picks the best-matching fallacy for every kept example from the FULL list of fallacies.
   If the example's label is not its best match (or in its 'also fits' list) the example is flagged.
2. A strong model looks at each flagged example and decides: keep / rewrite / drop.
3. Rewrites are re-checked (auto checks + cheap classifier must now agree with the label).
Every decision is logged to data/generated/review_log.jsonl.

Usage:
  python review_dataset.py --limit 60      # pilot on a random sample, prints token usage, writes nothing else
  python review_dataset.py                 # full run, rewrites train/eval splits
"""
import argparse
import collections
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import openai
import prompts
from build_dataset import (COPY_NGRAM, GEN_DIR, MIN_KEPT, MAX_WORDS_OVERRIDE, auto_check, client,
                           guidance_line, load_fallacies, ngrams, split_and_write)

CLASSIFY_MODEL = "gpt-5.4-mini"
REVIEW_MODEL = "gpt-5.5"
WORKERS = 8
CACHE = GEN_DIR / "review_cache.jsonl"  # one finished row per line; lets an interrupted run resume

usage = collections.defaultdict(lambda: {"in": 0, "cached": 0, "out": 0, "calls": 0})
lock = threading.Lock()


def call(model: str, system: str, user: str, schema: dict) -> dict:
    for attempt in range(8):
        try:
            resp = client.chat.completions.create(
                model=model, response_format=schema,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            break
        except openai.RateLimitError:
            time.sleep(2 ** attempt)  # tokens-per-minute limit; back off and retry
    else:
        raise RuntimeError("rate limited too many times")
    u = resp.usage
    with lock:
        s = usage[model]
        s["in"] += u.prompt_tokens
        s["cached"] += getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        s["out"] += u.completion_tokens
        s["calls"] += 1
    return json.loads(resp.choices[0].message.content)


def schema(name: str, props: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": list(props), "properties": props}}}


def build(fallacies: list[dict]):
    names = [f["name"] for f in fallacies]
    catalogue = "\n".join(f"- {f['name']}: {f['description'][:150]}" for f in fallacies)
    classify_system = (
        "You classify a short argument by the logical fallacy it commits. Choose from this exact list:\n"
        f"{catalogue}\n\n"
        "Return `best`: the single best-matching fallacy name. Return `also_fits`: other names from the list that "
        "genuinely describe the SAME flaw in this argument (near-synonyms or sub-types), at most 3, possibly empty. "
        "If the argument does not clearly commit any listed fallacy, set `clear` to false.")
    classify_schema = schema("classification", {
        "best": {"type": "string", "enum": names},
        "also_fits": {"type": "array", "items": {"type": "string", "enum": names}},
        "clear": {"type": "boolean"}})
    review_schema = schema("review", {
        "analysis": {"type": "string"},
        "action": {"type": "string", "enum": ["keep", "rewrite", "drop"]},
        "argument": {"type": "string"},
        "explanation": {"type": "string"}})
    return names, classify_system, classify_schema, review_schema


REVIEW_SYSTEM = """You are a senior logic instructor curating a training set. An example is labelled with a target fallacy but a
classifier thinks it fits a different one better. Decide:
- keep: the label is actually fine (the classifier is wrong, or the fallacies genuinely overlap). Leave text unchanged.
- rewrite: the example is salvageable. Write a NEW argument that clearly and unambiguously commits the TARGET fallacy, keeping the
  topic if possible. Follow the rules below. Also write the explanation.
- drop: the target fallacy cannot be shown convincingly in a short text, or a rewrite would be contrived.
`analysis` comes first: 2-3 sentences on what the example actually commits and whether it can be fixed.
For keep/drop return empty strings for argument and explanation.

Rules for rewritten examples:
{avoid}"""

REVIEW_USER = """Target fallacy: {name}
Definition: {description}
{guidance}
Classifier thinks it is: {alt_name} -- {alt_desc}

Current argument:
{argument}

Current explanation:
{explanation}"""


def run(rows: list, limit, dry: bool) -> None:
    fallacies = load_fallacies()
    by_name = {f["name"]: f for f in fallacies}
    names, classify_system, classify_schema, review_schema = build(fallacies)
    review_system = REVIEW_SYSTEM.format(avoid="\n".join(f"- {a}" for a in prompts.AVOID))

    book_grams: set = set()
    for f in fallacies:
        for ex in f["examples"]:
            book_grams |= ngrams(ex["text"], COPY_NGRAM)

    def classify(arg: str) -> dict:
        return call(CLASSIFY_MODEL, classify_system, f"Argument:\n{arg}", classify_schema)

    def agrees(label: str, c: dict) -> bool:
        return label == c["best"] or label in c["also_fits"]

    def process(row: dict) -> dict:
        key = row["fallacy"] + "|" + row["argument"]
        if key in done:
            return done[key]
        result = process_uncached(row)
        with lock:
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "row": result}, ensure_ascii=False) + "\n")
        return result

    def process_uncached(row: dict) -> dict:
        label = row["fallacy"]
        c = classify(row["argument"])
        out = {**row, "review": {"classifier_best": c["best"], "also_fits": c["also_fits"], "status": "unchanged"}}
        if agrees(label, c):
            return out
        f = by_name[label]
        alt = by_name[c["best"]]
        user = REVIEW_USER.format(name=label, description=f["description"], guidance=guidance_line(label),
                                  alt_name=alt["name"], alt_desc=alt["description"][:250],
                                  argument=row["argument"], explanation=row["explanation"])
        r = call(REVIEW_MODEL, review_system, user, review_schema)
        out["review"]["reason"] = r["analysis"]
        if r["action"] == "keep":
            out["review"]["status"] = "kept_after_review"
        elif r["action"] == "drop":
            out["review"]["status"] = "dropped"
        else:
            new = {"argument": r["argument"].strip(), "explanation": r["explanation"].strip()}
            fail = auto_check(new, f, book_grams, set())
            c2 = None if fail else classify(new["argument"])
            if fail or not agrees(label, c2):
                out["review"]["status"] = "dropped"
                out["review"]["reason"] += f" | rewrite rejected: {fail or 'classifier says ' + c2['best']}"
            else:
                out["review"].update(status="rewritten", old_argument=row["argument"],
                                     old_explanation=row["explanation"])
                out.update(new)
        return out

    done = {}
    if CACHE.exists() and not dry:
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["key"]] = d["row"]
        print(f"resuming: {len(done)} rows already reviewed")
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(process, rows))

    status = collections.Counter(r["review"]["status"] for r in results)
    print("status:", dict(status))
    for model, s in usage.items():
        print(f"{model}: {s['calls']} calls, in={s['in']:,} (cached {s['cached']:,}), out={s['out']:,}")
    (GEN_DIR / "review_log.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")

    if dry:
        return
    final = [r for r in results if r["review"]["status"] != "dropped"]
    per = collections.Counter(r["fallacy"] for r in final)
    low = {n for n, k in per.items() if k < MIN_KEPT}
    if low:
        print(f"{len(low)} fallacies fell below {MIN_KEPT} after review and are left out: {sorted(low)}")
    split_and_write([r for r in final if r["fallacy"] not in low])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="pilot: review a random sample of this many rows; no splits written")
    ap.add_argument("--raw", default="full_raw.jsonl")
    args = ap.parse_args()
    y = json.loads((GEN_DIR / "yield.json").read_text())
    rows = [json.loads(l) for l in (GEN_DIR / args.raw).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["keep"] and y.get(r["fallacy"], 0) >= MIN_KEPT]
    print(f"{len(rows)} kept rows in dataset")
    if args.limit:
        random.Random(1).shuffle(rows)
        rows = rows[:args.limit]
    run(rows, args.limit, dry=bool(args.limit))


if __name__ == "__main__":
    main()
