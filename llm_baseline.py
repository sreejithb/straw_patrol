"""Zero-shot LLM baseline: pick the best fallacy name (or None) for every text, from the same 186-name label space.

  python llm_baseline.py --limit 4                 # smoke test
  python llm_baseline.py                           # all sets
  python llm_baseline.py --probe                   # memorisation probe (does the model continue benchmark texts verbatim?)
Each text goes to the OpenAI API (the model sees name + one-sentence definition for every label, like a human annotator would).
Writes data/llm_<model>_<set>.json and prints top-1 / top-3 (best + up to two 'also fits') against the same answer keys.
"""
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

import build_dataset as bd
import build_realistic as br
from evaluate import norm
from score_candidates import candidate_names, read_set

ROOT = Path(__file__).parent
MODEL = "gpt-5.4"
SETS = {"eval": "data/generated/eval.jsonl", "eval_none": "data/generated/eval_none.jsonl",
        "cocolofa_calib": "external/cocolofa_calib.jsonl", "cocolofa_test": "external/cocolofa_test.jsonl",
        "mafalda_test": "external/mafalda_test.jsonl",
        "cocolofa_tradition_calib": "external/cocolofa_tradition_calib.jsonl", "cocolofa_tradition_test": "external/cocolofa_tradition_test.jsonl"}


def label_menu():
    names = candidate_names()
    defs = {f["name"]: f["description"] for f in bd.load_fallacies()} | {f["name"]: f["description"] for f in br.NEW_FALLACIES}
    first = lambda s: re.split(r"(?<=[.!?])\s", " ".join(s.split()))[0][:170]
    lines = [f"- {n}" + (f": {first(defs[n])}" if n in defs else "") for n in names if n != "None"]
    return names, "\n".join(lines)


NAMES, MENU = label_menu()
SYSTEM = f"""You are an expert in informal logic. You read a social-media or news comment and decide which logical fallacy, if any, the commenter commits.
Choose ONLY from this list (name: short definition):
{MENU}

Rules:
- If the comment contains no logical fallacy (a question, a plain opinion with reasons, an emotional but fair comment, a rebuttal of someone else's fallacy), answer "None".
- Judge what the commenter themselves argues, not fallacies that they merely mention or criticise.
- `best` is your single best answer. `also_fits` lists up to two other names from the list that could also reasonably apply (empty if none).
Return JSON only."""
SCHEMA = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["best", "also_fits", "why"],
    "properties": {"best": {"type": "string", "enum": NAMES}, "also_fits": {"type": "array", "items": {"type": "string", "enum": NAMES}},
                   "why": {"type": "string"}}}}}
lock = threading.Lock()
usage = {"in": 0, "cached": 0, "out": 0, "calls": 0}


def ask(text, model):
    for attempt in range(5):
        try:
            r = bd.client.chat.completions.create(model=model, response_format=SCHEMA,
                                                  messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
            with lock:
                u = r.usage
                usage["in"] += u.prompt_tokens
                usage["cached"] += getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
                usage["out"] += u.completion_tokens
                usage["calls"] += 1
            return json.loads(r.choices[0].message.content)
        except Exception as e:  # rate limits and transient errors
            if attempt == 4 or "no credits" in str(e):
                raise
            time.sleep(2 ** attempt)


def evaluate(items):
    n = len(items)
    top1 = sum(norm(i["answer"]["best"]) in {norm(g) for g in i["gold"]} for i in items)
    top3 = sum(any(norm(x) in {norm(g) for g in i["gold"]} for x in [i["answer"]["best"]] + i["answer"]["also_fits"][:2]) for i in items)
    isn = [i for i in items if i["gold"] == ["None"]]
    fal = [i for i in items if i["gold"] != ["None"]]
    said_none = lambda i: norm(i["answer"]["best"]) == "none"
    return dict(n=n, top1=100 * top1 / n, top3=100 * top3 / n,
                none_recall=100 * sum(map(said_none, isn)) / len(isn) if isn else None,
                fallacy_top1=100 * sum(norm(i["answer"]["best"]) in {norm(g) for g in i["gold"]} for i in fal) / len(fal) if fal else None,
                said_none_on_fallacy=100 * sum(map(said_none, fal)) / len(fal) if fal else None)


def probe(model, per_set=20):
    """Memorisation probe: give the first 12 words, ask for the continuation, compare 5-grams with the real continuation."""
    out = {}
    for name in ("cocolofa_test", "mafalda_test"):
        rows = [r for r in read_set(ROOT / SETS[name]) if len(r["text"].split()) >= 40 and "TITLE:" not in r["text"]][:per_set]
        hits = []
        for r in tqdm(rows, desc=f"probe {name}", dynamic_ncols=True):
            words = r["text"].split()
            head, tail = " ".join(words[:12]), words[12:42]
            resp = bd.client.chat.completions.create(model=model, messages=[
                {"role": "user", "content": f"Continue this online comment exactly as it was originally written, in about 30 more words. Output only the continuation.\n\n{head}"}])
            cont = resp.choices[0].message.content.lower().split()
            g = lambda w: {tuple(w[i:i + 5]) for i in range(len(w) - 4)}
            real = g([x.lower() for x in tail])
            hits.append(len(real & g(cont)) / max(len(real), 1))
        out[name] = sum(hits) / len(hits)
        print(f"  {name}: mean share of the true 5-grams reproduced = {100 * out[name]:.1f}%  (verbatim recall would be far above 10%)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    args = ap.parse_args()
    if args.probe:
        probe(args.model)
        return
    for name in args.sets:
        items = read_set(ROOT / SETS[name])[:args.limit]
        with ThreadPoolExecutor(8) as pool:
            answers = list(tqdm(pool.map(lambda it: ask(it["text"], args.model), items), total=len(items), desc=name, dynamic_ncols=True))
        for it, a in zip(items, answers):
            it["answer"] = a
        if not args.limit:
            (ROOT / "data" / f"llm_{args.model}_{name}.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        r = evaluate(items)
        print(f"{name:26} n={r['n']:4}  top-1 {r['top1']:5.1f}%  top-3 {r['top3']:5.1f}%"
              + (f"  None recall {r['none_recall']:.0f}%" if r["none_recall"] is not None else "")
              + (f"  fallacy top-1 {r['fallacy_top1']:.0f}% (said None on {r['said_none_on_fallacy']:.0f}%)" if r["fallacy_top1"] is not None else ""), flush=True)
    print(f"tokens: in {usage['in']:,} (cached {usage['cached']:,}), out {usage['out']:,}, calls {usage['calls']}")


if __name__ == "__main__":
    main()
