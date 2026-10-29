"""Build the v6 training file: generated training data + CoCoLoFa training comments with LLM-written explanations.

  python build_train_v6.py --limit 8          # smoke test, prints samples, writes nothing
  python build_train_v6.py                    # all CoCoLoFa train records (resumable: answers are cached)

CoCoLoFa has labels but no explanations, and its labels are noisy. For each comment the LLM (given the label and its definition)
(1) judges whether the label fits: yes / partly / no, and (2) writes a one-sentence explanation. Only `yes` comments are kept,
so the model is never trained to rationalise a label that is not in the text.
Input  data/v6/train/{generated,cocolofa}.jsonl      Output  data/v6/train_v6.jsonl (chat format) and data/v6/cocolofa_explanations.jsonl (cache)
The output contains CoCoLoFa text: local only (gitignored).
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

import build_dataset as bd
import build_realistic as br

ROOT = Path(__file__).parent
V6 = ROOT / "data" / "v6"
CACHE = V6 / "cocolofa_explanations.jsonl"
MODEL = "gpt-5.4"
SYSTEM_MSG = "You are a logical-fallacy expert. Identify the fallacy in the argument and explain it."
SYSTEM = """You prepare training targets for a model that names the logical fallacy in a news comment.
You get a comment, the label an annotator gave it (a fallacy name, or None) and the fallacy's definition.
1. `fits`: does the label really apply to what THIS commenter argues? yes = clearly; partly = arguably or weakly; no = it does not.
   For the label None: yes = the comment contains no logical fallacy; no = it does contain one.
2. `explanation`: ONE or at most two sentences, plain language, about this specific comment.
   - for a fallacy: start exactly with 'This is a <Name>:' (or 'This is an <Name>:') using the given name, then say what the commenter did wrong;
   - for None: start exactly with 'This is not a fallacy:' and say why there is nothing to flag.
   Do not mention annotators or labels. Write the explanation even when fits is partly or no."""
SCHEMA = {"type": "json_schema", "json_schema": {"name": "target", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["fits", "explanation"],
    "properties": {"fits": {"type": "string", "enum": ["yes", "partly", "no"]}, "explanation": {"type": "string"}}}}}
lock = threading.Lock()
usage = {"calls": 0, "in": 0, "out": 0}


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()] if Path(p).exists() else []


def definitions():
    d = {f["name"]: f["description"] for f in bd.load_fallacies()} | {f["name"]: f["description"] for f in br.NEW_FALLACIES}
    d["None"] = "No logical fallacy: the comment gives reasons, states an opinion, asks a question or reacts without a flawed inference."
    return d


def ask(r, defs):
    user = f"Label: {r['label']}\nDefinition: {defs.get(r['label'], '')}\n\nComment:\n\"\"\"{r['text']}\"\"\""
    for attempt in range(5):
        try:
            resp = bd.client.chat.completions.create(model=MODEL, response_format=SCHEMA,
                                                     messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
            with lock:
                usage["calls"] += 1
                usage["in"] += resp.usage.prompt_tokens
                usage["out"] += resp.usage.completion_tokens
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == 4 or "no credits" in str(e):
                raise
            time.sleep(2 ** attempt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    defs = definitions()
    coco = read(V6 / "train" / "cocolofa.jsonl")
    done = {r["id"]: r for r in read(CACHE)}
    todo = [r for r in coco if r["id"] not in done][:args.limit]
    print(f"{len(coco)} CoCoLoFa training records, {len(done)} already explained, {len(todo)} to do", flush=True)
    with ThreadPoolExecutor(8) as pool:
        answers = list(tqdm(pool.map(lambda r: ask(r, defs), todo), total=len(todo), desc="explain", dynamic_ncols=True))
    if args.limit:
        for r, a in list(zip(todo, answers))[:args.limit]:
            print(f"\n[{r['label']}] fits={a['fits']}\n  {r['text'][:260]!r}\n  => {a['explanation']}")
        print(f"\ntokens: in {usage['in']:,} out {usage['out']:,} over {usage['calls']} calls")
        return
    with CACHE.open("a", encoding="utf-8") as f:
        for r, a in zip(todo, answers):
            row = {"id": r["id"], **a}
            done[r["id"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    chat, kept = [], {"yes": 0, "partly": 0, "no": 0}
    for r in read(V6 / "train" / "generated.jsonl"):
        chat.append({"messages": [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": r["text"]}, {"role": "assistant", "content": r["answer"]}]})
    n_generated = len(chat)
    per_label = {}
    for r in coco:
        a = done[r["id"]]
        kept[a["fits"]] += 1
        if a["fits"] == "yes":
            per_label[r["label"]] = per_label.get(r["label"], 0) + 1
            chat.append({"messages": [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": r["text"]},
                                      {"role": "assistant", "content": f"Fallacy: {r['label']}\n{a['explanation'].strip()}"}]})
    (V6 / "train_v6.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chat), encoding="utf-8")
    print(f"label fit judged by the LLM over {len(coco)} CoCoLoFa comments: {kept}")
    print("kept per label (fits = yes):", dict(sorted(per_label.items(), key=lambda kv: -kv[1])))
    print(f"wrote {len(chat)} training examples to train_v6.jsonl ({n_generated} generated + {len(chat) - n_generated} CoCoLoFa)")
    print(f"tokens this run: in {usage['in']:,} out {usage['out']:,} over {usage['calls']} calls")


if __name__ == "__main__":
    main()
