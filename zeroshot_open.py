"""Zero-shot benchmark of an open chat model (Qwen, Gemma, Llama, ...) on our sets, with the same label menu as the GPT baseline.

  python zeroshot_open.py --model Qwen/Qwen2.5-7B-Instruct --four-bit
  python zeroshot_open.py --model unsloth/gemma-2-9b-it --four-bit --sets eval cocolofa_test mafalda_test
The system prompt (all 186 names with one-sentence definitions, about 7k tokens) is identical for every text, so its key/value
cache is computed once and reused. Prints top-1 and "None" statistics; writes data/zeroshot_<model>_<set>.json.
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm

import chat_utils
from evaluate import norm
from score_candidates import read_set

ROOT = Path(__file__).parent
SETS = {"eval": "data/generated/eval.jsonl", "eval_none": "data/generated/eval_none.jsonl",
        "cocolofa_calib": "external/cocolofa_calib.jsonl", "cocolofa_test": "external/cocolofa_test.jsonl",
        "mafalda_test": "external/mafalda_test.jsonl", "cocolofa_tradition_test": "external/cocolofa_tradition_test.jsonl"}
INSTRUCTIONS = """You are an expert in informal logic. You read a social-media or news comment and decide which logical fallacy, if any, the commenter commits.
Choose ONLY from this list (name: short definition):
{menu}

Rules:
- If the comment contains no logical fallacy (a question, a plain opinion with reasons, an emotional but fair comment, a rebuttal of someone else's fallacy), answer None.
- Judge what the commenter themselves argues, not fallacies that they merely mention or criticise.
- Reply with exactly one name from the list (or None) on a single line, and nothing else."""


def parse(output, names_norm):
    first = re.sub(r"^\W*(answer|fallacy)\s*:\s*", "", output.strip().split("\n")[0], flags=re.I)
    n = norm(first)
    if n in names_norm:
        return names_norm[n]
    hits = sorted((k for k in names_norm if k and k in norm(output)), key=len, reverse=True)
    return names_norm[hits[0]] if hits else None


def stats(items):
    ok = lambda i: i["pred"] is not None and norm(i["pred"]) in {norm(g) for g in i["gold"]}
    isn = [i for i in items if i["gold"] == ["None"]]
    fal = [i for i in items if i["gold"] != ["None"]]
    pct = lambda a, b: 100 * a / b if b else float("nan")
    return dict(n=len(items), top1=pct(sum(map(ok, items)), len(items)), none_recall=pct(sum(ok(i) for i in isn), len(isn)),
                fallacy_top1=pct(sum(ok(i) for i in fal), len(fal)), unparsed=sum(i["pred"] is None for i in items))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-prefix-cache", action="store_true", help="re-read the 7k-token menu for every text (slower, always works)")
    ap.add_argument("--short-menu", action="store_true", help="names only, no definitions (about 3k tokens)")
    args = ap.parse_args()

    tok, model, dtype, device = chat_utils.load_model(args.model, args.four_bit)
    model.eval()

    lm = json.loads((ROOT / "data" / "label_menu.json").read_text(encoding="utf-8"))
    menu = "\n".join(l.split(":")[0] for l in lm["menu"].split("\n")) if args.short_menu else lm["menu"]
    system = INSTRUCTIONS.format(menu=menu)
    names_norm = {norm(n): n for n in lm["names"]}
    build = lambda text: chat_utils.prompt_text(tok, [{"role": "system", "content": system}, {"role": "user", "content": text}])
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]

    # shared prefix = longest common token prefix of two prompts with different user texts (minus a safety margin)
    a, b = enc(build("alpha beta gamma")), enc(build("delta epsilon zeta"))
    P = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))) - 2
    prefix = torch.tensor([a[:P]], device=device)
    cache = None
    if not args.no_prefix_cache:
        try:
            with torch.no_grad():
                cache = model(prefix, use_cache=True).past_key_values
            print(f"prefix of {P} tokens cached", flush=True)
        except Exception as e:
            print("prefix cache unavailable, running without it:", type(e).__name__, str(e)[:100])

    safe = args.model.replace("/", "_")
    for name in args.sets:
        items = read_set(ROOT / SETS[name])[:args.limit]
        t0 = time.time()
        for it in tqdm(items, desc=f"{safe}/{name}", dynamic_ncols=True):
            ids = enc(build(it["text"]))
            x = torch.tensor([ids], device=device)
            with torch.no_grad():
                if cache is not None and ids[:P] == a[:P]:
                    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), past_key_values=cache, max_new_tokens=16, do_sample=False)
                    try:
                        cache.crop(P)
                    except Exception:  # cache type without crop (e.g. sliding-window): fall back to full prompts
                        cache = None
                else:
                    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=16, do_sample=False)
            it["output"] = tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)
            it["pred"] = parse(it["output"], names_norm)
        if not args.limit:
            (ROOT / "data" / f"zeroshot_{safe}_{name}.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        s = stats(items)
        print(f"{name:26} n={s['n']:4} top-1 {s['top1']:5.1f}%  None recall {s['none_recall']:5.1f}%  fallacy top-1 {s['fallacy_top1']:5.1f}%  "
              f"unparsed {s['unparsed']}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
