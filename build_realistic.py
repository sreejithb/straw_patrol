"""Realistic-style training data: every fallacy label, plus new 'None' kinds, written the way real comments look.

Fallacy part  every one of the 185 labels (+ Appeal to Tradition) gets N_ALL examples; the ~40 well-known ones get N_COMMON more.
None part     mention-not-use, heated-but-fair, positive/neutral, fair look-alikes and civil threads, in the same style and lengths.
Style         25-130 words, informal, X replies / Reddit / news comments / forums / group chats, fallacy not signalled by keywords.
Same checks   as the earlier sets (auto checks + gpt-5.4 judge) plus a dedupe/overlap check against every existing example.
Output        data/generated/train_real.jsonl and eval_real.jsonl (chat format, ~8% held out per label/kind) + *_raw.jsonl.

  python build_realistic.py --smoke      # 3 labels x 2 examples, 2 none kinds: prints samples
  python build_realistic.py              # full run
"""
import argparse
import collections
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import build_dataset as bd
import build_nofallacy as bn
import prompts
from prior_adjust import COMMON

N_ALL, N_COMMON = 12, 25
HOLDOUT = 0.08
ROUND_OFFSET = 500
OVERLAP_NGRAM = 8
lock = threading.Lock()

NEW_FALLACIES = [{
    "name": "Appeal to Tradition", "aka": "Appeal to antiquity",
    "description": "Claiming that something is right, good or better only because it has been done or believed for a long time, so that age replaces evidence.",
    "logical_form": "X has always been done (or believed). Therefore X is right and should continue.", "examples": []}]

FORMATS = ["X reply that starts with an @-mention and quotes or paraphrases the post it answers", "Reddit comment under a news post",
           "comment under a news article", "forum post in a long discussion thread", "YouTube comment", "group-chat message to friends",
           "X post of a few sentences with a hashtag", "Reddit reply that quotes the line it answers with '>'",
           "post in a local community group", "LinkedIn-style comment"]
LENGTHS = ["about 25 words", "about 40 words", "about 55 words", "about 70 words", "about 90 words", "about 110 words"]

GEN_SYSTEM = """You write training data for a model that spots logical fallacies in REAL social-media comments (X replies, Reddit, news comment sections, forums, YouTube, group chats). You get ONE fallacy (name, aliases, definition, general form). Write original comments in which the speaker commits it.

Requirements for every example:
- It sounds like a real person typing a comment, not a textbook scene: informal, opinionated, sometimes messy. Light typos, slang, an emoji or a hashtag are fine but not required.
- Follow the format and approximate length assigned to each example number. Topic is only a hint: ignore it when it makes the comment less natural.
- The fallacy is part of what the person actually argues, woven into a larger comment. Real comments also contain ordinary, reasonable remarks around the fallacy; include some.
- Do not announce the fallacy with obvious keywords ('everyone knows', 'obviously', 'always') unless the definition requires them.
- It clearly commits the target fallacy, and no other fallacy more obviously. The speaker is the one committing it.
- If the fallacy answers someone else's claim, make that claim visible (an @-reply, a quote, or a paraphrase) so the fallacy can be seen.
- The explanation (2-3 sentences) says what the speaker did wrong, in plain language, referring to the specific comment. It starts with exactly 'This is a <Name>:' or 'This is an <Name>:' using the given name.
{avoid}
Return JSON only."""

GEN_USER = """Fallacy: {name}
Also known as: {aka}
Definition: {description}
General form: {logical_form}
{guidance}
Write {n} different examples, one per assignment below:
{assignments}"""

NONE_SYSTEM = """You write test and training data for a bot that spots logical fallacies in REAL social-media comments. The bot must learn to say "no fallacy" when a comment contains none. You write comments that contain NO logical fallacy of any kind.

Kind of text to write:
{instruction}

Requirements for every example:
- It sounds like a real person typing a comment (X reply, Reddit, news comment, forum, YouTube, group chat): informal, sometimes messy, opinionated or emotional where the kind says so.
- It contains no logical fallacy. A careful reviewer looking for fallacies must find nothing to flag. Do not hide a subtle fallacy in it.
- Follow the format and approximate length assigned to each example number. Topic is only a hint.
- The distinguishing feature described under 'Kind of text to write' must be clearly present in the comment itself.
- Vary tone, length and structure. Do not repeat a sentence template.
- Also write `reason`: one or two sentences that start exactly with 'This is not a fallacy:' and say why there is nothing to flag.
{avoid}
Return JSON only."""

NONE_UNITS = [
    ("mention", "named", "A comment where the speaker NAMES or DESCRIBES a fallacy without committing it: accusing someone else of a strawman, saying that a friend keeps repeating the same point, explaining what a slippery slope is, complaining that a debate was full of ad hominems. The speaker's own reasoning is sound.", 150),
    ("heated", "fair", "A strongly worded, angry, sad or excited comment that is nevertheless fair: venting about a real problem with concrete reasons, a justified complaint, an emotional personal story that draws no sweeping conclusion, passionate agreement with reasons. Emotion is not a fallacy.", 150),
    ("plain", "positive", "A positive, supportive or neutral comment: praise, thanks, congratulations, sharing a link with a short remark, a joke, a question, a plain report of what happened. No argument is made.", 100),
    ("lookalike", "expert", "Cites a genuine, relevant expert or a real consensus within that expert's own field and says why they are credible. Looks like an appeal to authority but is legitimate.", 14),
    ("lookalike", "statistic", "Uses a relevant, honestly described statistic or study (sample size and scope are fair, conclusion is hedged). Looks like cherry picking or hasty generalisation but is legitimate.", 14),
    ("lookalike", "analogy", "A real analogy that matches the original case on the points that matter and explains why. Looks like a weak analogy but is legitimate.", 14),
    ("lookalike", "risk", "A cautious warning about a realistic, evidence-backed risk or a plausible chain of consequences where each step is supported. Looks like fear-mongering or a slippery slope but is legitimate.", 14),
    ("lookalike", "credibility", "Questions a person's credibility for a directly relevant reason (a documented record on this very topic, a clear stake in this outcome) AND also engages the claim itself. Looks like an ad hominem but is legitimate.", 14),
    ("lookalike", "popularity", "Uses many independent reviews or votes as relevant evidence for a practical choice (where to eat, which tool to try) without claiming popularity proves a fact. Looks like an appeal to the majority but is legitimate.", 14),
    ("lookalike", "dilemma", "An either/or that is genuinely exhaustive (a vote passes or fails, a deadline is met or missed) and leads to a valid conclusion. Looks like a false dilemma but is legitimate.", 14),
    ("lookalike", "causal", "A causal claim backed by more than coincidence (controlled comparison, known mechanism, dose-response). Looks like post hoc but is legitimate.", 14),
    ("thread", "civil", "An exchange of 2-3 comments from different people (X replies with @-mentions, or Reddit-style) who disagree or discuss, give evidence, and answer each other's actual points. Nobody commits a fallacy.", 60),
]


def my_none_check(item: dict, seen: set) -> str:
    """Like build_nofallacy.auto_check, but fallacy names are allowed in the text (mention-not-use)."""
    text, reason = item["text"].strip(), item["reason"].strip()
    words = len(text.split())
    if not 12 <= words <= 140:
        return f"length {words}"
    if not reason.lower().startswith("this is not a fallacy:"):
        return "reason prefix"
    key = " ".join(bd.re.findall(r"[a-z]+", text.lower()))[:80]
    if key in seen:
        return "duplicate"
    seen.add(key)
    return ""


def assign(uid, n, round_no, thread=False):
    rng = random.Random(f"{uid}/{round_no}")
    topics = rng.sample(bd.TOPICS, n)
    fmts = ["exchange of 2-3 comments from different people"] * n if thread else [rng.choice(FORMATS) for _ in range(n)]
    lens = [rng.choice(LENGTHS) for _ in range(n)] if not thread else ["about 80 words in total"] * n
    return "\n".join(f"{i + 1}. topic hint: {t}; format: {f}; length: {l}" for i, (t, f, l) in enumerate(zip(topics, fmts, lens)))


def generate_real(fallacy, n, round_no):
    system = GEN_SYSTEM.format(avoid="\n".join(f"- {a}" for a in prompts.AVOID))
    user = GEN_USER.format(name=fallacy["name"], aka=fallacy["aka"] or "n/a", description=fallacy["description"],
                           logical_form=fallacy["logical_form"] or "n/a", n=n, guidance=bd.guidance_line(fallacy["name"]),
                           assignments=assign(fallacy["name"], n, round_no))
    return [{"fallacy": fallacy["name"], **it} for it in bd.chat_json(bd.GEN_MODEL, system, user, bd.GEN_SCHEMA)["examples"]]


def run_label(fallacy, need, book_grams, keys, grams, bar, stats, out, raw_path):
    seen, kept, rows = set(keys), [], []
    per_call = min(10, max(4, need // 2 + 2))
    max_rounds = min(14, max(3, -(-need // max(1, int(per_call * 0.4))) + 1))  # ~40% survive the checks
    for r in range(max_rounds):
        items = generate_real(fallacy, per_call, ROUND_OFFSET + r)
        reasons = []
        for it in items:
            reason = bd.auto_check(it, fallacy, book_grams, seen)
            if not reason and bd.ngrams(it["argument"], OVERLAP_NGRAM) & grams:
                reason = "overlaps existing example"
            reasons.append(reason)
        with ThreadPoolExecutor(4) as pool:
            verdicts = list(pool.map(lambda x: None if x[1] else bd.judge(x[0], fallacy), zip(items, reasons)))
        for it, reason, v in zip(items, reasons, verdicts):
            keep = not reason and bd.passes(v)
            rows.append({**it, "keep": keep, "reject_reason": reason or ("" if keep else v["issues"] or "judge"), "judge": v})
            if keep and len(kept) < need:
                kept.append(it)
        if len(kept) >= need:
            break
    with lock:
        with raw_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.extend({"label": fallacy["name"], "chat": bd.to_chat(it)} for it in kept)
        stats["kept"] += len(kept)
        stats["generated"] += len(rows)
        stats["short"] += need - len(kept)
        bar.update(1)
        bar.set_postfix(kept=stats["kept"], generated=stats["generated"], short=stats["short"])


def load(name):
    p = bd.GEN_DIR / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse the raw files: no fallacy regeneration, only the 'None' units still below target")
    args = ap.parse_args()
    bd.MIN_WORDS, bd.MAX_WORDS = 20, 135  # build_dataset's checks use these module constants

    existing = [r["messages"][1]["content"] for f in ("train.jsonl", "train_extra.jsonl", "eval.jsonl", "train_none.jsonl", "eval_none.jsonl",
                                                       "calib_fallacy.jsonl", "calib_none.jsonl") for r in load(f)]
    key = lambda t: " ".join(bd.re.findall(r"[a-z]+", t.lower()))[:80]
    keys = {key(a) for a in existing}
    grams = set().union(*(bd.ngrams(a, OVERLAP_NGRAM) for a in existing))
    book_grams = set()
    for f in bd.load_fallacies():
        for ex in f["examples"]:
            book_grams |= bd.ngrams(ex["text"], bd.COPY_NGRAM)

    by_name = {f["name"]: f for f in bd.load_fallacies()} | {f["name"]: f for f in NEW_FALLACIES}
    labels = sorted({r["messages"][2]["content"].split("\n")[0][len("Fallacy: "):].strip() for f in ("train.jsonl", "train_extra.jsonl") for r in load(f)}
                    | {f["name"] for f in NEW_FALLACIES})
    labels = [n for n in labels if n in by_name]
    common = set(COMMON) | {"Appeal to Tradition"}
    need = {n: N_ALL + (N_COMMON if n in common else 0) for n in labels}
    none_units = NONE_UNITS
    if args.smoke:
        labels, none_units = ["Hasty Generalization", "Appeal to Tradition", "Kettle logic"], [NONE_UNITS[0], NONE_UNITS[3]]
        need = {n: 2 for n in labels}
        none_units = [(k, s, i, 3) for k, s, i, _ in none_units]
    print(f"{len(labels)} labels, {sum(need[n] for n in labels)} fallacy examples wanted; none units {len(none_units)}, "
          f"{sum(u[3] for u in none_units)} wanted", flush=True)

    suffix = "_smoke" if args.smoke else ""
    raw_f = bd.GEN_DIR / f"real_fallacy_raw{suffix}.jsonl"
    out_f, stats = [], {"kept": 0, "generated": 0, "short": 0}
    if args.resume:
        taken = collections.Counter()
        for row in load(raw_f.name):
            if row["keep"] and taken[row["fallacy"]] < need.get(row["fallacy"], 0):
                taken[row["fallacy"]] += 1
                out_f.append({"label": row["fallacy"], "chat": bd.to_chat(row)})
        stats["kept"] = len(out_f)
        stats["short"] = sum(need[n] - taken[n] for n in need)
        print(f"resumed {len(out_f)} fallacy examples from {raw_f.name}", flush=True)
    else:
        raw_f.write_text("", encoding="utf-8")
        bar = tqdm(total=len(labels), unit="label", desc="realistic fallacy", dynamic_ncols=True)
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda n: run_label(by_name[n], need[n], book_grams, keys, grams, bar, stats, out_f, raw_f), labels))
        bar.close()

    # ---- none part: reuse build_nofallacy's machinery with the realistic prompts and a check that allows fallacy names
    bn.GEN_SYSTEM, bn.auto_check, bn.ROUND_OFFSET = NONE_SYSTEM, my_none_check, ROUND_OFFSET
    bn.RAW = bd.GEN_DIR / f"real_none_raw{suffix}.jsonl"
    old_none = []
    if args.resume:
        ROUND = 600  # fresh topic/format draws for the units that still need examples
        bn.ROUND_OFFSET = ROUND
        old_none = [r for r in load(bn.RAW.name) if r["keep"]]
        have = collections.Counter(r["kind"] + "/" + r["subtype"] for r in old_none)
        bn.EXISTING.update(key(r["text"]) for r in load(bn.RAW.name))
        none_units = [(k, sub, i, t - have[k + "/" + sub]) for k, sub, i, t in none_units if have[k + "/" + sub] < t]
        print(f"resumed {len(old_none)} 'None' examples; {len(none_units)} units still short, {sum(u[3] for u in none_units)} wanted", flush=True)
    else:
        bn.RAW.write_text("", encoding="utf-8")
    bn.EXISTING.update(keys)
    bn.assign = lambda uid, kind, n, r: assign(uid, n, r, thread=(kind == "thread"))
    out_n, stats_n = [], {"kept": 0, "generated": 0}
    bar = tqdm(total=len(none_units), unit="unit", desc="realistic none", dynamic_ncols=True)
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(lambda u: bn.run_unit(u, bar, stats_n, out_n, min(45, max(3, -(-u[3] // 3) + 1)), 8), none_units))
    bar.close()
    out_n = old_none + out_n

    if args.smoke:
        for x in out_f[:4]:
            print(f"\n[{x['label']}] {x['chat']['messages'][1]['content']}\n   => {x['chat']['messages'][2]['content'].split(chr(10))[1][:200]}")
        for r in out_n[:6]:
            print(f"\n[None/{r['kind']}/{r['subtype']}] {r['text']}\n   => {r['reason'][:200]}")
        print(f"\nsmoke: fallacy kept {stats['kept']}/{stats['generated']} generated; none kept {stats_n['kept']}/{stats_n['generated']}")
        return

    rng = random.Random(0)
    groups = collections.defaultdict(list)
    for x in out_f:
        groups["F/" + x["label"]].append(x["chat"])
    for r in out_n:
        groups["N/" + r["kind"] + "/" + r["subtype"]].append(bn.to_chat(r))
    train, ev = [], []
    for g in groups.values():
        rng.shuffle(g)
        k = max(1, round(len(g) * HOLDOUT)) if len(g) >= 4 else 0
        ev += g[:k]
        train += g[k:]
    for name, part in (("train_real", train), ("eval_real", ev)):
        (bd.GEN_DIR / f"{name}.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in part), encoding="utf-8")
        print(f"wrote {len(part)} rows to {name}.jsonl")
    print(f"fallacy kept {stats['kept']} (short of target by {stats['short']}); none kept {stats_n['kept']}")


if __name__ == "__main__":
    main()
