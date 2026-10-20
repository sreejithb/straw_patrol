"""Generate the 'no fallacy' class: texts a fallacy bot should answer with `Fallacy: None`.

Four kinds of text (see UNITS):
  sound       short arguments with valid reasoning and relevant evidence
  lookalike   hard negatives that resemble a fallacy on the surface but are legitimate (a real expert, relevant credibility, ...)
  nonarg      comments that make no argument (opinion, question, greeting, announcement, joke, personal story, request)
  thread      civil multi-comment exchanges where everybody reasons properly

Pipeline: generate (gpt-5.4-mini) -> auto checks -> judge (gpt-5.4, must find NO fallacy) -> split by kind.
Writes data/generated/train_none.jsonl and eval_none.jsonl (chat format, answer `Fallacy: None` + one-sentence reason)
and none_raw.jsonl (everything, with judge verdicts). Existing train/eval files are not touched.

  python build_nofallacy.py --pilot          # 3 units, a few examples each, prints samples
  python build_nofallacy.py                  # full run
"""
import argparse
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import build_dataset as bd
import prompts

SYSTEM_MSG = "You are a logical-fallacy expert. Identify the fallacy in the argument and explain it."
EVAL_FRACTION = 0.2
MIN_WORDS, MAX_WORDS = 3, 120
PER_CALL = 8
ROUND_OFFSET = 100
RAW = bd.GEN_DIR / "none_raw.jsonl"
lock = threading.Lock()
EXISTING: set = set()  # dedupe keys of rows kept in an earlier pass

# (kind, subtype, instruction for the writer, number of examples to keep)
UNITS = [
    ("sound", "evidence", "A short argument whose reasoning is valid and whose evidence is relevant, specific and proportionate to the conclusion. The conclusion is hedged the way a careful person would hedge it.", 70),
    ("sound", "deductive", "A short argument with a logically valid structure (for example modus ponens, modus tollens, a valid syllogism, a correct conditional) about an everyday topic, stated naturally and not as a textbook exercise.", 40),
    ("sound", "disagreement", "A civil disagreement with someone's claim that engages the actual claim and gives a reason or evidence for the disagreement. Show the claim being answered.", 40),
    ("lookalike", "expert", "Relies on a genuine, relevant expert or a real scientific consensus on a question inside that expert's field, and says why that source is credible. On the surface it resembles an appeal to authority, but it is legitimate.", 14),
    ("lookalike", "credibility", "Questions a person's credibility for a directly relevant reason (a documented record of falsifying this very kind of data, a clear financial stake in this specific outcome) AND still addresses the claim itself. It resembles an ad hominem but is legitimate.", 14),
    ("lookalike", "risk", "Warns about a realistic, evidence-backed risk or a short causal chain whose every step is plausible and supported. It resembles an appeal to fear or a slippery slope, but is legitimate.", 14),
    ("lookalike", "analogy", "An analogy that really does match the original case on the points that matter, and says why. It resembles a weak analogy, but is legitimate.", 14),
    ("lookalike", "sample", "A cautious generalisation from a large, representative sample, with the scope stated honestly. It resembles a hasty generalisation, but is legitimate.", 14),
    ("lookalike", "causal", "A causal claim that is backed by more than a coincidence in time (a controlled comparison, a known mechanism, a dose-response pattern). It resembles post hoc or correlation-is-causation, but is legitimate.", 14),
    ("lookalike", "popularity", "Uses large numbers of independent reviews, votes or users as relevant evidence for a practical choice (where to eat, which tool to try), without claiming that popularity proves a fact. It resembles an appeal to popularity, but is legitimate.", 14),
    ("lookalike", "emotion", "An emotional appeal that is also relevant to the decision and supported by real facts (a charity describing the real situation of the people it helps). It resembles an appeal to emotion, but is legitimate.", 14),
    ("lookalike", "dilemma", "An either/or statement where the two options really are exhaustive or the situation really is binary (a vote passes or fails, a light is on or off), used to reach a valid conclusion. It resembles a false dilemma, but is legitimate.", 14),
    ("lookalike", "topic", "Politely moves to a related point, explicitly acknowledging the previous point first and saying why the new point matters to the same question. It resembles a red herring, but is legitimate.", 14),
    ("lookalike", "burden", "Asks a person making a surprising or extraordinary claim for evidence, without claiming the claim is false just because it lacks evidence. It resembles shifting the burden of proof or an argument from ignorance, but is legitimate.", 14),
    ("nonarg", "opinion", "A plain opinion or reaction to something (a show, a game, food, a product) that gives impressions but does not argue for a claim.", 14),
    ("nonarg", "question", "A genuine question asking for information or advice. It contains no claim presented as an argument.", 14),
    ("nonarg", "greeting", "A greeting, congratulation, thanks, condolence, or other social message.", 12),
    ("nonarg", "announcement", "A neutral announcement or report of facts (an event, a release, a schedule, a result) with no argument.", 12),
    ("nonarg", "joke", "A harmless joke or witty remark that does not try to prove anything.", 12),
    ("nonarg", "story", "A personal anecdote or experience told without drawing a general conclusion from it.", 12),
    ("nonarg", "request", "A request, suggestion or invitation (asking someone to share a link, proposing a plan for the weekend) without argument.", 12),
    ("thread", "civil", "A short exchange of 2-3 comments (write them as A: / B: or with uncommon names) in which people disagree or discuss a topic, reason properly, give evidence and respond to each other's actual points. Nobody commits any fallacy.", 70),
]

GEN_SYSTEM = """You write test data for a bot that spots logical fallacies in short social-media comments.
The bot must learn to say "no fallacy" when a comment contains none. You write comments that contain NO logical fallacy of any kind.

Kind of text to write:
{instruction}

Requirements for every example:
- The distinguishing feature described in 'Kind of text to write' must be clearly present in the text itself. If the text would not show it, write a different text.
- It is something a real person would plausibly post or say: natural voice, concrete topic, at most ~80 words (a thread may reach ~110).
- It must contain no logical fallacy at all. A careful reviewer who looks for fallacies must find nothing to flag. Do not hide a subtle fallacy in it.
- Follow the format assigned to each example number. The topic is only a hint: ignore it when it makes the text less natural.
- Do not mention fallacies or logic words in the text itself (no 'fallacy', 'valid', 'sound reasoning').
- Vary length, structure and tone across the examples. Do not repeat a sentence template.
- Also write `reason`: one or two sentences that start exactly with 'This is not a fallacy:' and say why there is nothing to flag
  (for example the reasoning holds because ..., or the comment is a question, not an argument).
{avoid}
Return JSON only."""

STYLES = ["tweet-length post", "forum reply", "single sentence in an op-ed", "text message to a friend",
          "workplace chat message", "reply that quotes and answers someone's claim", "short comment under a news article"]


def assign(uid: str, kind: str, n: int, round_no: int) -> str:
    rng = random.Random(f"{uid}/{round_no}")
    topics = rng.sample(bd.TOPICS, n)
    fmts = ["exchange of 2-3 comments from different people (A:/B: or uncommon names)"] * n if kind == "thread" \
        else [rng.choice(STYLES) for _ in range(n)]
    return "\n".join(f"{i + 1}. topic hint: {t}; format: {f}" for i, (t, f) in enumerate(zip(topics, fmts)))


GEN_USER = "Write {n} different examples, one per assignment:\n{assignments}"

AVOID = [prompts.AVOID[0], prompts.AVOID[1], prompts.AVOID[2],
         "Do not write a scripted scene where somebody announces a surprising fact just to be corrected."]

GEN_SCHEMA = {"type": "json_schema", "json_schema": {"name": "examples", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["examples"],
    "properties": {"examples": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["text", "reason"],
        "properties": {"text": {"type": "string"}, "reason": {"type": "string"}}}}}}}}

JUDGE_SYSTEM = """You are a strict reviewer of logical reasoning. You receive a social-media comment (or short thread) that was written to contain NO logical fallacy, plus the kind of text it was meant to be.
Decide if it really contains no fallacy. Be demanding: if a careful reviewer could reasonably flag any fallacy (appeal to authority, ad hominem, hasty generalisation, false dilemma, slippery slope, post hoc, straw man, appeal to emotion, bandwagon, and so on), say so.
Return JSON only."""

JUDGE_USER = """Intended kind: {kind} / {subtype} -- {instruction}

Text:
\"\"\"{text}\"\"\"

Proposed explanation: {reason}

Fill in: `analysis` (brief), `contains_fallacy` (true if any fallacy could reasonably be flagged), `fallacy_found` (name or empty),
`matches_kind` (the text really is the intended kind: e.g. a non-argument really makes no argument; a lookalike really is legitimate; a thread is civil and reasoned),
`realistic` (a real person could post it), `reason_ok` (the explanation is accurate and specific), `issues`."""

JUDGE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["analysis", "contains_fallacy", "fallacy_found", "matches_kind", "realistic", "reason_ok", "issues"],
    "properties": {"analysis": {"type": "string"}, "contains_fallacy": {"type": "boolean"}, "fallacy_found": {"type": "string"},
                   "matches_kind": {"type": "boolean"}, "realistic": {"type": "boolean"}, "reason_ok": {"type": "boolean"},
                   "issues": {"type": "string"}}}}}


def auto_check(item: dict, seen: set) -> str:
    text, reason = item["text"].strip(), item["reason"].strip()
    words = len(text.split())
    if not MIN_WORDS <= words <= MAX_WORDS:
        return f"length {words}"
    if not reason.lower().startswith("this is not a fallacy:"):
        return "reason prefix"
    if re.search(r"fallac|strawman|straw man|slippery slope", text.lower()):
        return "mentions fallacies"
    if re.match(r"\s*(tweet|comment|reply|op-ed|post|text|message)\b[^:]{0,30}:", text.lower()):
        return "scaffold label"
    key = " ".join(re.findall(r"[a-z]+", text.lower()))[:80]
    if key in seen:
        return "duplicate"
    seen.add(key)
    return ""


def run_unit(unit, bar, stats, out_rows, only_rounds=3, n=PER_CALL):
    kind, subtype, instruction, target = unit
    uid = f"{kind}/{subtype}"
    system = GEN_SYSTEM.format(instruction=instruction, avoid="\n".join(f"- {a}" for a in AVOID))
    seen, kept, rows = set(EXISTING), [], []
    for r in range(only_rounds):
        user = GEN_USER.format(n=n, assignments=assign(uid, kind, n, ROUND_OFFSET + r))
        items = bd.chat_json(bd.GEN_MODEL, system, user, GEN_SCHEMA)["examples"]
        reasons = [auto_check(it, seen) for it in items]
        with ThreadPoolExecutor(4) as pool:
            verdicts = list(pool.map(lambda x: None if x[1] else bd.chat_json(
                bd.JUDGE_MODEL, JUDGE_SYSTEM, JUDGE_USER.format(kind=kind, subtype=subtype, instruction=instruction,
                                                                 text=x[0]["text"], reason=x[0]["reason"]), JUDGE_SCHEMA),
                zip(items, reasons)))
        for it, reason, v in zip(items, reasons, verdicts):
            keep = not reason and not v["contains_fallacy"] and v["matches_kind"] and v["realistic"] and v["reason_ok"]
            rows.append({**it, "kind": kind, "subtype": subtype, "keep": keep, "round": r,
                         "reject_reason": reason or ("" if keep else (v["fallacy_found"] or v["issues"] or "judge")), "judge": v})
            if keep and len(kept) < target:
                kept.append(rows[-1])
        if len(kept) >= target:
            break
    with lock:
        with RAW.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_rows.extend(kept)
        stats["kept"] += len(kept)
        stats["generated"] += len(rows)
        bar.update(1)
        bar.set_postfix(kept=stats["kept"], generated=stats["generated"])


def to_chat(row: dict) -> dict:
    return {"messages": [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": row["text"].strip()},
                         {"role": "assistant", "content": f"Fallacy: None\n{row['reason'].strip()}"}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="3 units, one round, 4 examples each; prints the samples")
    ap.add_argument("--top-up", action="store_true", help="re-run only units below their target, keeping rows already in none_raw.jsonl")
    args = ap.parse_args()
    units = [UNITS[0], UNITS[3], UNITS[14]] if args.pilot else UNITS
    if args.pilot:
        units = [(k, s, i, 4) for k, s, i, _ in units]
    bd.GEN_DIR.mkdir(parents=True, exist_ok=True)
    kept_before = []
    if args.top_up:
        old = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
        kept_before = [r for r in old if r["keep"]]
        have = {}
        for r in kept_before:
            have[r["kind"] + "/" + r["subtype"]] = have.get(r["kind"] + "/" + r["subtype"], 0) + 1
        units = [(k, sub, i, t - have.get(k + "/" + sub, 0)) for k, sub, i, t in UNITS if have.get(k + "/" + sub, 0) < t]
        global ROUND_OFFSET
        ROUND_OFFSET = 200  # fresh topic/format draws
        EXISTING.update(" ".join(re.findall(r"[a-z]+", r["text"].lower()))[:80] for r in old)
        print(f"top-up: {len(units)} units short, {sum(u[3] for u in units)} examples wanted")
    elif not args.pilot:
        RAW.write_text("", encoding="utf-8")
    stats, out_rows = {"kept": 0, "generated": 0}, []
    bar = tqdm(total=len(units), unit="unit", desc="no-fallacy", dynamic_ncols=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda u: run_unit(u, bar, stats, out_rows, 1 if args.pilot else 3, 4 if args.pilot else PER_CALL), units))
    bar.close()
    if args.pilot:
        for r in out_rows:
            print(f"\n[{r['kind']}/{r['subtype']}]\n{r['text']}\n=> {r['reason']}")
        print(f"\npilot: kept {stats['kept']} of {stats['generated']}")
        return
    out_rows = kept_before + out_rows
    rng = random.Random(0)
    by_kind = {}
    for r in out_rows:
        by_kind.setdefault(r["subtype"] + "/" + r["kind"], []).append(r)
    train, ev = [], []
    for group in by_kind.values():  # stratified split so eval covers every subtype
        rng.shuffle(group)
        n_eval = max(1, round(len(group) * EVAL_FRACTION))
        ev += group[:n_eval]
        train += group[n_eval:]
    for name, part in (("train_none", train), ("eval_none", ev)):
        path = bd.GEN_DIR / f"{name}.jsonl"
        path.write_text("".join(json.dumps(to_chat(r), ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
        print(f"wrote {len(part)} rows to {path}")


if __name__ == "__main__":
    main()
