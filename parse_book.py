"""Parse transcripts/full_transcript.txt into one JSON record per fallacy.

Output: data/fallacies.jsonl with
  {name, aka, description, logical_form, examples: [{text, explanation}], exception, tip}
"""
import json
import re
from pathlib import Path

SRC = Path(__file__).parent / "transcripts" / "full_transcript.txt"
OUT = Path(__file__).parent / "data" / "fallacies.jsonl"
MANUAL = Path(__file__).parent / "data" / "manual_entries.jsonl"

MAX_NAME_WORDS = 10
# Latin alias lines that follow the English name, e.g. "Argumentum ad hominem."
LATIN = re.compile(r"(Argument(um)? |Ad |Post |Petitio|Ignoratio|Reductio|Plurium|Obscurum|Quaternio|Contradictio|Onus|Non )", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?]) +(?=[A-Z\"'])")


CAPS_SECTIONS = [
    (r"DESCRIPTION", "Description."),
    (r"EXAMPLE FORM", "Logical Form."),
    (r"LOGICAL FORMS?", "Logical Form."),
    (r"EXAMPLE (\d+)", r"Example \1."),
    (r"EXPLANATION", "Explanation."),
    (r"EXCEPTION", "Exception."),
    (r"TIP", "Tip."),
    (r"FUN FACT", "Fun fact."),
]


def normalise_caps_sections(text: str) -> str:
    # one stretch of the transcript marks headings with a pilcrow: "Description ¶ ..."
    text = re.sub(r"\s*¶\s*", ". ", text)
    text = re.sub(r"\.\s*\. ", ". ", text)
    for pat, repl in CAPS_SECTIONS:
        text = re.sub(rf"\b{pat}\b\.?", repl, text)
    return text


CAPS_PREFIX = re.compile(r"((?:[A-Z][A-Z'\-]+ )+)(?=[A-Za-z])")


def split_caps_name(sentence: str) -> tuple[str, str]:
    """'AD FIDENTIA Argumentum ad fidentia' -> ('Ad Fidentia', 'Argumentum ad fidentia')."""
    words = sentence.split()
    if len(words) > 1 and all(w.isupper() for w in words):
        return sentence.title(), ""
    m = CAPS_PREFIX.match(sentence)
    if not m or not m.group(0).strip():
        return sentence, ""
    rest = sentence[m.end():].strip()
    return m.group(1).strip().title(), rest


# Whisper garbled or fused some names. Fixes verified against the transcript text.
NAME_FIXES = {
    "Argumentum circumstantial": "Circumstantial Ad Hominem",
    "Ad Hominem Abusive": "Abusive Ad Hominem",
    "Ad hominem, guilt by association": "Guilt by Association",
    "Ad hominem tu quoque": "Tu Quoque",
    "Appeal to Fear Argumentum Interum": "Appeal to Fear",
    "Appeal to Intuition Also known as, appeal to the gut": "Appeal to Intuition",
    "Appeal to Novelty, Argumentum Ad Novitaeum": "Appeal to Novelty",
    "Appeal to Pity Ad misericordium": "Appeal to Pity",
    "Make your same point, but in a different way": "Argument by Selective Reading",
    "Ecological Fallacy Also known as, Ecological Inference Fallacy": "Ecological Fallacy",
    "Inconsistency also known as internal contradiction, logical inconsistency": "Inconsistency",
    "Ludic Fallacy, Ludicis": "Ludic Fallacy",
    "Shifting the Burden of Proof Onus probandi": "Shifting the Burden of Proof",
    "Traitorous Critic Fallacy, ergo decado": "Traitorous Critic Fallacy",
    "Communication of conditionals": "Commutation of Conditionals",
    "Hypothesis Theory": "Canceling Hypothesis",
    "Logical effort": "Appeal to Effort",
    "Logical Language": "Loaded Language",
    "Kumhawk ergo Propterhawk": "Cum Hoc Ergo Propter Hoc",
    "Failed Ignorance": "Willful Ignorance",
    "Circulus In Demonstrando": "Circular Reasoning",
    "Argument Of The Beard": "Argument of the Beard",
    "Ad Hoc Rescue Ad Hoc": "Ad Hoc Rescue",
}
# Name could not be recovered reliably, or the entry is junk.
DROP_NAMES = {"Righteousness Fallacy", "Fallacy", "Logical Form", "Gatorine swine fallacy"}
SMALL_WORDS = {"of", "the", "to", "a", "an", "in", "by", "from", "and", "or", "for"}


def clean_name(name: str, description: str) -> str:
    if name in NAME_FIXES:
        return NAME_FIXES[name]
    if name == "Disjunction Fallacy" and description.startswith("Similar to the disjunction fallacy"):
        return "Conjunction Fallacy"
    if name.isupper():
        words = name.title().split()
        return " ".join(w.lower() if i and w.lower() in SMALL_WORDS else w for i, w in enumerate(words))
    return name


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]


def is_alias_line(sentence: str) -> bool:
    """Latin name line that follows the real English name."""
    return bool(LATIN.match(sentence))


def find_header(chunk: str) -> tuple[int, str, str]:
    """chunk = text just before a 'Description' anchor.
    Returns (offset where header starts, name, aka)."""
    # drop a 'New terminology.' block that sits between header and Description
    cut = len(chunk)
    nt = [m.start() for m in re.finditer(r"New [Tt]erminology\.", chunk[-800:])]
    if nt:
        cut = len(chunk) - 800 + nt[0] if len(chunk) > 800 else nt[0]
    head = chunk[:cut].rstrip()

    # inline form: "Name, also known as A, B."
    m = None
    for m in re.finditer(r"[,.] [Aa]lso known as", head[-500:]):
        pass
    tail_start = max(0, len(head) - 500)
    if m:
        aka_pos = tail_start + m.start()
        before = head[:aka_pos + 1]  # includes the '.' or ','
        sents = split_sentences(before.rstrip(".,") + ".")
        name = sents[-1].rstrip(".,") if sents else ""
        aka = head[aka_pos:].split("known as", 1)[1].strip(" .,")
        one_word = len(name.split()) == 1 and len(sents) > 1 and len(sents[-2].split()) <= 3
        raw = name
        name, extra = split_caps_name(name)
        if extra:
            aka = f"{extra}, {aka}"
        elif name != raw:
            pass  # ALL-CAPS name, already complete
        elif len(sents) > 1 and (is_alias_line(name) or one_word):
            aka = f"{name}. {aka}"
            name = sents[-2].rstrip(".,")
        return before.rstrip(".,").rfind(name), name, aka

    sents = split_sentences(head)
    name = sents[-1].rstrip(".") if sents else ""
    latin = ""
    raw = name
    name, extra = split_caps_name(name)
    if extra:
        latin = extra
    elif name != raw:
        pass  # ALL-CAPS name, already complete
    elif len(sents) > 1 and is_alias_line(name):
        latin, name = name, sents[-2].rstrip(".")
    return head.rfind(name), name, latin


def grab(section: str, body: str, stops: str) -> str:
    m = re.search(rf"{section}\.? (.*?)(?=(?:{stops})\.? |$)", body, re.S)
    return m.group(1).strip() if m else ""


def parse_body(body: str) -> dict:
    stops = r"Logical [Ff]orms?|Examples?(?: \d+)?|Explanation|Exception|Tip|Fun fact|Description"
    desc = grab("Description", body, stops)
    form = grab(r"Logical [Ff]orms?", body, stops)
    exception = grab("Exception", body, stops)
    tip = grab("Tip", body, stops) or grab("Fun fact", body, stops)

    examples = []
    for m in re.finditer(
        r"Example(?: \d+)?\.? (.*?)(?:Explanation\.? (.*?))?(?=(?:Example(?: \d+)?|Exception|Tip|Fun fact)\.? |$)",
        body, re.S,
    ):
        examples.append({"text": m.group(1).strip(), "explanation": (m.group(2) or "").strip()})
    return {"description": desc, "logical_form": form, "examples": examples,
            "exception": exception, "tip": tip}


def main() -> None:
    text = normalise_caps_sections(SRC.read_text(encoding="utf-8").replace("\n", " "))
    anchors = [m for m in re.finditer(r"Description\.? (?=[A-Z])", text) if m.start() > 12000]

    headers = []  # (header_start, name, aka, body_start)
    prev_end = 0
    for m in anchors:
        chunk = text[prev_end:m.start()]
        off, name, aka = find_header(chunk[-1500:])
        start = m.start() - len(chunk[-1500:]) + off
        headers.append((start, name, aka, m.start()))
        prev_end = m.end()

    records, bad = [], 0
    for i, (start, name, aka, body_start) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        rec = {"name": name, "aka": aka, **parse_body(text[body_start:end])}
        rec["name"] = clean_name(name, rec["description"])
        if rec["name"] in DROP_NAMES:
            continue
        if len(name.split()) > MAX_NAME_WORDS or not rec["description"]:
            bad += 1
            rec["suspect"] = True
        if len(name.split()) > MAX_NAME_WORDS:
            continue  # intro / non-entry text
        records.append(rec)

    # hand-written entries for ones the transcript garbled (e.g. Straw Man)
    if MANUAL.exists():
        records += [json.loads(l) for l in MANUAL.read_text(encoding="utf-8").splitlines() if l.strip()]

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} entries ({bad} suspect) to {OUT}")


if __name__ == "__main__":
    main()
