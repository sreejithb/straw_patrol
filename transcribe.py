"""Transcribe every mp3 in ~/Desktop/LF/splits into individual .txt files,
then combine them into one big transcript."""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SPLITS_DIR = Path.home() / "Desktop" / "LF" / "splits"
OUT_DIR = Path(__file__).parent / "transcripts"
COMBINED_FILE = OUT_DIR / "full_transcript.txt"
MODEL = "whisper-1"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def transcribe_file(mp3_path: Path) -> str:
    with open(mp3_path, "rb") as f:
        result = client.audio.transcriptions.create(model=MODEL, file=f)
    return result.text


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    mp3_files = sorted(SPLITS_DIR.glob("*.mp3"))
    if not mp3_files:
        raise SystemExit(f"No mp3 files found in {SPLITS_DIR}")

    # 1. Individual transcripts (skips ones already done, so reruns resume)
    for mp3 in mp3_files:
        out = OUT_DIR / f"{mp3.stem}.txt"
        if out.exists():
            print(f"skip {mp3.name}")
            continue
        print(f"transcribing {mp3.name} ...")
        out.write_text(transcribe_file(mp3), encoding="utf-8")

    # 2. Combine into one big file, in order
    with open(COMBINED_FILE, "w", encoding="utf-8") as combined:
        for mp3 in mp3_files:
            text = (OUT_DIR / f"{mp3.stem}.txt").read_text(encoding="utf-8")
            combined.write(text.strip() + "\n\n")
    print(f"Wrote {COMBINED_FILE}")


if __name__ == "__main__":
    main()
