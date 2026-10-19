"""Upload the training/eval files to the private Hugging Face dataset repo.

  python upload_data.py            # shows what would be uploaded, asks for confirmation
  python upload_data.py --tag v2   # also tags this upload so code can pin the data version

Needs a Hugging Face token with write access (run `hf auth login` once).
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO = "sreejithb/fallacy-dataset"
ROOT = Path(__file__).parent
FILES = {  # local path -> path inside the dataset repo
    "data/generated/train.jsonl": "train.jsonl",
    "data/generated/train_extra.jsonl": "train_extra.jsonl",
    "data/generated/eval.jsonl": "eval.jsonl",
    "data/alias_groups.json": "alias_groups.json",
}
CARD = """---
pretty_name: Fallacy dataset
language: en
---
# Fallacy dataset

Synthetic arguments labelled with the logical fallacy they commit, in chat format
(`messages`: system, user = argument, assistant = `Fallacy: <name>` + short explanation).

- `train.jsonl`: 1,222 examples (first build, reviewed)
- `train_extra.jsonl`: 977 examples added later to bring every fallacy to about 12 examples
- `eval.jsonl`: 135 held-out examples, never used for training or early stopping
- `alias_groups.json`: gold label -> other fallacy names that also count as correct when scoring

Generated with an LLM from fallacy names and definitions, then filtered by automatic checks and an LLM judge.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="git tag to create on the dataset repo after uploading, e.g. v2")
    args = ap.parse_args()
    api = HfApi()
    print(f"Upload to {REPO} (private dataset):")
    for local, remote in FILES.items():
        print(f"  {local} ({(ROOT / local).stat().st_size / 1024:.0f} KB) -> {remote}")
    if input("Type 'yes' to upload: ").strip().lower() != "yes":
        raise SystemExit("cancelled")
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
    for local, remote in FILES.items():
        api.upload_file(path_or_fileobj=str(ROOT / local), path_in_repo=remote, repo_id=REPO, repo_type="dataset")
    if args.tag:
        api.create_tag(REPO, tag=args.tag, repo_type="dataset")
    print("done")


if __name__ == "__main__":
    main()
