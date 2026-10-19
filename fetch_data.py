"""Download the dataset files from the private Hugging Face repo into the places the scripts expect.

  python fetch_data.py                 # latest
  python fetch_data.py --rev v2        # a tagged data version

Token: `hf auth login` on a laptop, or an HF_TOKEN environment variable / Colab secret named HF_TOKEN.
"""
import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "sreejithb/fallacy-dataset"
ROOT = Path(__file__).parent
FILES = {  # path inside the repo -> local path
    "train.jsonl": "data/generated/train.jsonl",
    "train_extra.jsonl": "data/generated/train_extra.jsonl",
    "eval.jsonl": "data/generated/eval.jsonl",
    "alias_groups.json": "data/alias_groups.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", help="dataset version (tag or commit); default is the latest")
    args = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        try:  # inside Colab, read the notebook secret
            from google.colab import userdata
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        except Exception:
            pass  # fall back to the token saved by `hf auth login`
    for remote, local in FILES.items():
        src = hf_hub_download(REPO, remote, repo_type="dataset", revision=args.rev)
        dest = ROOT / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        print(f"{remote} -> {local}")


if __name__ == "__main__":
    main()
