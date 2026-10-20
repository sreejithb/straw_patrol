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
from huggingface_hub.errors import EntryNotFoundError

REPO = "sreejithb/fallacy-dataset"
ROOT = Path(__file__).parent
FILES = {  # path inside the repo -> local path
    "train.jsonl": "data/generated/train.jsonl",
    "train_extra.jsonl": "data/generated/train_extra.jsonl",
    "eval.jsonl": "data/generated/eval.jsonl",
    "train_none.jsonl": "data/generated/train_none.jsonl",
    "eval_none.jsonl": "data/generated/eval_none.jsonl",
    "calib_fallacy.jsonl": "data/generated/calib_fallacy.jsonl",
    "calib_none.jsonl": "data/generated/calib_none.jsonl",
    "train_real.jsonl": "data/generated/train_real.jsonl",
    "eval_real.jsonl": "data/generated/eval_real.jsonl",
    "alias_groups.json": "data/alias_groups.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", help="dataset version (tag or commit); default is the latest")
    args = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        try:  # inside Colab, read the notebook secret
            from google.colab import userdata
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN").strip()
            print("HF_TOKEN read from the Colab secret")
        except ImportError:
            print("not in Colab: using the token saved by `hf auth login`, if any")
        except Exception as e:
            raise SystemExit(f"Could not read the Colab secret HF_TOKEN ({type(e).__name__}: {e}).\n"
                             "In Colab, read the secret in a notebook cell and export it before running this script:\n"
                             "  import os; from google.colab import userdata; os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')")
    from huggingface_hub import HfApi
    try:
        print("authenticated as:", HfApi().whoami()["name"])
    except Exception:
        raise SystemExit("The token is missing or invalid (Hugging Face rejected it). Create a new token and update the secret.")
    for remote, local in FILES.items():
        try:
            src = hf_hub_download(REPO, remote, repo_type="dataset", revision=args.rev)
        except EntryNotFoundError:
            print(f"{remote}: not in version {args.rev}, skipped")  # older versions lack the newer files
            continue
        dest = ROOT / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        print(f"{remote} -> {local}")


if __name__ == "__main__":
    main()
