"""Explicit weight download for E23. Nothing else in M9 downloads on its own.

Downloads EXACTLY ONE weight format per repo. The first version of this script
called snapshot_download twice -- once excluding *.bin, then again including it --
which fetched both formats for every repo that ships both, wasting 26 GB on
pythia-6.9b-HC3 alone. The format is now decided up front from the Hub file list.
"""
import os, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download, HfApi

CORE = ["EleutherAI/pythia-2.8b", "EleutherAI/pythia-2.8b-deduped",
        "EleutherAI/pythia-6.9b", "EleutherAI/pythia-6.9b-deduped"]
DERIV = ["lambda/pythia-2.8b-deduped-synthetic-instruct", "pszemraj/pythia-6.9b-HC3"]
META = ["*.json", "*.txt", "*.model"]

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ids = CORE if which == "core" else (DERIV if which == "deriv" else CORE + DERIV)
    api = HfApi()
    for r in ids:
        files = [f.rfilename for f in api.model_info(r, files_metadata=False).siblings]
        fmt = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
        print(f"--- {r}  [{fmt}]", flush=True)
        snapshot_download(r, allow_patterns=META + [fmt], max_workers=4)
    print("DONE-FETCH")

if __name__ == "__main__":
    main()
