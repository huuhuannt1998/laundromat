"""E47 fetch -- read-only downloads for the PolyPythias separate-seed check (nothing is uploaded).

Main revisions follow M9/fetch_one.py (config/tokenizer files plus ONE weight format: safetensors
when the repo serves it, else .bin), so 'as served' means what the paper's earlier fetches meant.
step0 revisions: weights + config only, for the initialization screen.
"""
import os, sys, json, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download, HfApi
api = HfApi()
P = "EleutherAI/pythia-"
MAIN = [f"{P}{s}-seed{k}" for s in ("70m", "160m", "410m") for k in (1, 2, 3)] + \
       [f"{P}160m-data-seed1", f"{P}160m-weight-seed1"]
STEP0 = [f"{P}{s}{suf}" for s in ("70m", "160m", "410m") for suf in ("", "-deduped", "-seed1", "-seed2", "-seed3")] + \
        [f"{P}160m-data-seed1", f"{P}160m-weight-seed1"]

def fmt_of(r, rev):
    files = [f.rfilename for f in api.model_info(r, revision=rev).siblings]
    return "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"

log = {}
for r in MAIN:
    t = time.time(); fmt = fmt_of(r, None)
    p = snapshot_download(r, allow_patterns=["*.json", "*.txt", "*.model", fmt], max_workers=4)
    log[f"{r}@main"] = dict(path=p, fmt=fmt, secs=round(time.time()-t, 1)); print(r, "main", fmt, log[f"{r}@main"]["secs"], flush=True)
for r in STEP0:
    t = time.time(); fmt = fmt_of(r, "step0")
    p = snapshot_download(r, revision="step0", allow_patterns=["config.json", fmt], max_workers=4)
    log[f"{r}@step0"] = dict(path=p, fmt=fmt, secs=round(time.time()-t, 1)); print(r, "step0", fmt, log[f"{r}@step0"]["secs"], flush=True)
json.dump(log, open(os.path.join(os.path.dirname(__file__), "e47_fetch_paths.json"), "w"), indent=1)
print("DONE-FETCH")
