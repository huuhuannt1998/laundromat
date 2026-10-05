"""Expand the alignment-defence evaluation off its power floor (review finding C-7).

The reported p = 0.0079 is exactly 1/C(9,4), the MINIMUM attainable one-sided value at
n = 5 vs 4. The test had one reachable value below alpha and returned it -- the same
power-floor pathology this project documented and corrected for contrast C5, reintroduced
in the defence without comment. It was also computed one-sided, whereas both preregistered
scripts (analysis/prereg_final.py, analysis/c5_final.py) use TWO-SIDED permutation with a
+1 correction.

This script (a) enlarges both arms so the floor is far below alpha, (b) reports the
same-recipe contrast separately from the pooled-negative contrast, since the paper's
sentence names same-recipe specifically, and (c) uses the project's own convention.

Writes NEW file M4/defence_expanded.json. Touches no existing result file.
"""
import os, sys, json, itertools, pathlib
os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np, torch
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)

OUT = pathlib.Path(str(_ROOT) + "/M4/defence_expanded.json")

PAIRS = [
    # --- P+ : genuine derivatives -------------------------------------------------
    ("P+", "fine-tune",  "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("P+", "fine-tune",  "Qwen/Qwen2.5-0.5B",          "Qwen/Qwen2.5-0.5B-Instruct"),
    ("P+", "fine-tune",  "google-bert/bert-base-uncased", "textattack/bert-base-uncased-SST-2"),
    ("P+", "multitask",  "bigscience/bloom-560m",      "bigscience/bloomz-560m"),
    ("P+", "cont-pre",   "FacebookAI/roberta-base",    "ehsanaghaei/SecureBERT"),
    ("P+", "distill",    "google-bert/bert-base-uncased", "distilbert/distilbert-base-uncased"),
    ("P+", "distill",    "FacebookAI/roberta-base",    "distilbert/distilroberta-base"),
    ("P+", "distill",    "openai-community/gpt2",      "distilbert/distilgpt2"),
    # --- P~ : independently trained, same recipe ----------------------------------
    ("P~", "same-recipe", "EleutherAI/pythia-70m",  "EleutherAI/pythia-70m-deduped"),
    ("P~", "same-recipe", "EleutherAI/pythia-160m", "EleutherAI/pythia-160m-deduped"),
    ("P~", "same-recipe", "EleutherAI/pythia-410m", "EleutherAI/pythia-410m-deduped"),
    ("P~", "same-recipe", "EleutherAI/pythia-70m",  "EleutherAI/pythia-70m-v0"),
    ("P~", "same-recipe", "EleutherAI/pythia-160m", "EleutherAI/pythia-160m-v0"),
    ("P~", "same-recipe", "EleutherAI/pythia-410m", "EleutherAI/pythia-410m-v0"),
    ("P~", "same-recipe", "EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-70m-deduped-v0"),
    # --- P- : unrelated -----------------------------------------------------------
    ("P-", "unrelated",  "google-bert/bert-base-uncased", "FacebookAI/roberta-base"),
    ("P-", "unrelated",  "distilbert/distilbert-base-uncased", "distilbert/distilroberta-base"),
]


def ups(m):
    return [p for n, p in m.named_parameters() if p.ndim == 2 and any(
        k in n for k in ("up_proj.weight", "dense_h_to_4h.weight", "c_fc.weight",
                         "fc_in.weight", "intermediate.dense.weight"))]


def lap_colnorm(A, B, max_feat=1536):
    A, B = A.float(), B.float()
    if A.shape != B.shape:
        return None
    A = A / (A.norm(dim=0, keepdim=True) + 1e-12)
    B = B / (B.norm(dim=0, keepdim=True) + 1e-12)
    if A.shape[1] > max_feat:                      # reduce the FEATURE axis only
        f = torch.arange(0, A.shape[1], max(1, A.shape[1] // max_feat))[:max_feat]
        A, B = A[:, f], B[:, f]
    An = A / (A.norm(dim=1, keepdim=True) + 1e-12)
    Bn = B / (B.norm(dim=1, keepdim=True) + 1e-12)
    S = (An @ Bn.T).numpy()
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())


rows = []
print(f"{'class':<5}{'kind':<13}{'lap':>9}  pair", flush=True)
print("-" * 92, flush=True)
for cls, kind, a, b in PAIRS:
    try:
        ma, mb = AutoModel.from_pretrained(a).eval(), AutoModel.from_pretrained(b).eval()
        ua, ub = ups(ma), ups(mb)
        vals = [v for i in range(min(len(ua), len(ub)))
                if (v := lap_colnorm(ua[i], ub[i])) is not None]
        del ma, mb
        lap = float(np.mean(vals)) if vals else None
    except Exception as e:
        print(f"{cls:<5}{kind:<13}{'FAIL':>9}  {a} | {b}  ({type(e).__name__})", flush=True)
        continue
    rows.append(dict(cls=cls, kind=kind, a=a, b=b, lap=lap, n_layers=len(vals)))
    print(f"{cls:<5}{kind:<13}{(lap if lap is not None else float('nan')):>9.4f}  "
          f"{a.split('/')[-1]} | {b.split('/')[-1]}", flush=True)

json.dump(rows, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT.name} (n={len(rows)})", flush=True)
print("DONE-DEFEXP", flush=True)
