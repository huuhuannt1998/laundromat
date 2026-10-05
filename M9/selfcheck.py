"""Prove the memory-safe defence rewrite is numerically identical to M4's original.

M4/p_tilde_arm.py holds both models in fp32 and subsamples columns AFTER column
normalising. M9 subsamples BEFORE, in fp16, one model at a time. That reordering is
only exact because column normalisation is per-column. This asserts it on a cached
model rather than taking the argument on faith -- if it fails, every E23 number would
be incomparable to the existing P~ ladder.
"""
import sys, pathlib, numpy as np, torch
from scipy.optimize import linear_sum_assignment
torch.set_grad_enabled(False)
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
sys.path.insert(0, str(ROOT / "M9"))
MAX_FEAT = 1536

def m4_original(A, B):                     # verbatim from M4/p_tilde_arm.py
    A = A.float(); B = B.float()
    if A.shape != B.shape: return None
    A = A / (A.norm(dim=0, keepdim=True) + 1e-12); B = B / (B.norm(dim=0, keepdim=True) + 1e-12)
    if A.shape[1] > MAX_FEAT:
        f = torch.arange(0, A.shape[1], max(1, A.shape[1] // MAX_FEAT))[:MAX_FEAT]
        A = A[:, f]; B = B[:, f]
    An = A / (A.norm(dim=1, keepdim=True) + 1e-12); Bn = B / (B.norm(dim=1, keepdim=True) + 1e-12)
    S = (An @ Bn.T).numpy(); r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())

def m9_new(A, B):                          # subsample first, as reduce_model does
    from e23_scale import lap_colnorm
    def red(X):
        if X.shape[1] > MAX_FEAT:
            f = torch.arange(0, X.shape[1], max(1, X.shape[1] // MAX_FEAT))[:MAX_FEAT]
            X = X[:, f]
        return X.clone().float().numpy()
    return lap_colnorm(red(A), red(B))

from transformers import AutoModelForCausalLM
KEYS = ("up_proj.weight", "dense_h_to_4h.weight", "c_fc.weight", "fc_in.weight")
def ups(mid, dt):
    m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dt).eval()
    return [p.detach() for n, p in m.named_parameters()
            if p.ndim == 2 and any(k in n for k in KEYS)]

a, b = "EleutherAI/pythia-70m", "EleutherAI/pythia-70m-deduped"
A32, B32 = ups(a, torch.float32), ups(b, torch.float32)   # what M4 did
A16, B16 = ups(a, torch.float16), ups(b, torch.float16)   # what M9 does
worst = 0.0
for i in range(min(len(A32), len(B32))):
    o, n = m4_original(A32[i], B32[i]), m9_new(A16[i], B16[i])
    worst = max(worst, abs(o - n))
    print(f"  layer {i}: M4={o:.6f}  M9={n:.6f}  d={abs(o-n):.2e}")
print(f"\nmax |M4-M9| = {worst:.2e}")
print("PASS -- comparable to the existing ladder" if worst < 1e-3
      else "FAIL -- E23 would not be comparable to the 1b ladder")
