"""E34 -- score the alignment defence on MultiBERTs same-recipe pairs.

Review round 4 (meta-review item 5; Reviewer A W4, Reviewer C W6): the defence's full-corpus
margin is set by ONE same-family negative (legal-bert, 0.4513, against a lowest derivative of
0.4941 under the fixed mapping and 0.7918 under the recovered one). The corpus already holds a
source of harder negatives that section IX never scored: the MultiBERTs, twenty-five BERT-base
pretraining runs differing only in seed and data order. Section V uses six of their pairs as
same-recipe negatives for the verifier; this scores the SAME six pairs with the alignment
signal.

Declared before running. These are negatives. The result can go three ways and all are
reported: every pair below 0.4513 leaves the margins as printed; any pair in [0.4513, 0.4941)
becomes the binding negative and narrows the fixed-mapping margin; any pair at or above 0.4941
(fixed) or 0.7918 (recovered) breaks the perfect ordering under that mapping.

The scorer is E18/E19's, copied verbatim (ups, lap_colnorm, monotonic_dp) rather than imported,
because importing those scripts would run them. Equal depth (12 vs 12), so the fixed mapping is
index-to-index as in E18; the recovered mapping is E19's monotonic DP over the full 12x12
matrix. Weights are read from the local HF cache (HF_HUB_OFFLINE=1); snapshot hashes are
recorded. Writes a NEW file, M7/e34_multiberts_alignment.json. No existing result is touched.
"""
import json, gc, time
import os as _os, pathlib as _pl
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
import numpy as np, torch
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment
from huggingface_hub import snapshot_download
torch.set_grad_enabled(False)
OUT = _ROOT/"M7"/"e34_multiberts_alignment.json"

# ---- verbatim from M7/e19_layer_correspondence.py (identical to M7/e18_defence.py's) ----
KEYS = ("up_proj.weight","dense_h_to_4h.weight","c_fc.weight","fc_in.weight",
        "intermediate.dense.weight","ffn.lin1.weight")
def ups(m):
    return [p for n,p in m.named_parameters() if p.ndim==2 and any(k in n for k in KEYS)]

def lap_colnorm(A,B,max_feat=1536):
    A,B=A.float(),B.float()
    if A.shape!=B.shape: return None
    A=A/(A.norm(dim=0,keepdim=True)+1e-12); B=B/(B.norm(dim=0,keepdim=True)+1e-12)
    if A.shape[1]>max_feat:
        f=torch.arange(0,A.shape[1],max(1,A.shape[1]//max_feat))[:max_feat]; A,B=A[:,f],B[:,f]
    An=A/(A.norm(dim=1,keepdim=True)+1e-12); Bn=B/(B.norm(dim=1,keepdim=True)+1e-12)
    S=(An@Bn.T).numpy(); r,c=linear_sum_assignment(-S); return float(S[r,c].mean())

def monotonic_dp(S):
    """S is m x n (parent x child). Pick j_1<...<j_n maximising sum S[j_k, k]."""
    m,n = S.shape
    if n>m: return None,None
    NEG=-1e18
    f=np.full((n,m),NEG); bk=np.full((n,m),-1,int)
    for j in range(m): f[0,j]=S[j,0]
    for k in range(1,n):
        best=NEG; arg=-1
        for j in range(m):
            if j-1>=0 and f[k-1,j-1]>best: best=f[k-1,j-1]; arg=j-1
            if best>NEG/2: f[k,j]=best+S[j,k]; bk[k,j]=arg
    j=int(np.argmax(f[n-1])); path=[j]
    for k in range(n-1,0,-1):
        j=bk[k,j]; path.append(j)
    path=path[::-1]
    return [int(v) for v in path], float(np.mean([S[path[k],k] for k in range(n)]))
# ---- end verbatim ----

# The six pairs of M7/e5_same_recipe.py, in its order.
M = "google/multiberts-seed_"
PAIRS = [(f"{M}0", f"{M}1"), (f"{M}0", f"{M}2"), (f"{M}1", f"{M}2"),
         (f"{M}2", f"{M}3"), (f"{M}3", f"{M}4"), (f"{M}0", f"{M}4")]

def snap(repo):
    return _pl.Path(snapshot_download(repo, local_files_only=True)).name

rows=[]
if OUT.exists():                       # resume: keep pairs already scored by an interrupted run
    prev=json.loads(OUT.read_text())
    rows=[r for r in (prev["rows"] if isinstance(prev,dict) else prev) if (r["a"],r["b"]) in PAIRS]
done={(r["a"],r["b"]) for r in rows}
print(f"{'pair':<34}{'fixed':>8}{'DP':>8}  recovered mapping", flush=True)
for a,b in PAIRS:
    if (a,b) in done:
        r=next(x for x in rows if (x["a"],x["b"])==(a,b))
        print(f"{a.split('_')[-1]+' / '+b.split('_')[-1]:<34}{r['fixed']:>8.4f}{r['dp']:>8.4f}  (resumed)", flush=True)
        continue
    t0=time.time()
    ma=AutoModel.from_pretrained(a).eval(); mb=AutoModel.from_pretrained(b).eval()
    ua,ub=ups(ma),ups(mb); m,n=len(ua),len(ub)
    S=np.full((m,n),np.nan)
    for i in range(m):
        for j in range(n):
            v=lap_colnorm(ua[i],ub[j])
            if v is not None: S[i,j]=v
    del ma,mb; gc.collect()
    fixed=float(np.mean([S[k,k] for k in range(min(m,n))]))
    path,dp=monotonic_dp(np.nan_to_num(S,nan=-1.0))
    rows.append(dict(cls="N-SR", suite="multiberts", a=a, b=b, snapshot_a=snap(a),
                     snapshot_b=snap(b), m=m, n=n, fixed=fixed, per_layer_fixed=[float(S[k,k]) for k in range(min(m,n))],
                     dp=dp, path=path, identity_path=(path==list(range(n))),
                     secs=round(time.time()-t0,1)))
    OUT.write_text(json.dumps(rows,indent=1))
    print(f"{a.split('_')[-1]+' / '+b.split('_')[-1]:<34}{fixed:>8.4f}{dp:>8.4f}  {path}", flush=True)

rows=sorted(rows,key=lambda r: PAIRS.index((r["a"],r["b"])))
fx=[r["fixed"] for r in rows]; dps=[r["dp"] for r in rows]
summary=dict(n=len(rows), fixed_min=min(fx), fixed_max=max(fx), dp_min=min(dps), dp_max=max(dps),
             all_identity_path=all(r["identity_path"] for r in rows),
             printed_binding_negative=0.4513, printed_lowest_derived_fixed=0.4941,
             printed_lowest_derived_dp=0.7918,
             fixed_margin_if_included=0.4941-max(0.4513,max(fx)),
             dp_margin_if_included=0.7918-max(0.4513,max(dps)))
OUT.write_text(json.dumps(dict(rows=rows, summary=summary), indent=1))
print(json.dumps(summary, indent=1))
print(f"wrote {OUT.relative_to(_ROOT)}")
