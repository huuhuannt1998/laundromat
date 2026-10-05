"""E38 -- score the third rejected derivative with the alignment signal.

Review round 6 (R4): section IX says alignment "does not recover the two derivatives the verifier
rejects", but the paper counts THREE rejections. The third, the two-stage adaptation
cardiffnlp/twitter-roberta-base-sentiment-latest (identity 0.5375, Not Matched), is
equal-width and equal-depth with roberta-base and was never scored by alignment.

Scorer: E18's, copied verbatim (ups, lap_colnorm). Equal depth, so the fixed mapping is
index-to-index as in E18; the recovered (DP) mapping is E19's monotonic DP over the full matrix.
Not added to the thirty-pair statistics. Writes a NEW file, M7/e38_sentiment_alignment.json.
"""
import json, gc
import os as _os, pathlib as _pl
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
import numpy as np, torch
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment
torch.set_grad_enabled(False)
OUT = _ROOT/"M7"/"e38_sentiment_alignment.json"

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

a, b = "FacebookAI/roberta-base", "cardiffnlp/twitter-roberta-base-sentiment-latest"
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
rec=dict(cls="P+ (rejected by the verifier)", kind="cont-pre+fine-tune", a=a, b=b, m=m, n=n,
         fixed=fixed, dp=dp, path=path, threshold_0p4727_passed=fixed>0.472726018478473,
         printed_binding_negative=0.4513)
OUT.write_text(json.dumps(rec, indent=1))
print(json.dumps(rec))
