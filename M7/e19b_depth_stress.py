"""E19b -- the stress test E19 needs.

Every negative in E19 has EQUAL depth, where a monotonic search has no freedom: the
identity mapping is already optimal, so the DP cannot inflate the score. That is not
evidence the method is safe. The dangerous case is a DEPTH-MISMATCHED negative, where
the DP gets to choose 6 of 12 parent layers and could manufacture a match.

Three such negatives:
  - bert-base-uncased (12L) vs google/bert_uncased_L-6_H-768_A-12 (6L): same architecture
    family, same hidden size, independently pretrained from scratch. The hardest case.
  - bert-base-uncased (12L) vs distilroberta-base (6L): unrelated, depth-mismatched.
  - roberta-base (12L) vs distilbert-base-uncased (6L): unrelated, depth-mismatched.
"""
import json, gc
import os as _os, pathlib as _pl
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
import numpy as np, torch
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment
torch.set_grad_enabled(False)

KEYS=("up_proj.weight","dense_h_to_4h.weight","c_fc.weight","fc_in.weight",
      "intermediate.dense.weight","ffn.lin1.weight")
def ups(m): return [p for n,p in m.named_parameters() if p.ndim==2 and any(k in n for k in KEYS)]
def lap(A,B,max_feat=1536):
    A,B=A.float(),B.float()
    if A.shape!=B.shape: return None
    A=A/(A.norm(dim=0,keepdim=True)+1e-12); B=B/(B.norm(dim=0,keepdim=True)+1e-12)
    if A.shape[1]>max_feat:
        f=torch.arange(0,A.shape[1],max(1,A.shape[1]//max_feat))[:max_feat]; A,B=A[:,f],B[:,f]
    An=A/(A.norm(dim=1,keepdim=True)+1e-12); Bn=B/(B.norm(dim=1,keepdim=True)+1e-12)
    S=(An@Bn.T).numpy(); r,c=linear_sum_assignment(-S); return float(S[r,c].mean())
def monotonic_dp(S):
    m,n=S.shape
    if n>m: return None,None
    NEG=-1e18; f=np.full((n,m),NEG); bk=np.full((n,m),-1,int)
    for j in range(m): f[0,j]=S[j,0]
    for k in range(1,n):
        best=NEG; arg=-1
        for j in range(m):
            if j-1>=0 and f[k-1,j-1]>best: best=f[k-1,j-1]; arg=j-1
            if best>NEG/2: f[k,j]=best+S[j,k]; bk[k,j]=arg
    j=int(np.argmax(f[n-1])); path=[j]
    for k in range(n-1,0,-1): j=bk[k,j]; path.append(j)
    path=path[::-1]
    return [int(v) for v in path], float(np.mean([S[path[k],k] for k in range(n)]))

PAIRS=[("N~ depth-mismatched","google-bert/bert-base-uncased","google/bert_uncased_L-6_H-768_A-12"),
       ("P-  depth-mismatched","google-bert/bert-base-uncased","distilbert/distilroberta-base"),
       ("P-  depth-mismatched","FacebookAI/roberta-base","distilbert/distilbert-base-uncased")]
rows=[]
print(f"{'class':<22}{'pair':<50}{'m':>3}{'n':>3}{'stride':>8}{'DP':>8}  path")
print("-"*112)
for cls,a,b in PAIRS:
    try:
        ma=AutoModel.from_pretrained(a).eval(); mb=AutoModel.from_pretrained(b).eval()
        ua,ub=ups(ma),ups(mb); m,n=len(ua),len(ub)
        S=np.full((m,n),np.nan)
        for i in range(m):
            for j in range(n):
                v=lap(ua[i],ub[j])
                if v is not None: S[i,j]=v
        del ma,mb; gc.collect()
        if np.isnan(S).all(): raise ValueError("no comparable layers")
        S=np.nan_to_num(S,nan=-1.0)
        st=float(np.mean([S[k*(m//n),k] for k in range(n)])) if (m>n and m%n==0) else None
        path,dp=monotonic_dp(S)
        rows.append(dict(cls=cls,a=a,b=b,m=m,n=n,stride=st,dp=dp,path=path))
        print(f"{cls:<22}{(a.split('/')[-1]+' vs '+b.split('/')[-1])[:48]:<50}{m:>3}{n:>3}"
              f"{(st if st is not None else float('nan')):>8.4f}{dp:>8.4f}  {path}")
    except Exception as e:
        print(f"{cls:<22}{(a.split('/')[-1]+' vs '+b.split('/')[-1])[:48]:<50}  FAILED {type(e).__name__}: {e}")
        rows.append(dict(cls=cls,a=a,b=b,error=f"{type(e).__name__}: {e}"))
(_ROOT/"M7"/"e19b_depth_stress.json").write_text(json.dumps(rows,indent=1))
ok=[r for r in rows if r.get("dp") is not None]
if ok:
    worst=max(ok,key=lambda r:r["dp"])
    print(f"\nhighest depth-mismatched NEGATIVE under DP: {worst['dp']:.4f} "
          f"({worst['b'].split('/')[-1]})")
    print(f"lowest derived under DP (E19): 0.7918")
    print(f"margin against depth-mismatched negatives: {0.7918-worst['dp']:+.4f}  "
          f"{'HOLDS' if worst['dp']<0.7918 else 'LOST'}")
print("\nDONE-E19B")
