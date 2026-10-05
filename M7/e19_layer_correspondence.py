"""E19 -- recover the layer correspondence instead of being told it.

The distillation result currently works because the analysis KNOWS child layer i came from
parent layer 2i. That is circular: to verify lineage the detector needs to know part of the
derivation. This removes the assumption.

Method: compute the full parent x child alignment matrix S, then find the maximum-score
MONOTONIC correspondence by dynamic programming -- child layers must map to strictly
increasing parent layers, but nothing fixes the stride. Ordering is assumed; selection is not.

The honest test is not whether the recovered mapping scores well on derivatives. Maximising
over mappings inflates EVERY pair, negatives included, so both classes are rescored under the
same DP and the question is whether the separation survives.
"""
import json, itertools, gc
import os as _os, pathlib as _pl
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
import numpy as np, torch
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment
torch.set_grad_enabled(False)
OUT = _ROOT/"M7"/"e19_layer_correspondence.json"

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

PAIRS = [
 ("P+","distill","google-bert/bert-base-uncased","distilbert/distilbert-base-uncased"),
 ("P+","distill","google-bert/bert-base-cased","distilbert/distilbert-base-cased"),
 ("P+","distill","google-bert/bert-base-multilingual-cased","distilbert/distilbert-base-multilingual-cased"),
 ("P+","distill","FacebookAI/roberta-base","distilbert/distilroberta-base"),
 ("P+","distill","openai-community/gpt2","distilbert/distilgpt2"),
 ("P~","same-recipe","EleutherAI/pythia-70m","EleutherAI/pythia-70m-deduped"),
 ("P~","same-recipe","EleutherAI/pythia-160m","EleutherAI/pythia-160m-deduped"),
 ("P~","same-recipe","EleutherAI/pythia-410m","EleutherAI/pythia-410m-deduped"),
 ("N~","same-family","google-bert/bert-base-uncased","nlpaueb/legal-bert-base-uncased"),
 ("N~","same-family","google-bert/bert-base-uncased","microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"),
 ("P-","unrelated","google-bert/bert-base-uncased","FacebookAI/roberta-base"),
]
rows=[]
print(f"{'cls':<4}{'pair':<44}{'index':>8}{'stride':>8}{'DP':>8}  recovered mapping")
print("-"*104)
for cls,kind,a,b in PAIRS:
    try:
        ma=AutoModel.from_pretrained(a).eval(); mb=AutoModel.from_pretrained(b).eval()
        ua,ub=ups(ma),ups(mb)
        m,n=len(ua),len(ub)
        S=np.full((m,n),np.nan)
        for i in range(m):
            for j in range(n):
                v=lap_colnorm(ua[i],ub[j])
                if v is not None: S[i,j]=v
        del ma,mb; gc.collect()
        if np.isnan(S).all(): raise ValueError("no comparable layer pairs")
        S=np.nan_to_num(S,nan=-1.0)
        idx=float(np.mean([S[k,k] for k in range(min(m,n))]))
        stride=None
        if m>n and n>0 and m%n==0:
            st=m//n; stride=float(np.mean([S[k*st,k] for k in range(n)]))
        path,dp=monotonic_dp(S)
        exp=[int(k*(m//n)) for k in range(n)] if (m>n and n and m%n==0) else [int(k) for k in range(n)]
        rows.append(dict(cls=cls,kind=kind,a=a,b=b,m=m,n=n,index=idx,stride=stride,
                         dp=dp,path=path,expected_stride_path=exp,matches_stride=(path==exp)))
        print(f"{cls:<4}{(a.split('/')[-1]+' -> '+b.split('/')[-1])[:42]:<44}"
              f"{idx:>8.4f}{(stride if stride is not None else float('nan')):>8.4f}{dp:>8.4f}  "
              f"{path}{'  == stride' if path==exp else ''}")
    except Exception as e:
        print(f"{cls:<4}{(a.split('/')[-1]+' -> '+b.split('/')[-1])[:42]:<44}  FAILED {type(e).__name__}: {e}")
        rows.append(dict(cls=cls,kind=kind,a=a,b=b,error=f"{type(e).__name__}: {e}"))
OUT.write_text(json.dumps(rows,indent=1))
ok=[r for r in rows if r.get("dp") is not None]
pos=[r["dp"] for r in ok if r["cls"]=="P+"]; neg=[r["dp"] for r in ok if r["cls"]!="P+"]
if pos and neg:
    print(f"\nunder the recovered (DP) mapping, both classes rescored:")
    print(f"  derived  n={len(pos)}  {min(pos):.4f}-{max(pos):.4f}")
    print(f"  negative n={len(neg)}  {min(neg):.4f}-{max(neg):.4f}")
    print(f"  MARGIN {min(pos)-max(neg):+.4f}   separation {'HOLDS' if min(pos)>max(neg) else 'LOST'}")
    ds=[r for r in ok if r["kind"]=="distill"]
    print(f"  distillation pairs whose recovered mapping equals the stride mapping: "
          f"{sum(1 for r in ds if r['matches_stride'])}/{len(ds)}")
print(f"\nwrote {OUT.name}")
