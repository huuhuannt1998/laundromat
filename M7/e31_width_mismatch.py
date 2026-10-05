"""E31 -- the alignment signal on the derivatives the verifier rejects, and on width-changing
children.

The review (2026-09-08) noted that Table 4 omits the two rejected derivatives that motivate the
paper, Intel/dynamic_tinybert and microsoft/xtremedistil-l6-h256-uncased, and every
width-changing child. Two cases:

1. Depth mismatch only. dynamic_tinybert is a 6-layer, 768-wide BERT (card: TinyBERT6L
   architecture, general-distilled from BERT-base, fine-tuned on SQuAD). MLP rows have the
   parent's shape, so the E19 procedure applies unchanged: full parent x child alignment
   matrix, then the maximum-score monotonic layer correspondence by dynamic programming.

2. Width mismatch. xtremedistil-l6-h256 (256-wide), TinyBERT_General_4L_312D (312-wide),
   MiniLM-L12-H384 (384-wide) and google/bert_uncased_L-4_H-256_A-4 (256-wide) have MLP rows
   that live in a different hidden space from the parent's, so the row cosine the assignment
   is built on is undefined. Rectangular assignment does not help: it needs a common column
   space. The only way to compare is to learn a linear map between the two hidden spaces; the
   shared 30,522-entry vocabulary gives one, by least squares from the parent's word
   embeddings to the child's (E_c ~ E_p B). Parent rows are mapped through B and assigned to
   child rows by rectangular LAP. This is an exploratory construction, not the E19 signal, and
   it is run with controls: TinyBERT_General (distilled from BERT-base, card) and xtremedistil
   (manifest D-DIST) are the positives; MiniLM (distilled from UniLMv2, card) and
   bert_uncased_L-4 (pretrained from scratch, card) are negatives of the same widths; and the
   same projection is applied to the equal-width bert-base-uncased -> distilbert-base-uncased
   pair, where B should be near the identity and the projected score near the E19 value
   (0.832), as a check that the projection itself does not destroy a real correspondence.

Output: M7/e31_width_mismatch.json.
"""
import json, gc, os as _os, pathlib as _pl
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents if (p / "MANIFEST-dataintegrity.txt").exists())))
import numpy as np, torch
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment
torch.set_grad_enabled(False)
OUT = _ROOT/"M7"/"e31_width_mismatch.json"
KEYS = ("up_proj.weight","dense_h_to_4h.weight","c_fc.weight","fc_in.weight","intermediate.dense.weight","ffn.lin1.weight")

def ups(m):  return [p.float() for n,p in m.named_parameters() if p.ndim==2 and any(k in n for k in KEYS)]
def emb(m):  return m.get_input_embeddings().weight.float()

def lap_colnorm(A,B,max_feat=1536):
    """E19's comparator: column l2-normalise, subsample columns, row-normalise, LAP, mean matched cosine.
    Rectangular when the row counts differ (assignment covers the smaller side)."""
    if A.shape[1]!=B.shape[1]: return None
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
    path=path[::-1]; return [int(v) for v in path], float(np.mean([S[path[k],k] for k in range(n)]))

def layer_matrix(ua, ub, proj=None):
    m,n=len(ua),len(ub); S=np.full((m,n),-1.0)
    for i in range(m):
        Ai = ua[i] @ proj if proj is not None else ua[i]
        for j in range(n):
            v=lap_colnorm(Ai, ub[j]);  S[i,j] = v if v is not None else -1.0
    return S

def score_pair(a, b, project):
    ma=AutoModel.from_pretrained(a).eval(); mb=AutoModel.from_pretrained(b).eval()
    ua,ub=ups(ma),ups(mb)
    rec=dict(parent=a, child=b, parent_layers=len(ua), child_layers=len(ub),
             parent_hidden=int(ua[0].shape[1]), child_hidden=int(ub[0].shape[1]),
             parent_rows=int(ua[0].shape[0]), child_rows=int(ub[0].shape[0]))
    proj=None
    if project:
        Ea, Eb = emb(ma), emb(mb)
        assert Ea.shape[0]==Eb.shape[0], "projection needs a shared vocabulary"
        proj = torch.linalg.lstsq(Ea, Eb).solution            # (h_parent x h_child)
        resid = float((Ea@proj - Eb).norm()/Eb.norm())
        rec.update(projection="least-squares map fitted on shared word embeddings",
                   projection_relative_residual=resid)
    elif ua[0].shape[1]!=ub[0].shape[1]:
        rec.update(applicable=False, reason="hidden widths differ; row cosine undefined without a projection")
        del ma,mb; gc.collect(); return rec
    S=layer_matrix(ua,ub,proj)
    m,n=S.shape
    rec["index"]=float(np.mean([S[k,k] for k in range(min(m,n))]))
    if m>n and n and m%n==0:
        st=m//n; rec["stride"]=float(np.mean([S[k*st,k] for k in range(n)]))
    path,dp=monotonic_dp(S); rec["dp"]=dp; rec["dp_path"]=path; rec["applicable"]=True
    del ma,mb; gc.collect(); return rec

BERT="google-bert/bert-base-uncased"
PAIRS=[
 ("derived (distill, rejected by verifier)", BERT, "Intel/dynamic_tinybert", False),
 ("derived (distill, rejected by verifier)", BERT, "microsoft/xtremedistil-l6-h256-uncased", False),
 ("derived (distill, rejected by verifier)", BERT, "microsoft/xtremedistil-l6-h256-uncased", True),
 ("derived (distill, no verdict as published)", BERT, "huawei-noah/TinyBERT_General_4L_312D", True),
 ("negative (distilled from UniLMv2)", BERT, "microsoft/MiniLM-L12-H384-uncased", True),
 ("negative (from scratch, 256-wide)", BERT, "google/bert_uncased_L-4_H-256_A-4", True),
 ("check: equal width, projected", BERT, "distilbert/distilbert-base-uncased", True),
 ("check: equal width, unprojected (E19)", BERT, "distilbert/distilbert-base-uncased", False),
]
rows=[]
print(f"{'class':<44}{'child':<40}{'index':>7}{'stride':>7}{'DP':>7}  path")
for cls,a,b,project in PAIRS:
    try:
        r=score_pair(a,b,project); r["cls"]=cls; r["projected"]=project
        if r.get("applicable"):
            print(f"{cls:<44}{b.split('/')[-1]:<40}{r['index']:>7.4f}{r.get('stride',float('nan')):>7.4f}{r['dp']:>7.4f}  {r['dp_path']}")
        else:
            print(f"{cls:<44}{b.split('/')[-1]:<40}  NOT APPLICABLE: {r['reason']}")
    except Exception as e:
        r=dict(cls=cls,parent=a,child=b,projected=project,error=f"{type(e).__name__}: {e}")
        print(f"{cls:<44}{b.split('/')[-1]:<40}  FAILED {r['error']}")
    rows.append(r)
OUT.write_text(json.dumps(rows,indent=1)); print(f"wrote {OUT.relative_to(_ROOT)}")
