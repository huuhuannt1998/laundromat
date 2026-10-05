"""M4 — LAP-ALIGNED POSITIONAL SIMILARITY: the candidate that could be BOTH sound and robust.

The first M4 run showed a hard tension:
  SVD spectral (permutation-INVARIANT):  P+ 1.0000 | P~ 0.9924 | P- 0.9460 | laundered 0.9955
      -> robust to laundering, but cannot discriminate at all
  raw positional (permutation-SENSITIVE): P+ 0.996 | P~ 0.1341 | P- 0.0003 | laundered 0.2413
      -> discriminates perfectly, but free permutation destroys it
LAP alignment is the way out: solve the assignment problem to UNDO the permutation, then compare
VALUES. Permutation-invariant by construction, value-discriminative by design. This is AWM's
mechanism; the question is whether it is also SOUND on the P~ arm that defeats MPK.
"""
import os,sys,json,copy,pathlib,torch,numpy as np
os.environ["HF_HUB_DISABLE_XET"]="1"
sys.path.insert(0,str(_ROOT) + "/M2/transforms")
from transformers import AutoModelForCausalLM
from scipy.optimize import linear_sum_assignment
import laundry as LD

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)
RES=pathlib.Path(str(_ROOT) + "/M4")

def mlp_pairs(model):
    """Per-layer (gate/up-like, down-like) matrices whose SHARED axis is the permutable one."""
    ups, downs = [], []
    for n,p in model.named_parameters():
        if p.ndim!=2: continue
        if any(k in n for k in ("up_proj.weight","dense_h_to_4h.weight","fc_in.weight","c_fc.weight")):
            ups.append((n,p))
        if any(k in n for k in ("down_proj.weight","dense_4h_to_h.weight","fc_out.weight","c_proj.weight")):
            downs.append((n,p))
    return ups, downs

def lap_align_score(A, B, max_dim=1536):
    """Align B's intermediate neurons to A's by LAP on cosine similarity, then score."""
    A=A.float(); B=B.float()
    if A.shape!=B.shape: return None
    n=A.shape[0]
    if n>max_dim:
        idx=torch.arange(0,n,max(1,n//max_dim))[:max_dim]; A=A[idx]; B=B[idx]; n=A.shape[0]
    An=A/(A.norm(dim=1,keepdim=True)+1e-12); Bn=B/(B.norm(dim=1,keepdim=True)+1e-12)
    S=(An@Bn.T).numpy()                        # cost matrix: neuron i of A vs neuron j of B
    r,c=linear_sum_assignment(-S)              # maximise total cosine
    return float(S[r,c].mean())                # mean matched-neuron cosine

def score_pair(ma, mb):
    ua,da=mlp_pairs(ma); ub,db=mlp_pairs(mb)
    L=min(len(ua),len(ub))
    if L==0: return float("nan"), float("nan")
    lap=[]; naive=[]
    for i in range(L):
        A=ua[i][1]; B=ub[i][1]
        if A.shape!=B.shape: continue
        s=lap_align_score(A,B)
        if s is not None: lap.append(s)
        An=A.float().flatten(); Bn=B.float().flatten()
        k=min(An.numel(),4096); st=max(1,An.numel()//k)
        x=An[::st][:k]; y=Bn[::st][:k]
        naive.append(float(torch.nn.functional.cosine_similarity(x[None],y[None]).item()))
    return (float(np.mean(lap)) if lap else float("nan"),
            float(np.mean(naive)) if naive else float("nan"))

def load(mid, launder=False):
    m=AutoModelForCausalLM.from_pretrained(mid).eval()
    if launder:
        LD.X1a_mlp_permute(m,1.0,11); LD.X1b_head_permute(m,11); LD.X2_norm_scale(m,0.5,11)
    return m

PAIRS=[
 ("IDENTITY","HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M",False),
 ("P+ derived","HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M-Instruct",False),
 ("P+ derived","Qwen/Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B-Instruct",False),
 ("P~ SAME-RECIPE INDEP","EleutherAI/pythia-160m","EleutherAI/pythia-160m-deduped",False),
 ("P- unrelated","HuggingFaceTB/SmolLM2-135M","EleutherAI/pythia-160m",False),
 ("LAUNDERED","HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M-Instruct",True),
 ("LAUNDERED","Qwen/Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B-Instruct",True),
]
rows=[]
print(f"{'class':<24}{'pair':<48}{'LAP-aligned':>12}{'naive-pos':>11}")
print("-"*95)
for cls,a,b,l in PAIRS:
    ma=load(a); mb=load(b,l)
    lap,naive=score_pair(ma,mb)
    rows.append({"class":cls,"a":a,"b":b,"laundered":l,"lap":lap,"naive":naive})
    print(f"{cls:<24}{(a.split('/')[-1]+' | '+b.split('/')[-1]):<48}{lap:>12.4f}{naive:>11.4f}")
    del ma,mb
(RES/"lap_defence.json").write_text(json.dumps(rows,indent=1))
print("DONE-LAP")
