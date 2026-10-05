"""M4 — LAP + COLUMN-SCALE NORMALISATION.

lap_defence.py showed LAP-on-rows is SOUND (P+ 0.997 vs P~ 0.186) but only partly robust:
laundered recovered to 0.8807 (SmolLM2) and 0.3593 (Qwen), not ~0.99. The residue is X2:
LAP undoes the ROW permutation (X1a) but X2 divides the INPUT COLUMNS by alpha, and row
alignment cannot undo a column rescaling. Fix: normalise columns to unit norm BEFORE aligning.

Resulting invariances, by construction:
  column L2 normalisation  -> invariant to X2's per-column positive scaling
  row L2 normalisation     -> invariant to any per-row positive scaling
  LAP over rows            -> invariant to X1a's row permutation
and it still compares VALUES, so it should stay discriminative.
"""
import os,sys,json,pathlib,torch,numpy as np
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

def ups(model):
    out=[]
    for n,p in model.named_parameters():
        if p.ndim==2 and any(k in n for k in ("up_proj.weight","dense_h_to_4h.weight","c_fc.weight","fc_in.weight")):
            out.append(p)
    return out

def norm_align(A,B,colnorm=True,max_dim=1536):
    A=A.float(); B=B.float()
    if A.shape!=B.shape: return None
    if colnorm:
        A=A/(A.norm(dim=0,keepdim=True)+1e-12); B=B/(B.norm(dim=0,keepdim=True)+1e-12)
    # NEVER subsample ROWS before alignment: the rows are exactly what the permutation moved,
    # so striding picks a different subset from A than from B and the correspondence is destroyed.
    # (That bug is what produced the spurious Qwen number: inter=4864 > max_dim, SmolLM2's 1536 was not.)
    # Reduce the FEATURE axis instead -- it is shared and unpermuted.
    if A.shape[1]>max_dim:
        fidx=torch.arange(0,A.shape[1],max(1,A.shape[1]//max_dim))[:max_dim]
        A=A[:,fidx]; B=B[:,fidx]
    An=A/(A.norm(dim=1,keepdim=True)+1e-12); Bn=B/(B.norm(dim=1,keepdim=True)+1e-12)
    S=(An@Bn.T).numpy()
    r,c=linear_sum_assignment(-S)
    return float(S[r,c].mean())

def score(ma,mb,colnorm):
    a,b=ups(ma),ups(mb); L=min(len(a),len(b)); v=[]
    for i in range(L):
        s=norm_align(a[i],b[i],colnorm)
        if s is not None: v.append(s)
    return float(np.mean(v)) if v else float("nan")

def load(mid,launder=False,mode="full"):
    m=AutoModelForCausalLM.from_pretrained(mid).eval()
    if launder:
        LD.X1a_mlp_permute(m,1.0,11); LD.X1b_head_permute(m,11)
        if mode=="full": LD.X2_norm_scale(m,0.5,11)
    return m

PAIRS=[
 ("IDENTITY","HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M",False,"full"),
 ("P+ derived","HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M-Instruct",False,"full"),
 ("P+ derived","Qwen/Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B-Instruct",False,"full"),
 ("P~ SAME-RECIPE INDEP","EleutherAI/pythia-160m","EleutherAI/pythia-160m-deduped",False,"full"),
 ("LAUNDERED perm-only","HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M-Instruct",True,"perm"),
 ("LAUNDERED full","HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M-Instruct",True,"full"),
 ("LAUNDERED full","Qwen/Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B-Instruct",True,"full"),
]
rows=[]
print(f"{'class':<24}{'pair':<44}{'LAP+colnorm':>12}{'LAP only':>10}")
print("-"*92)
for cls,a,b,l,mode in PAIRS:
    ma=load(a); mb=load(b,l,mode)
    s1=score(ma,mb,True); s0=score(ma,mb,False)
    rows.append({"class":cls,"a":a,"b":b,"laundered":l,"mode":mode,"lap_colnorm":s1,"lap_only":s0})
    print(f"{cls:<24}{(a.split('/')[-1]+' | '+b.split('/')[-1]):<44}{s1:>12.4f}{s0:>10.4f}")
    del ma,mb
(RES/"lap2.json").write_text(json.dumps(rows,indent=1))
print("DONE-LAP2")
