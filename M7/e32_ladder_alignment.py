"""E32 -- the alignment signal along the training ladder.

E12 (M6/e12_pythia_ladder.jsonl) scored nine checkpoints of pythia-160m against the final
checkpoint of the same run with the verifier; the identity score was flat and non-monotone
(0.736-0.858 over a threefold range of intervening training) and two checkpoints would have
read Not Matched on weights. The review asked what the alignment signal of section IX does on
the same ladder, where the lineage is certain by construction and the compute between the
two checkpoints is known exactly. Comparator: E19's column-normalised LAP on the MLP
up-projection rows (dense_h_to_4h), mean over layers, equal depth so the identity layer
correspondence. Output: M7/e32_ladder_alignment.json.
"""
import json, gc, os as _os, pathlib as _pl
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents if (p / "MANIFEST-dataintegrity.txt").exists())))
import numpy as np, torch
from transformers import AutoModelForCausalLM
from scipy.optimize import linear_sum_assignment
torch.set_grad_enabled(False)
OUT=_ROOT/"M7"/"e32_ladder_alignment.json"
MODEL="EleutherAI/pythia-160m"; FINAL=143000
STEPS=[1000,2000,4000,8000,16000,32000,64000,96000,128000]
T=299.892736e9; TOK=1024*2048

def ups(m): return [p.float() for n,p in m.named_parameters() if p.ndim==2 and "dense_h_to_4h.weight" in n]
def lap_colnorm(A,B,max_feat=1536):
    A=A/(A.norm(dim=0,keepdim=True)+1e-12); B=B/(B.norm(dim=0,keepdim=True)+1e-12)
    if A.shape[1]>max_feat:
        f=torch.arange(0,A.shape[1],max(1,A.shape[1]//max_feat))[:max_feat]; A,B=A[:,f],B[:,f]
    An=A/(A.norm(dim=1,keepdim=True)+1e-12); Bn=B/(B.norm(dim=1,keepdim=True)+1e-12)
    S=(An@Bn.T).numpy(); r,c=linear_sum_assignment(-S); return float(S[r,c].mean())

e12={r["step"]: r for r in (json.loads(l) for l in (_ROOT/"M6"/"e12_pythia_ladder.jsonl").read_text().splitlines() if l.strip())}
final=AutoModelForCausalLM.from_pretrained(MODEL, revision=f"step{FINAL}").eval(); uf=ups(final); del final; gc.collect()
rows=[]
print(f"{'step':>7}{'after fork':>12}{'sigma_id':>10}{'LAP':>8}  per-layer min/max")
for s in STEPS:
    m=AutoModelForCausalLM.from_pretrained(MODEL, revision=f"step{s}").eval(); u=ups(m); del m; gc.collect()
    per=[lap_colnorm(a,b) for a,b in zip(uf,u)]
    frac=(FINAL-s)*TOK/T
    rec=dict(step=s, final_step=FINAL, tokens_between=(FINAL-s)*TOK, after_fork_frac=frac,
             identity_score=e12.get(s,{}).get("identity_score"), lap=float(np.mean(per)),
             lap_per_layer=per)
    rows.append(rec)
    print(f"{s:>7}{frac*100:>11.1f}%{rec['identity_score'] or float('nan'):>10.4f}{rec['lap']:>8.4f}  {min(per):.4f}/{max(per):.4f}")
OUT.write_text(json.dumps(rows,indent=1)); print(f"wrote {OUT.relative_to(_ROOT)}")
