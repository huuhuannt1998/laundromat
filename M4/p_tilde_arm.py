"""The P~ arm at scale — independent training runs of the SAME recipe.

This is the single biggest weakness in the project: M1's finding B6 (deployed lineage
verification confuses recipe convergence with weight derivation) and M4's soundness claim
for the LAP defence BOTH rest on one pair, pythia-1.4b vs -deduped. Pythia publishes the
same recipe trained independently on the Pile and the deduplicated Pile at many scales,
which gives a controlled ladder rather than an anecdote.

For each pair we record MPK's own signals AND the LAP+colnorm defence, so the two claims
are tested on the same data: does MPK confuse them, and does LAP separate them?
"""
import os,sys,json,subprocess,pathlib,torch,numpy as np
os.environ["HF_HUB_DISABLE_XET"]="1"
sys.path.insert(0,str(_ROOT) + "/M2/transforms")
sys.path.insert(0,str(_ROOT) + "/M4")
from transformers import AutoModelForCausalLM
from scipy.optimize import linear_sum_assignment

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)
REPO=pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
RES=pathlib.Path(str(_ROOT) + "/M4")

def mpk(a,b):
    p=subprocess.run(["uv","run","provenancekit","compare",a,b,"--json","--no-cache"],
        cwd=REPO,capture_output=True,text=True,timeout=7200,
        env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i=p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i>=0 else {"error":(p.stderr or "")[-200:]}

def ups(model):
    return [p for n,p in model.named_parameters()
            if p.ndim==2 and any(k in n for k in ("up_proj.weight","dense_h_to_4h.weight","c_fc.weight","fc_in.weight"))]

def lap_colnorm(A,B,max_feat=1536):
    A=A.float(); B=B.float()
    if A.shape!=B.shape: return None
    A=A/(A.norm(dim=0,keepdim=True)+1e-12); B=B/(B.norm(dim=0,keepdim=True)+1e-12)
    if A.shape[1]>max_feat:
        f=torch.arange(0,A.shape[1],max(1,A.shape[1]//max_feat))[:max_feat]; A=A[:,f]; B=B[:,f]
    An=A/(A.norm(dim=1,keepdim=True)+1e-12); Bn=B/(B.norm(dim=1,keepdim=True)+1e-12)
    S=(An@Bn.T).numpy(); r,c=linear_sum_assignment(-S)
    return float(S[r,c].mean())

def defence(a,b):
    ma=AutoModelForCausalLM.from_pretrained(a).eval(); mb=AutoModelForCausalLM.from_pretrained(b).eval()
    ua,ub=ups(ma),ups(mb); v=[]
    for i in range(min(len(ua),len(ub))):
        s=lap_colnorm(ua[i],ub[i])
        if s is not None: v.append(s)
    del ma,mb
    return float(np.mean(v)) if v else float("nan")

PAIRS=[("P~","EleutherAI/pythia-70m","EleutherAI/pythia-70m-deduped"),
       ("P~","EleutherAI/pythia-160m","EleutherAI/pythia-160m-deduped"),
       ("P~","EleutherAI/pythia-410m","EleutherAI/pythia-410m-deduped"),
       ("P~","EleutherAI/pythia-1b","EleutherAI/pythia-1b-deduped"),
       ("P~ cross-scale","EleutherAI/pythia-70m","EleutherAI/pythia-160m"),
       ("P= identity","EleutherAI/pythia-160m","EleutherAI/pythia-160m")]
out=(RES/"p_tilde_arm.jsonl").open("w")
print(f"{'class':<16}{'pair':<44}{'tier':>5}{'pipe':>8}{'sig_id':>8}{'EAS':>8}{'WVC':>8}{'LAPdef':>9}")
print("-"*106)
for cls,a,b in PAIRS:
    r=mpk(a,b)
    if "error" in r: print(f"{cls:<16}{a.split('/')[-1]+' | '+b.split('/')[-1]:<44} MPK ERROR {r['error'][:60]}"); continue
    s=r["scores"]; g=r["signals"]
    d=defence(a,b)
    out.write(json.dumps({"class":cls,"a":a,"b":b,"scores":s,"signals":g,"lap_defence":d})+"\n"); out.flush()
    print(f"{cls:<16}{a.split('/')[-1]+' | '+b.split('/')[-1]:<44}{s['mfi_tier']:>5}{s['pipeline_score']:>8.4f}"
          f"{s['identity_score']:>8.4f}{(g.get('eas') or 0):>8.4f}{(g.get('wvc') or 0):>8.4f}{d:>9.4f}",flush=True)
out.close(); print("DONE-PTILDE")
