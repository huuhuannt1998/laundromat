"""ADAPTIVE ATTACK on the LAP defence, using the QK invariance (X7).

LAP alignment undoes a PERMUTATION of rows. X7 applies an arbitrary invertible M per attention
head -- a general linear MIXING of coordinates, not a relabelling. No assignment problem can undo
that. So X7 should defeat LAP alignment ON THE MATRICES IT TOUCHES (attention W_q, W_k), while
leaving untouched any defence that reads the MLP.

This is the adaptive-attack evaluation I said I would not claim without running.
"""
import os,sys,json,copy,shutil,subprocess,pathlib,torch,numpy as np
os.environ["HF_HUB_DISABLE_XET"]="1"
sys.path.insert(0,str(_ROOT) + "/M2/transforms")
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.optimize import linear_sum_assignment
import laundry as LD

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)
REPO=pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
W=pathlib.Path(str(_ROOT) + "/M2/work"); W.mkdir(exist_ok=True)

def mpk(a,b):
    p=subprocess.run(["uv","run","provenancekit","compare",a,b,"--json","--no-cache"],cwd=REPO,
        capture_output=True,text=True,timeout=3600,env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i=p.stdout.find("{"); return json.loads(p.stdout[i:]) if i>=0 else {"error":p.stderr[-150:]}

def lap(A,B,mf=1536):
    A=A.float(); B=B.float()
    if A.shape!=B.shape: return None
    A=A/(A.norm(dim=0,keepdim=True)+1e-12); B=B/(B.norm(dim=0,keepdim=True)+1e-12)
    if A.shape[1]>mf:
        f=torch.arange(0,A.shape[1],max(1,A.shape[1]//mf))[:mf]; A=A[:,f]; B=B[:,f]
    An=A/(A.norm(dim=1,keepdim=True)+1e-12); Bn=B/(B.norm(dim=1,keepdim=True)+1e-12)
    S=(An@Bn.T).numpy(); r,c=linear_sum_assignment(-S); return float(S[r,c].mean())

def mlp_mats(m):  return [p for n,p in m.named_parameters() if p.ndim==2 and "mlp.c_fc" in n]
def attn_mats(m): return [p for n,p in m.named_parameters() if p.ndim==2 and "attn.c_attn" in n]

def defence_scores(ma,mb):
    out={}
    for tag,fn in (("LAP_on_MLP",mlp_mats),("LAP_on_ATTN",attn_mats)):
        A,B=fn(ma),fn(mb); v=[]
        for i in range(min(len(A),len(B))):
            s=lap(A[i],B[i])
            if s is not None: v.append(s)
        out[tag]=float(np.mean(v)) if v else float("nan")
    return out

BASE="gpt2"; tok=AutoTokenizer.from_pretrained(BASE)
ref=AutoModelForCausalLM.from_pretrained(BASE).eval()
PROBES=["The capital of France is Paris, and the capital of Germany is",
        "In 1969, the first humans landed on the surface of the"]*24
rows=[]
print(f"{'variant':<28}{'exact':>7}{'pipe':>8}{'sig_id':>8}{'EAS':>8}{'NLF':>8}{'LEP':>8}{'WVC':>8}"
      f"{'LAP-MLP':>9}{'LAP-ATTN':>10}")
print("-"*104)
for name,build in [("identity (control)", lambda m: None),
                   ("X7 qk-invariance s=0.3", lambda m: LD.X7_qk_invariance(m,0.3,7)),
                   ("X7 qk-invariance s=0.1", lambda m: LD.X7_qk_invariance(m,0.1,7))]:
    m=copy.deepcopy(ref); build(m)
    fid=LD.verify_exact(ref.float() if False else copy.deepcopy(ref).float(), copy.deepcopy(m).float(), tok, PROBES)
    o=W/("qk_"+name.split()[0])
    if o.exists(): shutil.rmtree(o)
    m.save_pretrained(o); AutoTokenizer.from_pretrained(BASE).save_pretrained(o)
    r=mpk(BASE,str(o)); s=r.get("scores",{}) or {}; g=r.get("signals",{}) or {}
    d=defence_scores(ref,m)
    f=lambda x: 0.0 if g.get(x) is None else g[x]
    print(f"{name:<28}{str(fid['exact_within_tolerance']):>7}{s.get('pipeline_score',0):>8.4f}"
          f"{s.get('identity_score',0):>8.4f}{f('eas'):>8.4f}{f('nlf'):>8.4f}{f('lep'):>8.4f}"
          f"{f('wvc'):>8.4f}{d['LAP_on_MLP']:>9.4f}{d['LAP_on_ATTN']:>10.4f}",flush=True)
    rows.append({"variant":name,"fid":fid,"scores":s,"signals":g,"defence":d})
    shutil.rmtree(o,ignore_errors=True); del m
pathlib.Path("adaptive_qk.json").write_text(json.dumps(rows,indent=1,default=str))
print("DONE-QK")
