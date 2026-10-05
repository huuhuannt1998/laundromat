"""X3 (OV invariance) vs MPK and vs the LAP defence, on a RoPE model.

X7 (QK) needed a non-RoPE model. X3 (OV) does not -- RoPE touches only Q and K. So this tests
whether the general-linear-mixing attack on alignment-based defences extends to Llama/Qwen-class
models, which is the population that matters."""
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
RES=pathlib.Path(str(_ROOT) + "/M2/results")
W=pathlib.Path(str(_ROOT) + "/M2/work"); W.mkdir(exist_ok=True)
NULL=json.loads((RES/"null_frozen.json").read_text())["scores"]
def pct(x): return 100.0*sum(1 for s in NULL if s<=x)/len(NULL)
def mpk(a,b):
    p=subprocess.run(["uv","run","provenancekit","compare",a,b,"--json","--no-cache"],cwd=REPO,
        capture_output=True,text=True,timeout=3600,env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i=p.stdout.find("{"); return json.loads(p.stdout[i:]) if i>=0 else {"error":p.stderr[-120:]}
def lap(A,B,mf=1536):
    A=A.float(); B=B.float()
    if A.shape!=B.shape: return None
    A=A/(A.norm(dim=0,keepdim=True)+1e-12); B=B/(B.norm(dim=0,keepdim=True)+1e-12)
    if A.shape[1]>mf:
        f=torch.arange(0,A.shape[1],max(1,A.shape[1]//mf))[:mf]; A=A[:,f]; B=B[:,f]
    An=A/(A.norm(dim=1,keepdim=True)+1e-12); Bn=B/(B.norm(dim=1,keepdim=True)+1e-12)
    S=(An@Bn.T).numpy(); r,c=linear_sum_assignment(-S); return float(S[r,c].mean())
def sel(m,keys): return [p for n,p in m.named_parameters() if p.ndim==2 and any(k in n for k in keys)]
def defence(ma,mb):
    out={}
    for tag,keys in (("LAP_MLP",("up_proj.weight",)),("LAP_VO",("v_proj.weight","o_proj.weight"))):
        A,B=sel(ma,keys),sel(mb,keys); v=[]
        for i in range(min(len(A),len(B))):
            s=lap(A[i],B[i])
            if s is not None: v.append(s)
        out[tag]=float(np.mean(v)) if v else float("nan")
    return out
P,C="HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M-Instruct"
tok=AutoTokenizer.from_pretrained(C); tok_save=AutoTokenizer.from_pretrained(C)
ref=AutoModelForCausalLM.from_pretrained(C).eval()
PROBES=["The capital of France is Paris, and the capital of Germany is",
        "In 1969, the first humans landed on the surface of the"]*16
rows=[]
print(f"{'variant':<26}{'exact':>7}{'tier':>5}{'pipe':>8}{'sig_id':>8}{'WVC':>8}{'pi':>7}{'LAP-MLP':>9}{'LAP-VO':>9}")
print("-"*90)
for name,fn in [("identity (control)",None),
                ("X3ov sigma=0.1",lambda m: LD.X3_ov_invariance(m,0.1,3)),
                ("X3ov sigma=0.3",lambda m: LD.X3_ov_invariance(m,0.3,3)),
                ("X3ov + X1a + X2",lambda m: [LD.X3_ov_invariance(m,0.3,3),LD.X1a_mlp_permute(m,1.0,3),LD.X2_norm_scale(m,0.5,3)])]:
    m=copy.deepcopy(ref)
    if fn: fn(m)
    fid=LD.verify_exact(copy.deepcopy(ref).float(), copy.deepcopy(m).float(), tok, PROBES)
    d=defence(ref,m)
    o=W/("ov_"+name.split()[0].replace("=","")); 
    if o.exists(): shutil.rmtree(o)
    m.save_pretrained(o); tok_save.save_pretrained(o); del m
    r=mpk(P,str(o)); s=r.get("scores",{}) or {}; g=r.get("signals",{}) or {}
    sp=s.get("pipeline_score"); pi=pct(sp) if sp is not None else 0
    rows.append({"variant":name,"fid":fid,"scores":s,"signals":g,"defence":d,"pi":pi})
    print(f"{name:<26}{str(fid['exact_within_tolerance']):>7}{str(s.get('mfi_tier')):>5}{sp or 0:>8.4f}"
          f"{s.get('identity_score',0):>8.4f}{(g.get('wvc') or 0):>8.4f}{pi:>7.1f}"
          f"{d['LAP_MLP']:>9.4f}{d['LAP_VO']:>9.4f}",flush=True)
    shutil.rmtree(o,ignore_errors=True)
pathlib.Path("adaptive_ov.json").write_text(json.dumps(rows,indent=1,default=str))
print("DONE-OV")
