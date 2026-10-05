"""Baseline B-1: quantize round-trip. The noise floor the transform arm has been missing."""
import os,sys,json,copy,shutil,subprocess,pathlib,torch,statistics as st
os.environ["HF_HUB_DISABLE_XET"]="1"
sys.path.insert(0,str(_ROOT) + "/M2/transforms")
from transformers import AutoModelForCausalLM, AutoTokenizer
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
CORPUS=["The Industrial Revolution began in Britain in the late eighteenth century and transformed manufacturing.",
 "In computer science a hash table implements an associative array, mapping keys to values.",
 "Photosynthesis converts light energy into chemical energy stored in glucose within chloroplasts.",
 "Quantum mechanics describes matter and energy at atomic scales, where particles behave as waves."]*10
P,C="HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M-Instruct"
tok=AutoTokenizer.from_pretrained(C); tok_save=AutoTokenizer.from_pretrained(C)
ref=AutoModelForCausalLM.from_pretrained(C).eval()
out=(RES/"q1_baseline.jsonl").open("w")
print(f"BASELINE B-1 — quantize round-trip on {C}")
print(f"{'variant':<26}{'exact':>7}{'ppl':>10}{'rho':>8}{'tier':>5}{'pipe':>8}{'sig_id':>8}{'EAS':>8}{'WVC':>8}{'pi':>7}")
print("-"*100)
for name,fn in [("baseline (no transform)",None),
                ("Q1 int8 per-channel",lambda m: LD.Q1_quant_roundtrip(m,8,True)),
                ("Q1 int4 per-channel", lambda m: LD.Q1_quant_roundtrip(m,4,True)),
                ("Q1 int8 per-tensor",  lambda m: LD.Q1_quant_roundtrip(m,8,False))]:
    m=copy.deepcopy(ref); meta=fn(m) if fn else None
    fid=LD.verify_exact(copy.deepcopy(ref).float(), copy.deepcopy(m).float(), tok, CORPUS)
    o=W/("q1_"+name.split()[0]+str(meta and meta.get("bits"))+str(meta and meta.get("per_channel")))
    if o.exists(): shutil.rmtree(o)
    m.save_pretrained(o); tok_save.save_pretrained(o); del m
    r=mpk(P,str(o)); s=r.get("scores",{}) or {}; g=r.get("signals",{}) or {}
    sp=s.get("pipeline_score"); pi=pct(sp) if sp is not None else 0
    rho=fid["ppl_ref"]/fid["ppl_new"] if fid.get("ppl_new") else 0
    f=lambda k: 0.0 if g.get(k) is None else g[k]
    out.write(json.dumps({"variant":name,"meta":meta,"fid":fid,"scores":s,"signals":g,"pi":pi,"rho":rho})+"\n"); out.flush()
    print(f"{name:<26}{str(fid['exact_within_tolerance']):>7}{fid['ppl_new']:>10.4f}{rho:>8.4f}"
          f"{str(s.get('mfi_tier')):>5}{sp or 0:>8.4f}{s.get('identity_score',0):>8.4f}"
          f"{f('eas'):>8.4f}{f('wvc'):>8.4f}{pi:>7.1f}",flush=True)
    shutil.rmtree(o,ignore_errors=True)
out.close(); print("DONE-Q1")
