"""M3 derived side — are declared base_model claims TRUE?

The claim survey measured how many models DECLARE a parent. It said nothing about whether the
declaration is correct. This samples declaring models small enough to fingerprint on CPU and runs
MPK compare(declared_parent, child). First direct claimed-vs-derived measurement in the project.

Selection is size-driven, so this is a SAMPLE OF SMALL MODELS, not of the hub. Stated as such.
"""
import os,sys,json,subprocess,pathlib,random
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

api=HfApi()
REPO=pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
RES=pathlib.Path(str(_ROOT) + "/M3")
MAXP=int(os.environ.get("MAXP","1000000000"))   # 1B params
WANT=int(sys.argv[1]) if len(sys.argv)>1 else 15

def nparams(mid):
    try:
        i=api.model_info(mid, files_metadata=False)
        st=getattr(i,"safetensors",None)
        if st and getattr(st,"total",None): return st.total
    except Exception: return None
    return None

# gather candidates from every claim sample we collected
cands=[]
for f in RES.glob("claim_*.jsonl"):
    for l in f.read_text().splitlines():
        d=json.loads(l)
        if d.get("declared") and d.get("n_base")==1: cands.append(d["id"])
cands=list(dict.fromkeys(cands)); random.Random(11).shuffle(cands)
print(f"candidate declaring models: {len(cands)}")

def mpk(a,b):
    p=subprocess.run(["uv","run","provenancekit","compare",a,b,"--json","--no-cache"],cwd=REPO,
        capture_output=True,text=True,timeout=3600,env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i=p.stdout.find("{"); return json.loads(p.stdout[i:]) if i>=0 else {"error":(p.stderr or "")[-140:]}

out=(RES/"verify_claims.jsonl").open("a"); done=0
print(f"{'child':<40}{'declared parent':<34}{'tier':>5}{'pipe':>8}{'sig_id':>8}  verdict")
print("-"*112)
for cid in cands:
    if done>=WANT: break
    try:
        info=api.model_info(cid); cd=info.card_data.to_dict() if info.card_data else {}
    except Exception: continue
    bm=cd.get("base_model")
    if isinstance(bm,str): bm=[bm]
    if not bm or len(bm)!=1: continue
    pid=bm[0]
    nc,np_=nparams(cid),nparams(pid)
    if not nc or not np_ or nc>MAXP or np_>MAXP: continue
    r=mpk(pid,cid)
    if "error" in r:
        print(f"{cid[:39]:<40}{pid[:33]:<34}  SKIP {r['error'][:40]}",flush=True); continue
    s=r["scores"]
    out.write(json.dumps({"child":cid,"declared_parent":pid,"n_child":nc,"n_parent":np_,
                          "scores":s,"signals":r.get("signals")})+"\n"); out.flush()
    print(f"{cid[:39]:<40}{pid[:33]:<34}{s['mfi_tier']:>5}{s['pipeline_score']:>8.4f}"
          f"{s['identity_score']:>8.4f}  {s['provenance_decision']}",flush=True)
    done+=1
out.close(); print(f"\nverified {done} declared parent-child pairs")
print("DONE-VERIFY")
