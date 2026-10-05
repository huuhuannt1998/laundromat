"""M3 derived side, v2. The v1 size filter rejected nearly every candidate because safetensors
metadata is often absent. This uses config.json (always present) to estimate parameter count,
which is both more reliable and cheaper."""
import os,sys,json,subprocess,pathlib,random
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi, hf_hub_download

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

api=HfApi()
REPO=pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
RES=pathlib.Path(str(_ROOT) + "/M3")
MAXP=1.2e9; WANT=int(sys.argv[1]) if len(sys.argv)>1 else 15

def est_params(mid):
    """Rough parameter count from config.json: 12*L*h^2 + V*h. Always available, no weights."""
    try:
        p=hf_hub_download(mid,"config.json"); c=json.load(open(p))
    except Exception: return None
    h=c.get("hidden_size") or c.get("n_embd") or c.get("d_model")
    L=c.get("num_hidden_layers") or c.get("n_layer") or c.get("num_layers")
    V=c.get("vocab_size")
    if not (h and L and V): return None
    return 12*L*h*h + V*h

def mpk(a,b):
    p=subprocess.run(["uv","run","provenancekit","compare",a,b,"--json","--no-cache"],cwd=REPO,
        capture_output=True,text=True,timeout=3600,env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i=p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i>=0 else {"error":(p.stderr or "")[-120:]}

cands=[]
for f in RES.glob("claim_*.jsonl"):
    for l in f.read_text().splitlines():
        d=json.loads(l)
        if d.get("declared") and d.get("n_base")==1: cands.append(d["id"])
cands=list(dict.fromkeys(cands)); random.Random(23).shuffle(cands)
print(f"declaring candidates available: {len(cands)}")
seen=set()
for l in (RES/"verify_claims.jsonl").read_text().splitlines():
    try: seen.add(json.loads(l)["child"])
    except Exception: pass
out=(RES/"verify_claims.jsonl").open("a"); done=0; tried=0
print(f"{'child':<44}{'declared parent':<32}{'tier':>5}{'pipe':>8}{'sig_id':>8}  verdict")
print("-"*110)
for cid in cands:
    if done>=WANT: break
    if cid in seen: continue
    tried+=1
    if tried>400: break
    nc=est_params(cid)
    if not nc or nc>MAXP: continue
    try:
        info=api.model_info(cid); cd=info.card_data.to_dict() if info.card_data else {}
    except Exception: continue
    bm=cd.get("base_model")
    if isinstance(bm,str): bm=[bm]
    if not bm or len(bm)!=1: continue
    pid=bm[0]
    np_=est_params(pid)
    if not np_ or np_>MAXP: continue
    r=mpk(pid,cid)
    if "error" in r: continue
    s=r["scores"]
    out.write(json.dumps({"child":cid,"declared_parent":pid,"n_child":nc,"n_parent":np_,
                          "scores":s,"signals":r.get("signals")})+"\n"); out.flush()
    print(f"{cid[:43]:<44}{pid[:31]:<32}{s['mfi_tier']:>5}{s['pipeline_score']:>8.4f}"
          f"{s['identity_score']:>8.4f}  {s['provenance_decision']}",flush=True)
    done+=1
out.close(); print(f"\nverified {done} declared parent-child pairs (tried {tried} candidates)")
print("DONE-VERIFY2")
