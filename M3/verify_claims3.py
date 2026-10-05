"""M3 derived side, v3 — run against the 78 models verified to be FINGERPRINTABLE
(top-level config.json + safetensors). v1/v2 failed because the declaring population is
dominated by LoRA adapters and GGUF conversions that MPK cannot open at all."""
import os,sys,json,subprocess,pathlib
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
MAXP=1.5e9; WANT=int(sys.argv[1]) if len(sys.argv)>1 else 12
def est(mid):
    try: c=json.load(open(hf_hub_download(mid,"config.json")))
    except Exception: return None
    h=c.get("hidden_size") or c.get("n_embd") or c.get("d_model")
    L=c.get("num_hidden_layers") or c.get("n_layer") or c.get("num_layers")
    V=c.get("vocab_size")
    return 12*L*h*h+V*h if (h and L and V) else None
def mpk(a,b):
    p=subprocess.run(["uv","run","provenancekit","compare",a,b,"--json","--no-cache"],cwd=REPO,
        capture_output=True,text=True,timeout=3600,env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i=p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i>=0 else {"error":(p.stderr or "")[-110:]}
cands=[x for x in (RES/"fingerprintable_declarers.txt").read_text().split() if x]
print(f"fingerprintable declaring models available: {len(cands)}")
out=(RES/"verify_claims.jsonl").open("a"); done=tried=err=0
print(f"{'child':<40}{'declared parent':<30}{'tier':>5}{'pipe':>8}{'sig_id':>8}  verdict")
print("-"*106)
for cid in cands:
    if done>=WANT: break
    tried+=1
    nc=est(cid)
    if not nc or nc>MAXP: continue
    try:
        info=api.model_info(cid); cd=info.card_data.to_dict() if info.card_data else {}
    except Exception: continue
    bm=cd.get("base_model")
    if isinstance(bm,str): bm=[bm]
    if not bm: continue
    pid=bm[0]
    np_=est(pid)
    if not np_ or np_>MAXP: continue
    r=mpk(pid,cid)
    if "error" in r:
        err+=1; print(f"{cid[:39]:<40}{pid[:29]:<30}  MPK-ERROR {r['error'][:34]}",flush=True); continue
    s=r["scores"]
    out.write(json.dumps({"child":cid,"declared_parent":pid,"n_child":nc,"n_parent":np_,
                          "scores":s,"signals":r.get("signals")})+"\n"); out.flush()
    print(f"{cid[:39]:<40}{pid[:29]:<30}{s['mfi_tier']:>5}{s['pipeline_score']:>8.4f}"
          f"{s['identity_score']:>8.4f}  {s['provenance_decision']}",flush=True)
    done+=1
out.close()
print(f"\nverified {done} declared pairs (tried {tried}, MPK errors {err})")
print("DONE-VERIFY3")
