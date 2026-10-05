"""M3 derived side, v4. Three earlier attempts failed for three different reasons:
  v1  safetensors metadata usually absent          -> 0 pairs
  v2  declaring population is LoRA/GGUF, no config -> 0 pairs (that became a finding: only 52%
                                                      of declarations sit on a fingerprintable artifact)
  v3  hf_hub_download broken by a filelock version clash, and the popular declaring head is 7B+
Now: filelock fixed, no broken API size filter, iterate the live listing and size-check both
sides from config.json. Gated repos are SKIPPED, never requested."""
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
        capture_output=True,text=True,timeout=2400,env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i=p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i>=0 else {"error":(p.stderr or "")[-110:]}
seen=set()
vp=RES/"verify_claims.jsonl"
if vp.exists():
    for l in vp.read_text().splitlines():
        try: seen.add(json.loads(l)["child"])
        except Exception: pass
out=vp.open("a"); done=scanned=skipped_big=err=0
print(f"{'child':<44}{'declared parent':<30}{'tier':>5}{'pipe':>8}{'sig_id':>8}  verdict")
print("-"*110)
for m in api.list_models(pipeline_tag="text-generation", sort="downloads", cardData=True, limit=1500):
    if done>=WANT: break
    scanned+=1
    cd=m.card_data.to_dict() if getattr(m,"card_data",None) else {}
    bm=cd.get("base_model")
    if isinstance(bm,str): bm=[bm]
    if not bm or m.id in seen: continue
    nc=est(m.id)
    if not nc or nc>MAXP: skipped_big+=1; continue
    pid=bm[0]; np_=est(pid)
    if not np_ or np_>MAXP: skipped_big+=1; continue
    r=mpk(pid,m.id)
    if "error" in r: err+=1; print(f"{m.id[:43]:<44}{pid[:29]:<30}  MPK-ERR {r['error'][:32]}",flush=True); continue
    s=r["scores"]
    out.write(json.dumps({"child":m.id,"declared_parent":pid,"n_child":nc,"n_parent":np_,
                          "scores":s,"signals":r.get("signals")})+"\n"); out.flush()
    print(f"{m.id[:43]:<44}{pid[:29]:<30}{s['mfi_tier']:>5}{s['pipeline_score']:>8.4f}"
          f"{s['identity_score']:>8.4f}  {s['provenance_decision']}",flush=True)
    done+=1
out.close()
print(f"\nverified {done} declared pairs | scanned {scanned} | too big {skipped_big} | MPK errors {err}")
print("DONE-VERIFY4")
