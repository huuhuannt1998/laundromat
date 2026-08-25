import os,json,subprocess,pathlib,sys
os.environ["HF_HUB_DISABLE_XET"]="1"

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

REPO=pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
RES=pathlib.Path(str(_ROOT) + "/M4")
def mpk(a,b):
    p=subprocess.run(["uv","run","provenancekit","compare",a,b,"--json","--no-cache"],
        cwd=REPO,capture_output=True,text=True,timeout=7200,
        env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i=p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i>=0 else {"error":(p.stderr or p.stdout)[-200:]}
PAIRS=json.loads(sys.argv[1])
f=(RES/"family_generality.jsonl").open("a")
for fam,a,b in PAIRS:
    try:
        r=mpk(a,b)
    except Exception as e:
        print(f"{fam:<12} EXC {type(e).__name__}: {str(e)[:100]}",flush=True); continue
    if "error" in r:
        print(f"{fam:<12}{a.split('/')[-1]+' | '+b.split('/')[-1]:<44}  ERROR {r['error'][:90]}",flush=True); continue
    s=r["scores"]; g=r["signals"]
    f.write(json.dumps({"family":fam,"a":a,"b":b,"scores":s,"signals":g})+"\n"); f.flush()
    print(f"{fam:<12}{a.split('/')[-1]+' | '+b.split('/')[-1]:<44}{s['mfi_tier']:>5}{s['pipeline_score']:>8.4f}"
          f"{s['identity_score']:>8.4f}{(g.get('eas') or 0):>8.4f}"
          f"{(0 if g.get('wvc') is None else g['wvc']):>8.4f}  {s['provenance_decision']}",flush=True)
f.close(); print("CHUNK-DONE")
