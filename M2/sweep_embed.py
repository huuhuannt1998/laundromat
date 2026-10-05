"""Exit-2 probe: the embedding-perturbation exchange rate.

G1-B proved sigma_id >= w_EAS + w_END ~ 0.55 for any embedding-PRESERVING exact transform,
while the predeclared pi*=75 cut is 0.5298. So evasion requires touching the embedding, which
is inexact. This sweeps that trade-off at kappa = 0.
"""
import os, sys, json, shutil, subprocess, pathlib, copy, torch, statistics as st
os.environ["HF_HUB_DISABLE_XET"]="1"
sys.path.insert(0,str(_ROOT) + "/M2/transforms")
from transformers import AutoModelForCausalLM, AutoTokenizer
import laundry as LD

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

REPO=pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
RES =pathlib.Path(str(_ROOT) + "/M2/results")
W   =pathlib.Path(str(_ROOT) + "/M2/work"); W.mkdir(exist_ok=True)
NULL=json.loads((RES/"null_frozen.json").read_text())["scores"]
PI_CUT=st.quantiles(NULL,n=100)[74]
def mpk(a):
    p=subprocess.run(["uv","run","provenancekit",*a,"--json","--no-cache"],cwd=REPO,
        capture_output=True,text=True,timeout=5400,env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i=p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i>=0 else {"error":p.stderr[-200:]}
def pct(x): return 100.0*sum(1 for s in NULL if s<=x)/len(NULL)
PROBES=["The professor's arguments were criticized for their",
        "In 1969, the first humans landed on the surface of",
        "The capital of France is Paris, and the capital of Germany is",
        "Water boils at 100 degrees Celsius at standard atmospheric"]*24

P,C = sys.argv[1], sys.argv[2]
SIGMAS=[0.0,0.003,0.01,0.03,0.06,0.10,0.20,0.40]
tok=AutoTokenizer.from_pretrained(C)
out_f=(RES/f"sweep_embed_{C.replace('/','--')}.jsonl").open("w")
print(f"pi* cut={PI_CUT:.4f}  parent={P}")
print(f"{'rel_sigma':>10}{'EAS':>8}{'END':>8}{'WVC':>8}{'sig_id':>8}{'pipe':>8}{'rank':>6}{'pi':>7}"
      f"{'ppl_ref':>9}{'ppl_new':>9}{'rho_ppl':>9}{'E':>3}  kind")
for s in SIGMAS:
    ref=AutoModelForCausalLM.from_pretrained(C).eval(); m=copy.deepcopy(ref)
    applied=[]
    if s>0:
        applied.append(LD.X1a_mlp_permute(m,1.0,11)); applied.append(LD.X1b_head_permute(m,11))
        applied.append(LD.N1_embed_noise(m, rel_sigma=s, seed=11))
    fid=LD.verify_exact(ref.float(), copy.deepcopy(m).float(), tok, PROBES)
    o=W/f"emb{str(s).replace('.','p')}"
    if o.exists(): shutil.rmtree(o)
    m.save_pretrained(o); tok.save_pretrained(o)
    if s>0: applied.append(LD.M1t_metadata(o,mode="both"))
    del m, ref
    cmp_=mpk(["compare",P,str(o)]); scan=mpk(["scan",str(o),"--top-k","300","--threshold","0.0"])
    sc=cmp_.get("scores",{}) or {}; sg=cmp_.get("signals",{}) or {}
    ms=scan.get("matches",[]); ps=P.split("/")[-1]
    idx=[i for i,x in enumerate(ms,1) if x["model_id"]==ps]; R=idx[0] if idx else None
    sp=sc.get("pipeline_score"); pi=pct(sp) if sp is not None else None
    # rho on perplexity: capability retention proxy (lower ppl = better; rho = ppl_ref/ppl_new)
    rho = (fid["ppl_ref"]/fid["ppl_new"]) if fid.get("ppl_new") else None
    E = 1 if (R and R>5 and pi is not None and pi<=75) else 0
    kind = ("SUCCESS" if E and rho and rho>=0.98 else "DESTRUCTIVE" if E else "RESISTED")
    out_f.write(json.dumps({"rel_sigma":s,"applied":applied,"scores":sc,"signals":sg,
        "rank":R,"pi":pi,"fid":fid,"rho_ppl":rho,"E":E,"kind":kind,"kappa":0.0},default=str)+"\n"); out_f.flush()
    print(f"{s:>10.3f}{sg.get('eas',0):>8.4f}{sg.get('end',0):>8.4f}{sg.get('wvc',0) or 0:>8.4f}"
          f"{sc.get('identity_score',0):>8.4f}{sc.get('pipeline_score',0):>8.4f}{str(R):>6}{(pi or 0):>7.1f}"
          f"{fid['ppl_ref']:>9.4f}{fid['ppl_new']:>9.4f}{(rho or 0):>9.4f}{E:>3}  {kind}",flush=True)
    shutil.rmtree(o,ignore_errors=True)
out_f.close(); print("DONE")
