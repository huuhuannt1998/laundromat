"""M2 Alg-1 EVAL: apply a kappa=0 transform, score it against MPK, delete the artifact.

Discipline enforced here:
  * artifact saved in the PARENT'S ORIGINAL DTYPE (a dtype change would itself perturb weights)
  * MPK --no-cache on every call (a stale cache keyed on model id returns the parent's features)
  * artifact generated -> evaluated -> DELETED inside one call (<=60 GB disk)
  * identity control must reproduce the published numbers exactly
"""
import os, sys, json, shutil, subprocess, pathlib, tempfile, copy, torch, statistics as st
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
WORK=pathlib.Path(str(_ROOT) + "/M2/work"); WORK.mkdir(exist_ok=True)
NULL=json.loads((RES/"null_frozen.json").read_text())["scores"]
K_STAR, PI_STAR = 5, 75
PI_CUT = st.quantiles(NULL,n=100)[PI_STAR-1]

def mpk(args):
    p=subprocess.run(["uv","run","provenancekit",*args,"--json","--no-cache"],
        cwd=REPO,capture_output=True,text=True,timeout=5400,
        env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    if p.returncode!=0: return {"error":p.stderr[-300:]}
    i=p.stdout.find("{");  return json.loads(p.stdout[i:]) if i>=0 else {"error":"nojson"}

def pctile(x):
    return 100.0*sum(1 for s in NULL if s<=x)/len(NULL)

PROBES=["The professor's arguments were criticized for their",
        "In 1969, the first humans landed on the surface of",
        "The capital of France is Paris, and the capital of Germany is",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot ="]*24

def EVAL(parent, child, name, ops, meta_mode=None):
    out = WORK/f"{name.replace('/','_').replace(' ','')}"
    if out.exists(): shutil.rmtree(out)
    tok = AutoTokenizer.from_pretrained(child)                    # used for probing (mutated by truncation)
    tok_save = AutoTokenizer.from_pretrained(child)               # PRISTINE copy for the artifact.
    # Calling a fast tokenizer with truncation=True mutates its backend state, and
    # save_pretrained serialises that as a "truncation" field in tokenizer.json. MPK hashes the
    # tokenizer backend into family_hash, so a probed tokenizer silently demotes MFI tier 2 -> 3.
    ref = AutoModelForCausalLM.from_pretrained(child).eval()      # original dtype
    m   = copy.deepcopy(ref)
    applied=[op(m) for op in ops]
    fid = LD.verify_exact(ref.float(), copy.deepcopy(m).float(), tok, PROBES) if ops else \
          {"max_abs_logit_delta":0.0,"rel_ppl_delta":0.0,"exact_within_tolerance":True,
           "ppl_ref":None,"ppl_new":None,"n_tokens":0}
    m.save_pretrained(out); tok_save.save_pretrained(out)
    if meta_mode:
        applied.append(LD.M1t_metadata(out, mode=("both" if meta_mode=="gate" else meta_mode)))
        if meta_mode == "gate":      # architecture edit PLUS tokenizer-config edit
            applied.append(LD.M3t_tokenizer_config(out))
    del m, ref
    cmp_ = mpk(["compare", parent, str(out)])
    scan = mpk(["scan", str(out), "--top-k","300","--threshold","0.0"])
    sc = cmp_.get("scores",{})
    ms = scan.get("matches",[])
    pshort = parent.split("/")[-1]
    idx=[i for i,x in enumerate(ms,1) if x["model_id"]==pshort]
    R = idx[0] if idx else None
    sixth = sorted([x["scores"]["pipeline_score"] for x in ms],reverse=True)[5] if len(ms)>5 else None
    sp = sc.get("pipeline_score")
    pi = pctile(sp) if sp is not None else None
    E  = 1 if (R is not None and R>K_STAR and pi is not None and pi<=PI_STAR) else 0
    exact = fid["exact_within_tolerance"]
    kind = ("SUCCESS" if E and exact else "DESTRUCTIVE" if E and not exact else "RESISTED")
    shutil.rmtree(out, ignore_errors=True)
    return {"name":name,"parent":parent,"child":child,"applied":applied,"meta_mode":meta_mode,
            "fidelity":fid,"cmp":cmp_.get("scores"),"signals":cmp_.get("signals"),
            "parent_rank":R,"sixth_score":sixth,"pi":pi,"E":E,"kind":kind,"kappa":0.0,
            "capability_record":{"record_version":"2.0","observability_level":"O1","kappa":0.0,
              "training_compute":False,"backward_pass":False,"exactness_verified":exact,
              "hardware":"Apple M4 10c/24GB CPU-only"}}

if __name__=="__main__":
    PARENT, CHILD = sys.argv[1], sys.argv[2]
    CASES=[
     ("S0-identity",            [],                                                   None),
     ("S1-X1a",                 [lambda m: LD.X1a_mlp_permute(m,1.0,11)],             None),
     ("S1-X1b",                 [lambda m: LD.X1b_head_permute(m,11)],                None),
     ("S1-X2",                  [lambda m: LD.X2_norm_scale(m,0.5,11)],               None),
     ("S1-X1a.X1b.X2",          [lambda m: LD.X1a_mlp_permute(m,1.0,11),
                                 lambda m: LD.X1b_head_permute(m,11),
                                 lambda m: LD.X2_norm_scale(m,0.5,11)],               None),
     ("M1t-only",               [],                                                   "both"),
     ("S2-X1a.M1t",             [lambda m: LD.X1a_mlp_permute(m,1.0,11)],             "both"),
     ("S2-X2.M1t",              [lambda m: LD.X2_norm_scale(m,0.5,11)],               "both"),
     ("S2-FLAGSHIP-X1a.X1b.X2.M1t",[lambda m: LD.X1a_mlp_permute(m,1.0,11),
                                 lambda m: LD.X1b_head_permute(m,11),
                                 lambda m: LD.X2_norm_scale(m,0.5,11)],               "both"),
    ]
    tag=CHILD.replace("/","--")
    fh=(RES/f"gate_{tag}.jsonl").open("w")
    print(f"pi* cut (null p75) = {PI_CUT:.4f}   k* = {K_STAR}\n")
    print(f"{'case':<30}{'tier':>5}{'sigma':>8}{'sig_id':>8}{'rank':>6}{'6th':>8}{'pi':>7}{'E':>3}  kind")
    for name,ops,mm in CASES:
        r=EVAL(PARENT,CHILD,name,ops,mm)
        fh.write(json.dumps(r,default=str)+"\n"); fh.flush()
        c=r["cmp"] or {}
        print(f"{name:<30}{str(c.get('mfi_tier')):>5}{c.get('pipeline_score',0):>8.4f}"
              f"{c.get('identity_score',0):>8.4f}{str(r['parent_rank']):>6}"
              f"{(r['sixth_score'] or 0):>8.4f}{(r['pi'] or 0):>7.1f}{r['E']:>3}  {r['kind']}",flush=True)
    fh.close(); print("DONE")
