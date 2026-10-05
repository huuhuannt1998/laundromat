"""E5 -- is the same-recipe failure a Pythia artifact?

Written inclusion rule, applied before collecting:
  A pair qualifies as SAME-RECIPE INDEPENDENT (N-SR) only if the publisher documents that
  the two checkpoints are separate pretraining runs of the SAME architecture at the SAME
  size under the SAME procedure, differing only in random seed and/or data order, with no
  shared checkpoint. Independence must be stated by the publisher; it is never inferred
  from naming.

Under that rule two suites qualify:
  - EleutherAI Pythia: the deduped and v0 suites are separate runs (already used).
  - Google MultiBERTs: 25 pretraining runs of BERT-base differing only in seed and data
    order, released precisely so that seed variance can be studied. Same architecture,
    same size, same recipe, independent initialisation.

MultiBERTs is the test: it shares no lineage, no vocabulary and no corpus with Pythia, so
if the failure reproduces there it is not a Pythia artifact.
"""
import json, subprocess, time, gc
import os as _os, pathlib as _pl
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
REPO = _ROOT/"M1"/"oracle"/"model-provenance-kit"
OUT  = _ROOT/"M7"/"e5_same_recipe.jsonl"

def mpk(args):
    p = subprocess.run(["uv","run","provenancekit",*args,"--json","--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**_os.environ,"HF_HUB_DISABLE_XET":"1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i>=0 else {"__error__":(p.stderr or "")[-300:]}

M = "google/multiberts-seed_"
PAIRS = [(f"{M}0", f"{M}1"), (f"{M}0", f"{M}2"), (f"{M}1", f"{M}2"),
         (f"{M}2", f"{M}3"), (f"{M}3", f"{M}4"), (f"{M}0", f"{M}4")]

done=set()
if OUT.exists():
    done={(json.loads(l)["a"],json.loads(l)["b"]) for l in OUT.read_text().splitlines() if l.strip()}
print(f"{'pair':<44}{'tier':>5}{'pipeline':>10}{'identity':>10}  verdict")
print("-"*88)
for a,b in PAIRS:
    if (a,b) in done: print(f"{a.split('/')[-1]+' / '+b.split('/')[-1]:<44}  (cached)"); continue
    t0=time.time(); r=mpk(["compare",a,b])
    sc=r.get("scores",{}); sg=r.get("signals",{})
    rec=dict(a=a,b=b,cls="N-SR",suite="multiberts",
             tier=sc.get("mfi_tier"), pipeline=sc.get("pipeline_score"),
             identity=sc.get("identity_score"), verdict=sc.get("provenance_decision"),
             signals={k:sg.get(k) for k in ("eas","wvc","end","lep","nlf")},
             error=r.get("__error__"), secs=round(time.time()-t0,1))
    with OUT.open("a") as fh: fh.write(json.dumps(rec)+"\n")
    if rec["error"]:
        print(f"{a.split('/')[-1]+' / '+b.split('/')[-1]:<44}  ERROR {rec['error'][:40]}")
    else:
        print(f"{a.split('/')[-1]+' / '+b.split('/')[-1]:<44}{str(rec['tier']):>5}"
              f"{rec['pipeline']:>10}{rec['identity']:>10}  {rec['verdict']}")
    gc.collect()

rows=[json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
ok=[r for r in rows if r.get("identity") is not None]
if ok:
    ids=[r["identity"] for r in ok]
    conf=[r for r in ok if (r["verdict"] or "").lower().startswith(("confirmed","high"))]
    print(f"\nMultiBERTs, {len(ok)} independent same-recipe pairs:")
    print(f"  identity score {min(ids):.4f}-{max(ids):.4f}  mean {sum(ids)/len(ids):.4f}")
    print(f"  returned a positive verdict: {len(conf)}/{len(ok)}")
    print(f"  Pythia comparison (Table fp): 0.6745-0.7976, 5/5 positive")
print("\nDONE-E5")
