"""M2 Stage-0 T1/T2 — freeze the benign-independent null N, profile DB density, compute the C4 rate."""
import json, pathlib, collections, statistics as st

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

R=pathlib.Path(str(_ROOT) + "/M2/results")
rows=[json.loads(l) for l in (R/"wide_scans.jsonl").read_text().splitlines()]

# ---- known-related exclusions: a scan hit is DERIVED if it is the true parent,
#      the target itself, or a documented same-lineage sibling. Everything else in the
#      ranked list is treated as NOT-DERIVED for the null. Conservative: when unsure, exclude.
def short(x): return x.split("/")[-1]
RELATED = {
 "HuggingFaceTB/SmolLM2-135M":          {"SmolLM2-135M"},
 "HuggingFaceTB/SmolLM2-135M-Instruct": {"SmolLM2-135M"},
 "Qwen/Qwen2.5-0.5B":                   {"Qwen2.5-0.5B"},
 "Qwen/Qwen2.5-0.5B-Instruct":          {"Qwen2.5-0.5B"},
 "gpt2":        {"gpt2","gpt2-medium","gpt2-large","gpt2-xl"},   # same family, conservative exclude
 "distilgpt2":  {"gpt2","gpt2-medium","gpt2-large","gpt2-xl"},
 "bigscience/bloom-560m": {"bloomz-560m"},
 "google-bert/bert-base-uncased": {"bert-base-uncased"},
 "EleutherAI/pythia-160m": set(), "EleutherAI/pythia-160m-deduped": set(),
}
NULL=[]; prof=[]
for r in rows:
    t=r["target"]; res=r["result"]
    if "error" in res: print("ERR",t); continue
    ms=res["matches"]; excl=RELATED.get(t,set())
    sc=[m["scores"]["pipeline_score"] for m in ms]
    for m in ms:
        if m["model_id"] in excl: continue
        NULL.append({"target":t,"db":m["model_id"],"family":m["family_id"],
                     "bucket":m["param_bucket"],"score":m["scores"]["pipeline_score"]})
    pr=None
    if r["true_parent"]:
        idx=[i for i,m in enumerate(ms,1) if m["model_id"]==r["true_parent"]]
        pr=idx[0] if idx else None
    prof.append({"target":t,"parent":r["true_parent"],"parent_rank":pr,"n_cand":len(ms),
        "n_gt_075":sum(s>0.75 for s in sc),"n_gt_065":sum(s>0.65 for s in sc),
        "s6":sorted(sc,reverse=True)[5] if len(sc)>5 else None,
        "max":max(sc),"median":st.median(sc),"min":min(sc)})

scores=[x["score"] for x in NULL]
strata=collections.Counter((x["family"],x["bucket"]) for x in NULL)
(R/"null_frozen.json").write_text(json.dumps(
  {"n":len(NULL),"scores":scores,"strata":{f"{a}|{b}":c for (a,b),c in strata.items()},
   "quantiles":{q:round(st.quantiles(scores,n=100)[q-1],4) for q in (10,25,50,75,90,95,99)}},indent=1))

print("="*104); print("FROZEN NULL N"); print("="*104)
print(f"|N| = {len(scores)}   strata = {len(strata)}   strata with >=10 members = {sum(1 for c in strata.values() if c>=10)}")
qs=st.quantiles(scores,n=100)
print("quantiles: " + "  ".join(f"p{q}={qs[q-1]:.4f}" for q in (10,25,50,75,90,95,99)))
print(f"PASS |N|>=200: {len(scores)>=200}    PASS >=10 per stratum: {all(c>=10 for c in strata.values())}"
      f"  (min stratum={min(strata.values())})")

print("\n"+"="*104); print("DB DENSITY PROFILE  --  the C4 / rank-validity check"); print("="*104)
print(f"{'target':<40}{'cand':>5}{'rank':>6}{'>0.75':>7}{'>0.65':>7}{'6th':>8}{'max':>8}{'median':>8}")
for p in prof:
    print(f"{p['target']:<40}{p['n_cand']:>5}{str(p['parent_rank']):>6}{p['n_gt_075']:>7}"
          f"{p['n_gt_065']:>7}{(f'{p[chr(115)+chr(54)]:.4f}' if p['s6'] else '-'):>8}"
          f"{p['max']:>8.4f}{p['median']:>8.4f}")
print("\nINTERPRETATION: 'rank>5' requires the parent to fall BELOW the 6th-highest score in its own")
print("scan. That target is parent-specific and is printed above as '6th'. It is the real evasion bar.")
pi75=qs[74]
print(f"\nNull p75 = {pi75:.4f}  -> the predeclared pi* = 75 requires laundered sigma_pipe <= {pi75:.4f}")
# C4: benign untransformed targets where rank>5 while percentile>75
c4=[p for p in prof if p["parent_rank"] and p["parent_rank"]>5]
print(f"\nC4 CHECK: benign untransformed targets with parent_rank > 5: {len(c4)}/{sum(1 for p in prof if p['parent_rank'])}"
      f"  -> {'PASS (<20%)' if len(c4)/max(1,sum(1 for p in prof if p['parent_rank']))<0.2 else 'FAIL'}")
