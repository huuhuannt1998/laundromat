"""Print every paper table's numbers from its frozen source, so a reviewer can answer
'where did Table X row Y come from?' without reading LaTeX."""
import json, pathlib, csv, collections

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

R = pathlib.Path(_ROOT)
def jl(p):
    return [json.loads(l) for l in (R/p).read_text().splitlines() if l.strip()]
def hdr(t, src):
    print(f"\n{'='*78}\n{t}\n  source: {src}\n{'-'*78}")

hdr("TABLE: same-recipe false positives (pythia base vs -deduped)", "M1/results/pairs.jsonl")
for r in jl("M1/results/pairs.jsonl"):
    a,b = r["model_a"], r["model_b"]
    if "deduped" in b and "pythia" in a:
        s,g = r["result"]["scores"], r["result"]["signals"]
        print(f"  {a.split('/')[-1]:<14} tier {s['mfi_tier']}  sig_id {s['identity_score']}  "
              f"eas {g['eas']}  wvc {g['wvc']}  {s['provenance_decision']}")

hdr("TABLE: weight-decided derivatives", "M1/results/tier3_positives.jsonl")
DIS = {"nreimers/BERT-Tiny_L-2_H-128_A-2"}
for r in sorted(jl("M1/results/tier3_positives.jsonl"), key=lambda r: r["scores"].get("identity_score") or 0):
    s = r["scores"]
    if s.get("mfi_tier") != 3: continue
    mark = "  [EXCLUDED: label defect]" if r["child"] in DIS else ""
    print(f"  {s['identity_score']:.4f}  wvc {r['signals'].get('wvc')}  "
          f"{s['provenance_decision']:<22} {r['child']}{mark}")

hdr("TABLE: models returning no verdict", "M1/results/mpk_crashes.jsonl")
for r in jl("M1/results/mpk_crashes.jsonl"):
    print(f"  arch_absent={str(r['architectures_absent']):<5} exit={r['mpk_exit']} "
          f"defect={str(r['defect']):<5} {r['child']}")

hdr("TABLE: training-progress ladder (E12)", "M6/e12_pythia_ladder.jsonl")
T = 299.892736
for r in sorted(jl("M6/e12_pythia_ladder.jsonl"), key=lambda r: r["step"]):
    print(f"  step{r['step']:<7} after-fork {r['tokens_between']/1e9/T*100:5.1f}%  "
          f"sig_id {r['identity_score']:.4f}  pipeline {r['pipeline_score']}  "
          f"d_emb {r['d_embedding']:.3f}")

hdr("AWM baseline (E3)", "M7/awm/e3_awm_results.jsonl")
p = R/"M7"/"awm"/"e3_awm_results.jsonl"
if p.exists():
    for r in jl("M7/awm/e3_awm_results.jsonl"):
        if r.get("status") != "OK":
            print(f"  {r['cls'] if 'cls' in r else '?':<8} {r['label']:<30} {r.get('status')}"); continue
        print(f"  {r['cls']:<8} {r['label']:<30} wq_wk {r['awm_wqwk']:.4f}  z {r['awm_z']}")

hdr("Second substrate (E4), all runs", "M1/oracle/model_provenance_testing/runs/*.csv")
runs = sorted((R/"M1/oracle/model_provenance_testing/runs").glob("*.csv"))
for f in runs:
    rows = list(csv.DictReader(open(f)))
    if not rows: continue
    print(f"  -- {f.name}")
    for r in rows:
        r = {k.strip(): (v.strip() if isinstance(v,str) else v) for k,v in r.items()}
        print(f"     {r['outcome']:<3} p={float(r['p-value']):.5f}  n={r['tot prompts']}  "
              f"{r['model name'].split('/')[-1]}")

hdr("Alignment defence", "M4/defence_expanded.json  (+ M7/e18_defence.json if present)")
for src in ("M4/defence_expanded.json", "M7/e18_defence.json"):
    q = R/src
    if not q.exists(): continue
    rows = json.loads(q.read_text())
    byc = collections.defaultdict(list)
    for r in rows:
        if r.get("lap") is not None: byc[r["cls"]].append(r["lap"])
    print(f"  -- {src}  (n={len(rows)})")
    for c in sorted(byc):
        v = byc[c]
        print(f"     {c:<3} n={len(v):<3} {min(v):.4f}-{max(v):.4f}")
    pos = byc.get("P+", []); neg = [x for c,v in byc.items() if c!="P+" for x in v]
    if pos and neg:
        print(f"     margin (min P+ - max negative) = {min(pos)-max(neg):+.4f}")
print()
