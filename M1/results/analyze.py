import json, pathlib, statistics as st

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

R=pathlib.Path(str(_ROOT) + "/M1/results")
rows=[]
for f in ["pairs.jsonl","pairs2.jsonl"]:
    p=R/f
    if not p.exists(): continue
    for line in p.read_text().splitlines():
        d=json.loads(line); s=d["result"].get("scores")
        if not s: continue
        rows.append(dict(a=d["model_a"],b=d["model_b"],gt=d["ground_truth"],why=d["rationale"],
                         pipe=s["pipeline_score"],ident=s["identity_score"],tier=s["mfi_tier"],
                         tok=s["tokenizer_score"],dec=s["provenance_decision"],
                         sig=d["result"].get("signals",{})))
def verdict(p):
    return "Confirmed" if p==1.0 else ("High-Confidence" if p>0.75 else ("Weak" if p>0.65 else "Not Matched"))

print("="*118)
print("MPK v1.1.0 BASELINE — all pairs (thresholds from source: HIGH=0.75, WEAK=0.65; SIMILARITY_THRESHOLD=0.75)")
print("="*118)
print(f"{'GT':<9} {'tier':<5} {'pipe':>6} {'ident':>6} {'tok':>6}  {'decision':<21} pair")
for r in sorted(rows,key=lambda r:(-r["pipe"])):
    print(f"{r['gt']:<9} {r['tier']:<5} {r['pipe']:>6.4f} {r['ident']:>6.4f} {r['tok']:>6.4f}  {r['dec']:<21} {r['a']} | {r['b']}")

t3p=[r for r in rows if r["tier"]==3 and r["gt"]=="related"]
t3n=[r for r in rows if r["tier"]==3 and r["gt"]=="unrelated"]
t1 =[r for r in rows if r["tier"]<=2]
print("\n"+"="*118)
print("TIER-3 (WEIGHT-DECIDED) SEPARATION — the only regime where weight evidence decides")
print("="*118)
print(f"positives n={len(t3p)}: {sorted(round(r['pipe'],4) for r in t3p)}")
print(f"negatives n={len(t3n)}: {sorted(round(r['pipe'],4) for r in t3n)}")
lo_p=min(r['pipe'] for r in t3p); hi_n=max(r['pipe'] for r in t3n)
print(f"\nmin positive = {lo_p:.4f}  ({[f'{r[chr(97)]}|{r[chr(98)]}' for r in t3p if r['pipe']==lo_p][0]})")
print(f"max negative = {hi_n:.4f}  ({[f'{r[chr(97)]}|{r[chr(98)]}' for r in t3n if r['pipe']==hi_n][0]})")
print(f"MARGIN (min_pos - max_neg) = {lo_p-hi_n:+.4f}   -> {'SEPARABLE' if lo_p>hi_n else '*** DISTRIBUTIONS OVERLAP: NO SEPARATING THRESHOLD EXISTS ***'}")
fn=[r for r in t3p if r['pipe']<=0.75]; fp=[r for r in t3n if r['pipe']>0.75]
print(f"\nAt documented 0.75 threshold:  FN={len(fn)}/{len(t3p)}  FP={len(fp)}/{len(t3n)}")
for r in fn: print(f"   FALSE NEGATIVE  {r['pipe']:.4f}  {r['a']} -> {r['b']}   ({r['why']})")
for r in fp: print(f"   FALSE POSITIVE  {r['pipe']:.4f}  {r['a']} vs {r['b']}   ({r['why']})")

print("\n"+"="*118)
print("MFI GATE (Tier<=2) — decided by config.json alone, weights never consulted for the verdict")
print("="*118)
for r in sorted(t1,key=lambda r:r["gt"]):
    flag="  <== FALSE 'Confirmed Match' ON A PAIR CISCO'S OWN CONSTITUTION CALLS INDEPENDENT" if r["gt"]=="unrelated" else ""
    print(f"  gt={r['gt']:<9} tier={r['tier']} pipe={r['pipe']:.4f} BUT identity={r['ident']:.4f} ({verdict(r['ident'])} on weights)  {r['a']} | {r['b']}{flag}")
