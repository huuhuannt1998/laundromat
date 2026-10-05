"""Final M5 analysis across both parent families, keyed on VERIFIED lineage.

The first draft of sweep_bert.py assumed lineages from repo names.  Three of seven
were wrong (two from-scratch models labelled as continued-pretraining, one whose
true parent is bert-base-CASED via BioBERT).  The displacement measure flagged the
legal-bert case before the model card was read, which is itself evidence the measure
works.  Everything below keys on M5/lineage.json, where each claim carries its source.
"""
import json, pathlib
import numpy as np

R = pathlib.Path(__file__).resolve().parent
LIN = json.load(open(R / "lineage.json"))
BAR, WEAK = 0.5298, 0.65


def load(fn):
    p = R / fn
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip().startswith("{")] if p.exists() else []


rows = []
for fn in ("corpus_distance.jsonl", "sweep_bert.jsonl"):
    for r in load(fn):
        par, ch = r["parent"], r["child"]
        info = LIN.get(par, {}).get(ch, {})
        rows.append(dict(parent=par, child=ch, rel=info.get("rel", "UNVERIFIED"),
                         parent_ok=info.get("parent_ok"), depth=info.get("depth", 1),
                         disp=r["displacement"]["overall"], demb=r["displacement"].get("embedding"),
                         sid=r["scores"].get("identity_score"), tier=r["scores"].get("mfi_tier"),
                         dec=r["scores"].get("provenance_decision"),
                         eas=r["signals"].get("eas"), end=r["signals"].get("end"),
                         wvc=r["signals"].get("wvc"), repaired=r.get("config_repaired", False)))

print("=" * 118)
print("M5 — DISPLACEMENT vs THE DETECTOR, BOTH FAMILIES, VERIFIED LINEAGE")
print("=" * 118)
for par in sorted({r["parent"] for r in rows}):
    print(f"\nPARENT: {par}")
    print(f"  {'relationship':<28}{'disp':>7}{'dispEmb':>9}{'sig_id':>8}{'EAS':>8}{'END':>8}"
          f"{'tier':>5}  verdict            child")
    print("  " + "-" * 114)
    for r in sorted([x for x in rows if x["parent"] == par], key=lambda x: (x["rel"], x["demb"] or 0)):
        nf = lambda v: float("nan") if v is None else v
        # is the verdict CORRECT given the verified lineage?
        derived = r["parent_ok"] and r["rel"] != "FROM_SCRATCH"
        matched = r["dec"] != "Not Matched"
        mark = "" if derived == matched else ("  <-- FALSE NEG" if derived else "  <-- FALSE POS")
        print(f"  {r['rel']:<28}{r['disp']:>7.4f}{nf(r['demb']):>9.4f}{nf(r['sid']):>8.4f}"
              f"{nf(r['eas']):>8.4f}{nf(r['end']):>8.4f}{str(r['tier']):>5}  {r['dec']:<18}"
              f" {r['child'].split('/')[-1][:30]}{mark}")

# ---- correctness tally on VERIFIED lineage only
ver = [r for r in rows if r["rel"] != "UNVERIFIED" and r["parent_ok"] is not None]
tp = sum(1 for r in ver if r["parent_ok"] and r["rel"] != "FROM_SCRATCH" and r["dec"] != "Not Matched")
fn_ = sum(1 for r in ver if r["parent_ok"] and r["rel"] != "FROM_SCRATCH" and r["dec"] == "Not Matched")
tn = sum(1 for r in ver if (not r["parent_ok"] or r["rel"] == "FROM_SCRATCH") and r["dec"] == "Not Matched")
fp = sum(1 for r in ver if (not r["parent_ok"] or r["rel"] == "FROM_SCRATCH") and r["dec"] != "Not Matched")
print(f"\nCONFUSION ON VERIFIED LINEAGE (n={len(ver)}):  TP {tp}  FN {fn_}  TN {tn}  FP {fp}")
if tp + fn_:
    print(f"  recall on true derivatives      = {tp}/{tp+fn_} = {100*tp/(tp+fn_):.0f}%")
if tn + fp:
    print(f"  specificity on non-derivatives  = {tn}/{tn+fp} = {100*tn/(tn+fp):.0f}%")

# ---- regression on TRUE DERIVATIVES ONLY (from-scratch models are not on this curve)
d = [r for r in ver if r["parent_ok"] and r["rel"] != "FROM_SCRATCH"
     and r["demb"] is not None and r["sid"] is not None]
if len(d) >= 3:
    X = np.array([r["demb"] for r in d]); Y = np.array([r["sid"] for r in d])
    Xo = np.array([r["disp"] for r in d])
    def fit(x, y):
        m, c = np.polyfit(x, y, 1)
        return m, c, 1 - ((y-(m*x+c))**2).sum()/((y-y.mean())**2).sum()
    me, ce, r2e = fit(X, Y); mo, co, r2o = fit(Xo, Y)
    print(f"\nREGRESSION ON TRUE DERIVATIVES ONLY (n={len(d)}, both families pooled):")
    print(f"  overall    sigma_id = {co:.4f} {mo:+.4f}*d   R^2 = {r2o:.4f}")
    print(f"  embedding  sigma_id = {ce:.4f} {me:+.4f}*d   R^2 = {r2e:.4f}")
    print(f"  -> {'EMBEDDING' if r2e > r2o else 'OVERALL'} displacement is the better predictor")
    scratch = [r["demb"] for r in ver if r["rel"] == "FROM_SCRATCH" and r["demb"]]
    if scratch:
        print(f"\n  FROM-SCRATCH controls in-family: dispEmb {[round(s,4) for s in scratch]}")
        print(f"  (independent-run anchor from Pythia: 1.2044)")
    for lbl, thr in (("MPK 'Not Matched' (0.65)", WEAK), ("predeclared bar (0.5298)", BAR)):
        dc = (thr - ce)/me if me else float("nan")
        inside = dc <= X.max()
        print(f"  crosses {lbl:<28} at dispEmb {dc:.4f}  "
              f"[{'interpolated' if inside else 'EXTRAPOLATED beyond max observed %.4f' % X.max()}]")
json.dump(rows, open(R / "analyze_all.json", "w"), indent=1, default=str)
print("\nwrote M5/analyze_all.json")
