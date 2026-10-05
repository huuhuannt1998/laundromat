"""E11 -- how much does the 0.5298 evasion bar depend on the null we happened to draw?

The bar is the 75th percentile of a frozen unrelated-pair distribution. The review asks
whether the no-zero-compute-evasion conclusion survives anywhere inside the uncertainty
on that percentile. It does not need the null's per-score labels, so the parts below are
computable; the stratified-by-scale and same-family-versus-cross-family splits DO need
labels the frozen artefact does not retain, and are reported as blocked rather than
approximated.
"""
import json, pathlib, statistics as st
import numpy as np
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

froz = json.loads((_ROOT/"M2"/"results"/"null_frozen.json").read_text())
s = np.array(froz["scores"], dtype=float)
n = len(s)
p75 = float(np.percentile(s, 75))
print(f"frozen null: n = {n}, p75 = {p75:.4f} (paper uses {froz['quantiles']['75']})")

rng = np.random.default_rng(17); B = 20000
bs = np.array([np.percentile(rng.choice(s, n, replace=True), 75) for _ in range(B)])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\nbootstrap ({B} resamples, seed 17)")
print(f"  median p75 {np.median(bs):.4f}   95% CI [{lo:.4f}, {hi:.4f}]   min {bs.min():.4f}  max {bs.max():.4f}")

# disjoint halves, repeated: independent-ish null sets from the same population
halves = []
for k in range(500):
    idx = rng.permutation(n); a, b = idx[:n//2], idx[n//2:]
    halves += [np.percentile(s[a], 75), np.percentile(s[b], 75)]
halves = np.array(halves)
print(f"  500 disjoint half-splits: p75 {halves.min():.4f}..{halves.max():.4f}, median {np.median(halves):.4f}")

# a differently-constructed null: every scan match, without the frozen set's exclusions
alt = []
for line in (_ROOT/"M2"/"results"/"wide_scans.jsonl").read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    for m in (r.get("result") or {}).get("matches", []):
        v = (m.get("scores") or {}).get("identity_score")
        if v is not None: alt.append(v)
print(f"  alternative construction (all {len(alt)} scan matches, no exclusions): p75 = {np.percentile(alt,75):.4f}")

# --- does any conclusion move inside the interval? ---
print(f"\ndoes the evasion conclusion change anywhere in [{lo:.4f}, {hi:.4f}]?")
import csv
mat = list(csv.DictReader(open(_ROOT/"M7"/"e15_transform_matrix.csv")))
def cap(r):
    if r["exact"] == "True": return "exact"
    d = r["max_dlogit"]
    if not d: return "unmeasured"
    return "preserved" if float(d) <= 1e-2 else "BROKEN"
pres = [(r["arm"], float(r["identity"])) for r in mat
        if r["identity"] and cap(r) in ("exact", "preserved")]
worst = min(pres, key=lambda t: t[1]) if pres else None
print(f"  capability-preserving arms: {len(pres)}   lowest identity score {worst[1]:.4f} ({worst[0]})")
print(f"  bar would have to rise to {worst[1]:.4f} to make any of them an evasion;")
print(f"  the upper end of the 95% interval is {hi:.4f}, short of it by {worst[1]-hi:.4f}.")
print(f"  CONCLUSION UNCHANGED across the whole interval." if hi < worst[1] else
      f"  WARNING: the conclusion is threshold-sensitive.")
json.dump(dict(n=n, p75=p75, boot=dict(B=B, seed=17, median=float(np.median(bs)),
               ci=[float(lo), float(hi)], min=float(bs.min()), max=float(bs.max())),
               half_split=dict(min=float(halves.min()), max=float(halves.max())),
               alt_construction_p75=float(np.percentile(alt,75)), alt_n=len(alt),
               lowest_capability_preserving=dict(arm=worst[0], identity=worst[1]),
               conclusion_stable=bool(hi < worst[1]),
               blocked="per-score stratum labels are not retained in null_frozen.json, so the "
                       "stratified-by-scale and same-family-vs-cross-family splits could not be run"),
          open(_ROOT/"M7"/"e11_null_robustness.json","w"), indent=1)
print("\nwrote M7/e11_null_robustness.json")
