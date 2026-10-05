"""E13 -- robust displacement regression with uncertainty.

The paper currently reports a point fit and an R^2. The review asks for the
uncertainty that actually matters: a confidence interval on the threshold crossing,
plus leave-one-out and robustness checks, and a linear-vs-quadratic comparison.
Pure recomputation over frozen sweep files; no model is touched.
"""
import json, pathlib
import numpy as np

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

R = pathlib.Path(_ROOT)
EXCL = {"nlpaueb/legal-bert-base-uncased",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "emilyalsentzer/Bio_ClinicalBERT"}
pts = []
for f in ("M5/sweep_bert.jsonl", "M5/corpus_distance.jsonl"):
    for line in (R/f).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line); d = r.get("displacement", {}); s = r.get("scores", {})
        if d.get("embedding") is not None and s.get("identity_score") is not None \
           and r["child"] not in EXCL:
            pts.append((r["child"], r["parent"], d["embedding"], s["identity_score"]))
x = np.array([p[2] for p in pts]); y = np.array([p[3] for p in pts])
name = [p[0] for p in pts]; fam = [p[1].split("/")[-1] for p in pts]
n = len(x)

def fit(xx, yy, deg=1):
    c = np.polyfit(xx, yy, deg); pred = np.polyval(c, xx)
    ss = 1 - ((yy-pred)**2).sum()/((yy-yy.mean())**2).sum()
    return c, ss
def cross(c, t):
    """d at which the fit crosses t (linear only)."""
    return (t - c[1]) / c[0]
def theilsen(xx, yy):
    sl = [ (yy[j]-yy[i])/(xx[j]-xx[i]) for i in range(len(xx)) for j in range(i+1,len(xx))
           if xx[j] != xx[i] ]
    m = float(np.median(sl)); b = float(np.median(yy - m*xx)); return np.array([m, b])

c1, r2_1 = fit(x, y, 1)
c2, r2_2 = fit(x, y, 2)
print(f"n = {n}, last observation d_emb = {x.max():.4f}")
print(f"linear    : sigma = {c1[1]:.4f} {c1[0]:+.4f} d      R2 = {r2_1:.4f}")
print(f"quadratic : R2 = {r2_2:.4f}   (delta R2 = {r2_2-r2_1:+.4f})")
ts = theilsen(x, y)
print(f"Theil-Sen : sigma = {ts[1]:.4f} {ts[0]:+.4f} d   (robust, median-of-slopes)")

# --- bootstrap ---
rng = np.random.default_rng(17); B = 10000
sl, ic, c65, c53 = [], [], [], []
for _ in range(B):
    idx = rng.integers(0, n, n)
    if len(set(x[idx])) < 2: continue
    c = np.polyfit(x[idx], y[idx], 1)
    sl.append(c[0]); ic.append(c[1])
    if c[0] != 0:
        c65.append(cross(c, 0.65)); c53.append(cross(c, 0.5298))
q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
print(f"\nbootstrap ({B} resamples, seed 17)")
print(f"  slope      {c1[0]:+.4f}  95% CI [{q(sl)[0]:+.4f}, {q(sl)[1]:+.4f}]")
print(f"  intercept  {c1[1]:.4f}  95% CI [{q(ic)[0]:.4f}, {q(ic)[1]:.4f}]")
print(f"  d* at 0.65    {cross(c1,0.65):.3f}  95% CI [{q(c65)[0]:.3f}, {q(c65)[1]:.3f}]")
print(f"  d* at 0.5298  {cross(c1,0.5298):.3f}  95% CI [{q(c53)[0]:.3f}, {q(c53)[1]:.3f}]")

# --- leave-one-out ---
print(f"\nleave-one-model-out: slope and d*(0.65) range")
lo_s, lo_c = [], []
for i in range(n):
    m = np.ones(n, bool); m[i] = False
    c = np.polyfit(x[m], y[m], 1); lo_s.append(c[0]); lo_c.append(cross(c, 0.65))
worst = int(np.argmax(np.abs(np.array(lo_c) - cross(c1, 0.65))))
print(f"  slope  {min(lo_s):+.4f} .. {max(lo_s):+.4f}")
print(f"  d*     {min(lo_c):.3f} .. {max(lo_c):.3f}   most influential: {name[worst]}")
print(f"\nleave-one-parent-family-out")
for f in sorted(set(fam)):
    m = np.array([ff != f for ff in fam])
    if m.sum() < 3 or len(set(x[m])) < 2: continue
    c, r2 = fit(x[m], y[m], 1)
    print(f"  drop {f:<16} n={m.sum():<3} sigma = {c[1]:.4f} {c[0]:+.4f} d  R2={r2:.3f}  "
          f"d*(0.65)={cross(c,0.65):.3f}")

json.dump(dict(n=n, last_obs=float(x.max()),
               linear=dict(slope=float(c1[0]), intercept=float(c1[1]), r2=float(r2_1)),
               quadratic_r2=float(r2_2), theilsen=dict(slope=float(ts[0]), intercept=float(ts[1])),
               boot=dict(B=B, seed=17,
                         slope_ci=[float(v) for v in q(sl)], intercept_ci=[float(v) for v in q(ic)],
                         d065=float(cross(c1,0.65)), d065_ci=[float(v) for v in q(c65)],
                         d05298=float(cross(c1,0.5298)), d05298_ci=[float(v) for v in q(c53)]),
               loo=dict(slope_min=float(min(lo_s)), slope_max=float(max(lo_s)),
                        d065_min=float(min(lo_c)), d065_max=float(max(lo_c)),
                        most_influential=name[worst])),
          open(R/"M7"/"e13_regression.json","w"), indent=1)
print("\nwrote M7/e13_regression.json")
