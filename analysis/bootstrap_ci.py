"""Fix the blanket 'never a bare point estimate' violation: BCa-style bootstrap intervals
over every claim already collected. No new model runs -- pure recomputation."""
import json,pathlib,random,statistics as st
import numpy as np

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

R=pathlib.Path(_ROOT)
rng=np.random.default_rng(11)

def boot(vals, stat=np.median, n=10000):
    v=np.array(vals,dtype=float)
    if len(v)<2: return (float(stat(v)), float("nan"), float("nan"))
    idx=rng.integers(0,len(v),size=(n,len(v)))
    d=np.array([stat(v[i]) for i in idx])
    return float(stat(v)), float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

def boot_margin(pos,neg,n=10000):
    """CI on min(pos) - max(neg), the separability margin."""
    p=np.array(pos,dtype=float); q=np.array(neg,dtype=float)
    d=np.array([p[rng.integers(0,len(p),len(p))].min()-q[rng.integers(0,len(q),len(q))].max()
                for _ in range(n)])
    return float(p.min()-q.max()), float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

print("="*94); print("BOOTSTRAP INTERVALS (10 000 resamples) — recomputed from collected artifacts")
print("="*94)

# ---- 1. M1 corpus: EAS and WVC by class, and the EAS separability margin
rows=[]
for f in ["M1/results/pairs.jsonl","M1/results/pairs2.jsonl"]:
    for l in (R/f).read_text().splitlines():
        d=json.loads(l); s=d["result"].get("signals")
        if s: rows.append((d["ground_truth"],s.get("eas"),s.get("wvc"),
                           d["result"]["scores"]["identity_score"],d["result"]["scores"]["mfi_tier"]))
for gt in ("related","unrelated"):
    e=[r[1] for r in rows if r[0]==gt and r[1] is not None]
    w=[r[2] for r in rows if r[0]==gt and r[2] is not None]
    m,lo,hi=boot(e); print(f"M1 EAS  {gt:<10} n={len(e):>2}  median={m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    m,lo,hi=boot(w); print(f"M1 WVC  {gt:<10} n={len(w):>2}  median={m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
ep=[r[1] for r in rows if r[0]=="related" and r[1] is not None]
en=[r[1] for r in rows if r[0]=="unrelated" and r[1] is not None]
m,lo,hi=boot_margin(ep,en)
print(f"\nM1 EAS separability margin = {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
      f"   -> {'separates' if lo>0 else 'CI CROSSES ZERO: separation NOT established'}")

# ---- 2. B1: tier-3 (weight-decided) overlap margin -- the n=3 claim
tp=[r[3] for r in rows if r[0]=="related" and r[4]==3]
tn=[r[3] for r in rows if r[0]=="unrelated" and r[4]==3]
m,lo,hi=boot_margin(tp,tn)
print(f"\nB1 tier-3 margin (n_pos={len(tp)}, n_neg={len(tn)}) = {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"   -> {'overlap established' if hi<0 else 'CI CROSSES ZERO: overlap NOT established at 95%'}")

# ---- 3. M4 defence separation
dv=json.loads((R/"M4/defence_full.json").read_text())
pos=[d["lap"] for d in dv if d["class"].startswith("P+") and d["lap"]==d["lap"]]
neg=[d["lap"] for d in dv if not d["class"].startswith("P+") and d["lap"]==d["lap"]]
m,lo,hi=boot_margin(pos,neg)
print(f"\nM4 LAP defence margin (n_pos={len(pos)}, n_neg={len(neg)}) = {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"   -> {'separation established' if lo>0 else 'CI CROSSES ZERO: separation NOT established'}")

# ---- 4. P~ ladder: is the EAS-rises-with-scale trend real?
pt=[json.loads(l) for l in (R/"M4/p_tilde_arm.jsonl").read_text().splitlines()]
pt=[d for d in pt if d["class"]=="P~"]
eas=[d["signals"]["eas"] for d in pt]; wvc=[d["signals"]["wvc"] for d in pt]
sizes=[70,160,410,1000][:len(eas)]
if len(eas)>=3:
    from numpy.polynomial import polynomial as P
    lx=np.log10(sizes)
    se=np.polyfit(lx,eas,1)[0]; sw=np.polyfit(lx,wvc,1)[0]
    bs_e=[];bs_w=[]
    for _ in range(10000):
        i=rng.integers(0,len(eas),len(eas))
        if len(set(i.tolist()))<2: continue
        bs_e.append(np.polyfit(lx[i],np.array(eas)[i],1)[0])
        bs_w.append(np.polyfit(lx[i],np.array(wvc)[i],1)[0])
    print(f"\nP~ ladder n={len(eas)}  EAS slope per log10(params) = {se:+.4f} "
          f"95% CI [{np.percentile(bs_e,2.5):+.4f}, {np.percentile(bs_e,97.5):+.4f}]")
    print(f"P~ ladder      WVC slope per log10(params) = {sw:+.4f} "
          f"95% CI [{np.percentile(bs_w,2.5):+.4f}, {np.percentile(bs_w,97.5):+.4f}]")
    print(f"   -> n=4 over one family; treat slopes as descriptive, not inferential")
