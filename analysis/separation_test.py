"""CORRECTION: bootstrap on a min(pos)-max(neg) margin is DEGENERATE.

Resampling with replacement can only yield min >= observed min and max <= observed max, so the
margin can only move UP. The lower CI bound is pinned at the point estimate by construction and
says nothing. I nearly reported that as validation.

Correct instrument for "does this signal separate these classes" at small n:
  AUC (= Mann-Whitney U / n_pos*n_neg), with an EXACT permutation test on the class labels.
"""
import json,pathlib,itertools
import numpy as np

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

R=pathlib.Path(_ROOT)
rng=np.random.default_rng(11)

def auc(pos,neg):
    p=np.array(pos,float); n=np.array(neg,float)
    gt=(p[:,None]>n[None,:]).sum(); eq=(p[:,None]==n[None,:]).sum()
    return (gt+0.5*eq)/(len(p)*len(n))

def perm_p(pos,neg,n_perm=200000):
    """Exact if the label space is small enough, else Monte-Carlo. One-sided: AUC > 0.5."""
    obs=auc(pos,neg); allv=np.array(list(pos)+list(neg),float); k=len(pos)
    total=len(allv)
    n_exact=1
    for i in range(k): n_exact=n_exact*(total-i)//(i+1)
    if n_exact<=200000:
        cnt=0; tot=0
        for idx in itertools.combinations(range(total),k):
            m=np.zeros(total,bool); m[list(idx)]=True
            if auc(allv[m],allv[~m])>=obs: cnt+=1
            tot+=1
        return obs,cnt/tot,f"exact ({tot} labelings)"
    cnt=0
    for _ in range(n_perm):
        pm=rng.permutation(allv)
        if auc(pm[:k],pm[k:])>=obs: cnt+=1
    return obs,cnt/n_perm,f"monte-carlo ({n_perm})"

def boot_auc(pos,neg,n=10000):
    p=np.array(pos,float); q=np.array(neg,float)
    d=[auc(p[rng.integers(0,len(p),len(p))],q[rng.integers(0,len(q),len(q))]) for _ in range(n)]
    return float(np.percentile(d,2.5)),float(np.percentile(d,97.5))

def report(label,pos,neg):
    a,p,how=perm_p(pos,neg); lo,hi=boot_auc(pos,neg)
    verdict = "SEPARATES" if p<0.05 else "NOT ESTABLISHED (p>=0.05)"
    print(f"{label:<40} n={len(pos)}/{len(neg)}  AUC={a:.3f}  95%CI[{lo:.3f},{hi:.3f}]  "
          f"perm p={p:.4f} {how:<24} -> {verdict}")

rows=[]
for f in ["M1/results/pairs.jsonl","M1/results/pairs2.jsonl"]:
    for l in (R/f).read_text().splitlines():
        d=json.loads(l); s=d["result"].get("signals")
        if s: rows.append((d["ground_truth"],s.get("eas"),s.get("wvc"),
                           d["result"]["scores"]["identity_score"],d["result"]["scores"]["mfi_tier"]))

# expanded tier-3 positive arm (built to close gate G0's unmet >=12 criterion)
_t3=pathlib.Path(str(_ROOT) + "/M1/results/tier3_positives.jsonl")
if _t3.exists():
    for _l in [x for x in _t3.read_text().splitlines() if x.strip().startswith("{")]:
        _d=json.loads(_l); _s=_d["scores"]; _g=_d["signals"]
        if not _g: continue
        rows.append({"gt":"related","sig":_g,"tier":_s["mfi_tier"],"id":_s["identity_score"],
                     "pair":_d["parent"].split("/")[-1]+"|"+_d["child"].split("/")[-1]}
                    if "ablations" in __file__ else
                    ("related",_g.get("eas"),_g.get("wvc"),_s["identity_score"],_s["mfi_tier"]))

print("="*118)
print("SEPARATION TESTS — AUC + permutation test (replaces the degenerate margin bootstrap)")
print("="*118)
report("M1 EAS: related vs unrelated",
       [r[1] for r in rows if r[0]=="related" and r[1] is not None],
       [r[1] for r in rows if r[0]=="unrelated" and r[1] is not None])
report("M1 WVC: related vs unrelated",
       [r[2] for r in rows if r[0]=="related" and r[2] is not None],
       [r[2] for r in rows if r[0]=="unrelated" and r[2] is not None])
report("B1 sigma_id tier-3: related vs unrelated",
       [r[3] for r in rows if r[0]=="related" and r[4]==3],
       [r[3] for r in rows if r[0]=="unrelated" and r[4]==3])
dv=json.loads((R/"M4/defence_full.json").read_text())
report("M4 LAP defence: derived vs not-derived",
       [d["lap"] for d in dv if d["class"].startswith("P+") and d["lap"]==d["lap"]],
       [d["lap"] for d in dv if not d["class"].startswith("P+") and d["lap"]==d["lap"]])
# MPK's own identity score on the same 9 pairs the defence was scored on, for a like-for-like contrast
print("\nNote: B1's claim is OVERLAP (AUC materially < 1), not separation. AUC=1.0 with p<0.05")
print("      would REFUTE overlap; a mid AUC with p>=0.05 leaves it unestablished either way.")
