"""E18 analysis: class coverage, overall separation, and a FAMILY-DISJOINT holdout.

The prior split was on parameter count, which placed every distillation pair in
development. Splitting on parent family instead tests whether the threshold transfers
to architectures never seen during development, which is what generalisation means here.
"""
import json, pathlib, collections, itertools

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

R=pathlib.Path(_ROOT)
rows=[r for r in json.loads((R/"M7"/"e18_defence.json").read_text()) if r.get("lap") is not None]
drop=[r for r in json.loads((R/"M7"/"e18_defence.json").read_text()) if r.get("lap") is None]

POS={"P+"}
def lab(r): return "derived" if r["cls"] in POS else "negative"

print(f"n scored = {len(rows)}   n unscorable = {len(drop)}")
for r in drop:
    print(f"   unscorable: {r['a'].split('/')[-1]} | {r['b'].split('/')[-1]}  "
          f"(shape-incompatible MLP layouts)")

by=collections.defaultdict(list)
for r in rows: by[(r["cls"],r["kind"])].append(r["lap"])
print("\nclass coverage")
for k in sorted(by):
    v=by[k]; print(f"  {k[0]:<3} {k[1]:<12} n={len(v):<3} {min(v):.4f}-{max(v):.4f}")

pos=[r["lap"] for r in rows if lab(r)=="derived"]
neg=[r["lap"] for r in rows if lab(r)=="negative"]
print(f"\nderived  n={len(pos)}  {min(pos):.4f}-{max(pos):.4f}")
print(f"negative n={len(neg)}  {min(neg):.4f}-{max(neg):.4f}")
print(f"MARGIN (min derived - max negative) = {min(pos)-max(neg):+.4f}")
def auc(p,n): return sum((1.0 if a>b else .5 if a==b else 0.) for a in p for b in n)/(len(p)*len(n))
print(f"AUC = {auc(pos,neg):.4f}   (perfect ordering: {'yes' if min(pos)>max(neg) else 'NO'})")

# --- family-disjoint holdout ---
# Balanced family-disjoint split: BOTH arms must carry positives and negatives,
# otherwise the holdout tests only recall. Development keeps the hardest negative
# (legal-bert, trained from scratch with its own vocabulary); the held-out families
# are never seen during threshold fitting.
DEV={"bert","bert-cased","mbert","roberta"}
dev=[r for r in rows if r["family"] in DEV]
hld=[r for r in rows if r["family"] not in DEV]
def arms(rs): return ([r["lap"] for r in rs if lab(r)=="derived"],
                      [r["lap"] for r in rs if lab(r)=="negative"])
dp,dn=arms(dev); hp,hn=arms(hld)
thr=(min(dp)+max(dn))/2
print(f"\n--- family-disjoint holdout ---")
print(f"development families {sorted(DEV)}: n={len(dev)} ({len(dp)} derived, {len(dn)} negative)")
print(f"held-out families    {sorted({r['family'] for r in hld})}: n={len(hld)} ({len(hp)} derived, {len(hn)} negative)")
print(f"threshold fitted on development only = {thr:.4f}   (dev margin {min(dp)-max(dn):+.4f})")
tp=sum(1 for v in hp if v>thr); fn=len(hp)-tp
tn=sum(1 for v in hn if v<=thr); fp=len(hn)-tn
print(f"held-out: TP {tp}  FN {fn}  TN {tn}  FP {fp}   accuracy {(tp+tn)/max(1,len(hp)+len(hn)):.3f}")
if hp and hn: print(f"held-out margin = {min(hp)-max(hn):+.4f}")
print(f"held-out derived  : {sorted(round(v,4) for v in hp)}")
print(f"held-out negative : {sorted(round(v,4) for v in hn)}")
json.dump(dict(n_scored=len(rows), n_unscorable=len(drop),
               derived=[min(pos),max(pos)], negative=[min(neg),max(neg)],
               margin=min(pos)-max(neg), auc=auc(pos,neg),
               dev_families=sorted(DEV), threshold=thr,
               dev_margin=min(dp)-max(dn), heldout_n=len(hld),
               heldout_tp=tp, heldout_fn=fn, heldout_tn=tn, heldout_fp=fp,
               heldout_margin=(min(hp)-max(hn)) if hp and hn else None),
          open(R/"M7"/"e18_holdout.json","w"), indent=1)
print("\nwrote M7/e18_holdout.json")
