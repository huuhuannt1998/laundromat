"""C4/C5 corrected: identity pair removed from P~, shape-stratified, TOST for C4."""
import json, pathlib, itertools
import numpy as np
from scipy import stats

R = pathlib.Path(__file__).resolve().parent.parent
DIMS = json.load(open(R / "analysis" / "model_dims.json"))
W = {"eas": 0.36, "end": 0.19, "lep": 0.16, "nlf": 0.08}


def jl(p):
    q = R / p
    return [json.loads(l) for l in q.read_text().splitlines() if l.strip().startswith("{")] if q.exists() else []


def s_struct(s):
    n = sum(W[k] * s[k] for k in W if s.get(k) is not None)
    d = sum(W[k] for k in W if s.get(k) is not None)
    return n / d if d else None


def shape(a, b):
    da, db = DIMS.get(a), DIMS.get(b)
    if not da or not db or None in da or None in db:
        return None
    return "SAME-SHAPE" if tuple(da) == tuple(db) else "DIFF-SHAPE"


Pp, Pt = [], []
for d in jl("M1/results/pairs.jsonl") + jl("M1/results/pairs2.jsonl"):
    s = d.get("result", {}).get("signals")
    if s and d["ground_truth"] == "related":
        Pp.append((d["model_a"], d["model_b"], s))
for d in jl("M1/results/tier3_positives.jsonl"):
    if d.get("signals"):
        Pp.append((d["parent"], d["child"], d["signals"]))
for d in jl("M4/p_tilde_arm.jsonl") + jl("M4/family_generality.jsonl"):
    if d["a"] == d["b"]:
        print(f"  EXCLUDED from P~ (identity pair, belongs to control arm): {d['a']}")
        continue
    Pt.append((d["a"], d["b"], d["signals"]))
# dedupe P+ on (a,b)
seen, Pp2 = set(), []
for a, b, s in Pp:
    if (a, b) in seen:
        continue
    seen.add((a, b)); Pp2.append((a, b, s))
Pp = Pp2


def perm_p(a, b):
    obs = a.mean() - b.mean(); pool = np.concatenate([a, b]); na = len(a)
    combs = list(itertools.combinations(range(len(pool)), na))
    if len(combs) > 300000:
        rng = np.random.default_rng(0)
        combs = [tuple(rng.permutation(len(pool))[:na]) for _ in range(20000)]
        ex = False
    else:
        ex = True
    ds = np.array([pool[list(c)].mean() - pool[[j for j in range(len(pool)) if j not in set(c)]].mean() for c in combs])
    return (np.sum(np.abs(ds) >= abs(obs) - 1e-12) + (1 if ex else 0)) / (len(ds) + (1 if ex else 0)), ex


def run(tag, fn, stratum):
    A = [(x, fn(s)) for x, _, s in [(f"{a}|{b}", None, s) for a, b, s in Pt] ] if False else None
    a = np.array([v for x, y, s in Pt if (v := fn(s)) is not None and (stratum is None or shape(x, y) == stratum)])
    b = np.array([v for x, y, s in Pp if (v := fn(s)) is not None and (stratum is None or shape(x, y) == stratum)])
    lbl = stratum or "ALL"
    if len(a) < 2 or len(b) < 2:
        print(f"  {tag:<10} [{lbl:<10}] n(P~)={len(a)} n(P+)={len(b)}  -- insufficient"); return None
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = U / (len(a) * len(b))
    p, ex = perm_p(a, b)
    print(f"  {tag:<10} [{lbl:<10}] P~ {a.mean():.4f} (n={len(a)})  P+ {b.mean():.4f} (n={len(b)})   "
          f"AUC {auc:.3f}  p={p:.4f} {'exact' if ex else 'MC'}")
    return dict(stratum=lbl, pt=a.tolist(), pp=b.tolist(), auc=auc, p=p)


print("=" * 96)
print("C4 / C5 CORRECTED  (identity pair removed; stratified by shape compatibility)")
print("=" * 96)
res = {}
print("\nC4  s_struct  (structural 0.79 of identity weight)  -- masking claim")
for st in (None, "SAME-SHAPE", "DIFF-SHAPE"):
    r = run("C4", s_struct, st)
    if r: res[f"C4/{r['stratum']}"] = r
print("\nC5  s_raw = WVC  (the only positional signal, 0.21)  -- separability claim")
for st in (None, "SAME-SHAPE", "DIFF-SHAPE"):
    r = run("C5", lambda s: s.get("wvc"), st)
    if r: res[f"C5/{r['stratum']}"] = r

# TOST equivalence for C4 (masking = the two arms are EQUIVALENT, not merely not-different)
print("\n" + "-" * 96)
print("C4 as an EQUIVALENCE claim (TOST).  Masking means 'indistinguishable', which a")
print("failure-to-reject does NOT establish.  Equivalence margin delta predeclared at 0.05")
print("of s_struct (about 1/20 of the signal's usable range).")
for key in ("C4/ALL", "C4/SAME-SHAPE"):
    if key not in res:
        continue
    a, b = np.array(res[key]["pt"]), np.array(res[key]["pp"])
    d = 0.05
    t1, p1 = stats.ttest_ind(a, b - d, equal_var=False, alternative="greater")
    t2, p2 = stats.ttest_ind(a, b + d, equal_var=False, alternative="less")
    ptost = max(p1, p2)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    print(f"  {key:<16} diff {diff:+.4f}  90%-ish CI [{lo:+.4f},{hi:+.4f}]  TOST p={ptost:.4f}  "
          f"{'EQUIVALENT within +/-0.05' if ptost < 0.05 else 'equivalence NOT established'}")

json.dump(res, open(R / "analysis" / "prereg_stats2.json", "w"), indent=1)
print("\nwrote analysis/prereg_stats2.json")
