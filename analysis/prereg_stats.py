"""Preregistered contrast set (design 11.4), Holm-corrected.

Design 11.4 fixes 5 contrasts BEFORE any T2 run.  It ALSO states that exact
arms need no testing: "a p-value on an exhaustive enumeration is a category
error."  Contrasts 1-3 are exhaustive or true-by-construction, so they are
reported as exact differences with enumeration completeness, and the Holm
family is the inferential subset {4,5}.  Both corrections are printed.
"""
import json, pathlib, itertools
import numpy as np
from scipy import stats

R = pathlib.Path(__file__).resolve().parent.parent
W = {"eas": 0.36, "wvc": 0.21, "end": 0.19, "lep": 0.16, "nlf": 0.08}
STRUCT = ["eas", "nlf", "lep", "end"]          # 0.79 of identity weight
RAW = ["wvc"]                                   # 0.21, the only positional signal


def jl(p):
    q = R / p
    if not q.exists():
        return []
    return [json.loads(l) for l in q.read_text().splitlines() if l.strip().startswith("{")]


def s_struct(sig):
    num = sum(W[k] * sig[k] for k in STRUCT if sig.get(k) is not None)
    den = sum(W[k] for k in STRUCT if sig.get(k) is not None)
    return num / den if den else None


def s_raw(sig):
    return sig.get("wvc")


# ---------------------------------------------------------------- build arms
Pplus, Ptilde = [], []
for d in jl("M1/results/pairs.jsonl") + jl("M1/results/pairs2.jsonl"):
    sig = d.get("result", {}).get("signals")
    if not sig:
        continue
    rec = dict(a=d["model_a"], b=d["model_b"], sig=sig,
               fam=d.get("family") or d["model_a"].split("/")[-1].split("-")[0])
    if d["ground_truth"] == "related":
        Pplus.append(rec)
for d in jl("M1/results/tier3_positives.jsonl"):
    if d.get("signals"):
        Pplus.append(dict(a=d["parent"], b=d["child"], sig=d["signals"],
                          fam=d["parent"].split("/")[-1].split("-")[0]))
for d in jl("M4/p_tilde_arm.jsonl"):
    Ptilde.append(dict(a=d["a"], b=d["b"], sig=d["signals"],
                       fam=d["a"].split("/")[-1].split("-")[0]))
for d in jl("M4/family_generality.jsonl"):
    Ptilde.append(dict(a=d["a"], b=d["b"], sig=d["signals"], fam=d["family"]))


def arm(recs, fn):
    out = [(r, fn(r["sig"])) for r in recs]
    return [(r, v) for r, v in out if v is not None]


# ------------------------------------------------------------- contrasts 1-2
GATE = {  # exhaustive gate enumeration (M2 gate arm, every point measured)
    "S1": {"X1a": 1.0, "X1b": 1.0, "X2": 1.0, "X1a.X1b.X2": 1.0},
    "S2": {"X1a.M1t": 0.9116, "X2.M1t": 0.9518, "FLAGSHIP X1a.X1b.X2.M1t": 0.8137},
    "M1t": {"M1t-only": 0.9000},
}
BAR = 0.5298  # pi* cut: null p75.  Evasion needs sigma at/below this.

rows = []
s1, s2, m1 = (np.array(list(GATE[k].values())) for k in ("S1", "S2", "M1t"))
rows.append(dict(n="C1  S2 vs S1 (composition beats free symmetry?)",
                 eff=f"sigma_pipe {s2.mean():.4f} vs {s1.mean():.4f}  (delta {s2.mean()-s1.mean():+.4f})",
                 p=None, note=f"EXHAUSTIVE ({len(s1)}+{len(s2)} pts). Both > bar {BAR}. YES it helps, NO it does not evade."))
rows.append(dict(n="C2  S2 vs M1t-alone (composition beats metadata?)",
                 eff=f"sigma_pipe {s2.mean():.4f} vs {m1.mean():.4f}  (delta {s2.mean()-m1.mean():+.4f})",
                 p=None, note=f"EXHAUSTIVE. chi_comp with C1. Both > bar {BAR}."))
rows.append(dict(n="C3  kappa=0 frontier vs B-3 (scratch @ matched kappa)",
                 eff="won by construction (B-3 is empty at kappa=0)",
                 p=None, note="TRUE BY CONSTRUCTION. Reported for completeness; carries no evidential weight."))

# ------------------------------------------------------------- contrasts 4-5
infer = []
for tag, fn in (("C4  P~ vs P+ on s_struct (masking existence)", s_struct),
                ("C5  P~ vs P+ on s_raw / WVC (separability -> RESCORE)", s_raw)):
    A, B = arm(Ptilde, fn), arm(Pplus, fn)
    a, b = np.array([v for _, v in A]), np.array([v for _, v in B])
    if len(a) < 2 or len(b) < 2:
        rows.append(dict(n=tag, eff=f"n(P~)={len(a)} n(P+)={len(b)} INSUFFICIENT", p=None, note="skipped"))
        continue
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = U / (len(a) * len(b))
    # exact permutation on the difference in means
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a)
    idx = list(itertools.combinations(range(len(pool)), na))
    if len(idx) > 200000:
        rng = np.random.default_rng(0)
        perm = [rng.permutation(len(pool))[:na] for _ in range(20000)]
        exact = False
    else:
        perm, exact = idx, True
    diffs = np.array([pool[list(s)].mean() - pool[[j for j in range(len(pool)) if j not in set(s)]].mean()
                      for s in perm])
    pp = (np.sum(np.abs(diffs) >= abs(obs) - 1e-12) + (1 if exact else 0)) / (len(diffs) + (1 if exact else 0))
    infer.append((tag, pp))
    rows.append(dict(n=tag,
                     eff=f"P~ {a.mean():.4f} (n={len(a)})  vs  P+ {b.mean():.4f} (n={len(b)})   AUC {auc:.3f}",
                     p=pp, note=f"{'exact' if exact else '20k Monte-Carlo'} permutation; MWU p={p:.4g}"))

# --------------------------------------------------------------------- Holm
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj

print("=" * 100)
print("PREREGISTERED CONTRAST SET  (design 11.4)   Holm-corrected")
print("=" * 100)
for r in rows:
    print(f"\n{r['n']}")
    print(f"    effect : {r['eff']}")
    print(f"    p      : {'n/a (exhaustive / by construction)' if r['p'] is None else f'{r[chr(39)+chr(39)] if False else r['p']:.5f}'}")
    print(f"    note   : {r['note']}")

if infer:
    ps = [p for _, p in infer]
    adj = holm(ps)
    print("\n" + "-" * 100)
    print(f"HOLM over the INFERENTIAL subset (m={len(ps)}) -- the defensible family:")
    for (tag, p), q in zip(infer, adj):
        print(f"    {tag.split('  ')[0]}  raw p={p:.5f}   Holm p={q:.5f}   {'REJECT H0' if q < 0.05 else 'retain H0'}")
    adj5 = holm(ps + [1.0, 1.0, 1.0])[:len(ps)]
    print(f"\nHOLM over the LITERAL preregistered m=5 (conservative, counts the 3 exhaustive contrasts):")
    for (tag, p), q in zip(infer, adj5):
        print(f"    {tag.split('  ')[0]}  raw p={p:.5f}   Holm p={q:.5f}   {'REJECT H0' if q < 0.05 else 'retain H0'}")

json.dump({"rows": [{k: v for k, v in r.items()} for r in rows],
           "holm_inferential": dict(zip([t for t, _ in infer], holm([p for _, p in infer])) ) if infer else {}},
          open(R / "analysis" / "prereg_stats.json", "w"), indent=1)
print(f"\nwrote analysis/prereg_stats.json")
