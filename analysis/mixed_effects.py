"""Design 11.4: sigma_id ~ pair_class + (1|parent_family) + (1|recipe_family).

The design flags this as "the single most likely way to overstate significance":
pairs within a family are not independent, and treating them as independent
inflates significance.  This fits the model and compares it against the naive
pooled test that ignores clustering.
"""
import json, pathlib, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings("ignore")

R = pathlib.Path(__file__).resolve().parent.parent


def jl(p):
    q = R / p
    return [json.loads(l) for l in q.read_text().splitlines() if l.strip().startswith("{")] if q.exists() else []


def fam(n):
    s = n.split("/")[-1].lower()
    for k in ("pythia", "smollm2", "qwen", "bloom", "gpt2", "roberta", "bert", "opt", "bart", "distil"):
        if k in s:
            return k
    return s.split("-")[0]


rows = []
for d in jl("M1/results/pairs.jsonl") + jl("M1/results/pairs2.jsonl"):
    sc = d.get("result", {}).get("scores") or {}
    if sc.get("identity_score") is None:
        continue
    rows.append(dict(sigma=sc["identity_score"],
                     pair_class="derived" if d["ground_truth"] == "related" else "independent",
                     parent_family=fam(d["model_a"]), recipe_family=fam(d["model_b"])))
for d in jl("M1/results/tier3_positives.jsonl"):
    sc = d.get("scores") or {}
    if sc.get("identity_score") is None:
        continue
    rows.append(dict(sigma=sc["identity_score"], pair_class="derived",
                     parent_family=fam(d["parent"]), recipe_family=fam(d["child"])))
for d in jl("M4/p_tilde_arm.jsonl") + jl("M4/family_generality.jsonl"):
    if d["a"] == d["b"]:
        continue
    sc = d.get("scores") or {}
    if sc.get("identity_score") is None:
        continue
    rows.append(dict(sigma=sc["identity_score"], pair_class="independent",
                     parent_family=fam(d["a"]), recipe_family=fam(d["b"])))

df = pd.DataFrame(rows)
print("=" * 92); print("MIXED-EFFECTS MODEL (design 11.4)"); print("=" * 92)
print(f"\nn = {len(df)}   derived = {(df.pair_class=='derived').sum()}   "
      f"independent = {(df.pair_class=='independent').sum()}")
print(f"parent families = {df.parent_family.nunique()}   recipe families = {df.recipe_family.nunique()}")
print("\nby parent family:")
print(df.groupby(["parent_family", "pair_class"]).sigma.agg(["count", "mean"]).to_string())

a = df[df.pair_class == "derived"].sigma.values
b = df[df.pair_class == "independent"].sigma.values
t, p_naive = stats.ttest_ind(a, b, equal_var=False)
U, p_mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
print(f"\n--- NAIVE pooled (ignores family clustering; this is the OVERSTATED one) ---")
print(f"    derived {a.mean():.4f} (n={len(a)})   independent {b.mean():.4f} (n={len(b)})")
print(f"    Welch t p = {p_naive:.5f}    Mann-Whitney p = {p_mwu:.5f}    AUC = {U/(len(a)*len(b)):.3f}")

print(f"\n--- MIXED-EFFECTS: sigma ~ pair_class + (1|parent_family) ---")
try:
    m = smf.mixedlm("sigma ~ pair_class", df, groups=df.parent_family).fit(reml=False)
    co = m.params.get("pair_class[T.independent]", np.nan)
    pv = m.pvalues.get("pair_class[T.independent]", np.nan)
    print(f"    coef(independent) = {co:+.4f}   p = {pv:.5f}")
    print(f"    group variance    = {m.cov_re.iloc[0,0]:.6f}   residual = {m.scale:.6f}")
    icc = m.cov_re.iloc[0, 0] / (m.cov_re.iloc[0, 0] + m.scale)
    print(f"    ICC (family)      = {icc:.4f}   <- fraction of variance that is family, not class")
    print(f"\n    INFLATION FACTOR: naive p {p_naive:.5f} -> clustered p {pv:.5f}  "
          f"({pv/p_naive:.1f}x)" if p_naive > 0 else "")
except Exception as e:
    print(f"    FAILED TO CONVERGE: {e}")
    print("    With this few families the random-intercept model is not identifiable.")

print(f"\n--- crossed (1|parent_family) + (1|recipe_family) ---")
try:
    df["_g"] = 1
    vc = {"pf": "0 + C(parent_family)", "rf": "0 + C(recipe_family)"}
    m2 = smf.mixedlm("sigma ~ pair_class", df, groups=df["_g"], vc_formula=vc).fit(reml=False)
    print(f"    coef(independent) = {m2.params.get('pair_class[T.independent]', np.nan):+.4f}   "
          f"p = {m2.pvalues.get('pair_class[T.independent]', np.nan):.5f}")
    print(f"    vc: {dict(m2.vcomp.round(6)) if hasattr(m2,'vcomp') else 'n/a'}")
except Exception as e:
    print(f"    FAILED: {str(e)[:150]}")

df.to_csv(R / "analysis" / "mixed_effects_data.csv", index=False)
print("\nwrote analysis/mixed_effects_data.csv")
