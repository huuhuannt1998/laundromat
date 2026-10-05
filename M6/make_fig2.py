"""Rebuild fig2 from the audited corpus (E2), so figure and manifest cannot drift.

Derived scores come from the gold lineage manifest, excluding any row the label
audit disqualified.  Same-recipe scores are the six independently-trained Pythia
pairs.  Same-family are the two from-scratch models in the BERT family.  Unrelated
is the frozen null.
"""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

plt.rcParams.update({"font.size": 7, "font.family": "serif"})
ROOT = pathlib.Path(_ROOT)

man = json.loads((ROOT/"M6"/"gold_lineage_manifest.json").read_text())
derived, same_fam = [], []
for r in man["rows"]:
    if r.get("identity_score") is None or r["analysis_set"] == "excluded":
        continue
    (same_fam if r["relationship_class"] == "N-FAM" else derived).append(r["identity_score"])

same_recipe = [0.7856, 0.6745, 0.7810, 0.7262, 0.7976, 0.8211]  # pythia X vs X-deduped, Table II
                                                            # 0.8211 is the 6.9B pair added by E23
null = json.loads((ROOT/"M2"/"results"/"null_frozen.json").read_text())["scores"]

rows = [("derived\n(true descent)", derived, "#1f4e79"),
        ("same recipe,\nindependent",  same_recipe, "#c00000"),
        ("same family,\nindependent",  same_fam, "#e08214"),
        ("unrelated\n(frozen null)",   null, "#808080")]

fig, ax = plt.subplots(figsize=(3.4, 2.15))
rng = np.random.default_rng(7)
for i, (lab, v, c) in enumerate(rows):
    v = np.asarray(v, dtype=float)
    y = i + (rng.random(len(v)) - 0.5) * (0.34 if len(v) > 50 else 0.22)
    ax.scatter(v, y, s=(2.0 if len(v) > 50 else 13), c=c,
               alpha=(0.10 if len(v) > 50 else 0.85), lw=0, zorder=3)
    ax.text(1.035, i, f"n={len(v)}", fontsize=5.6, va="center", color="0.35")

for x, lab, ls in ((0.75, "0.75 high-conf.", "--"), (0.65, "0.65 weak", ":")):
    ax.axvline(x, color="0.35", lw=0.7, ls=ls, zorder=1)
    ax.text(x, 3.62, lab, fontsize=5.4, ha="center", color="0.35")

ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6)
ax.set_xlabel(r"identity score $\sigma_{\mathrm{id}}$", fontsize=7)
ax.set_xlim(-0.03, 1.10); ax.set_ylim(-0.55, 3.75); ax.invert_yaxis()
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.tick_params(length=2, labelsize=6)
fig.tight_layout(pad=0.15)
fig.savefig(ROOT/"manuscripts"/"laundromat"/"figures"/"fig2_distributions.pdf", dpi=300)

print(f"derived      n={len(derived)}  {min(derived):.4f}-{max(derived):.4f}")
print(f"same-recipe  n={len(same_recipe)}  {min(same_recipe):.4f}-{max(same_recipe):.4f}")
print(f"same-family  n={len(same_fam)}  {min(same_fam):.4f}-{max(same_fam):.4f}")
print(f"null         n={len(null)}")
inside = [s for s in same_recipe if min(derived) <= s <= max(derived)]
print(f"same-recipe points inside the derived range: {len(inside)}/{len(same_recipe)}")
