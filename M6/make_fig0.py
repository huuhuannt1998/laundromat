# OBSOLETE (2026-10-04): wrote only into the removed [READY] (S&P 2027) folder; the TMLR manuscript's F1 (this schematic) comes from '[READY] (TMLR) LAUNDROMAT - P6/floats/make_floats.py'.
"""Figure 0 -- the system this paper studies, end to end.

Schematic. It merges what was Figure 1 (the verifier's tier logic) with the supply
chain either side of it, so one figure orients the reader in the introduction instead
of two figures splitting that job. Numbers in the caption come from Sec. soundness,
Sec. denial and Table 'deploy', not from this script.

Text is plain unicode: matplotlib is not LaTeX, so \\texttt and \\S render literally.
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
plt.rcParams.update({'font.size': 7, 'font.family': 'serif'})
OUT = _ROOT/"[READY] (S&P 2027) LAUNDROMAT - P6"/"figures"/"fig0_architecture.pdf"

RED, BLUE, GREY, GREEN = '#c00000', '#1f4e79', '0.25', '#1a6b1a'
fig, ax = plt.subplots(figsize=(3.4, 3.25))
ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10.2)

def box(x, y, w, h, txt, fc, ec=GREY, fs=5.9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                                fc=fc, ec=ec, lw=0.7, zorder=3))
    ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=fs, zorder=4,
            fontweight=('bold' if bold else 'normal'))

def arr(x1, y1, x2, y2, c=GREY, lw=0.8, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', color=c,
                                 lw=lw, linestyle=ls, mutation_scale=7, zorder=2))

# ---- lanes --------------------------------------------------------------
for x, t in ((1.6, "PUBLISHER"), (5.05, "VERIFIER (studied)"), (8.5, "CONSUMER")):
    ax.text(x, 9.85, t, fontsize=6.0, color=GREY, ha='center', fontweight='bold')
for x in (3.3, 6.8):
    ax.plot([x, x], [2.35, 9.55], color='0.85', lw=0.6, zorder=1)

# ---- publisher ----------------------------------------------------------
box(0.15, 8.35, 3.0, 0.80, "model weights", '#eeeeee')
box(0.15, 6.95, 3.0, 0.95, "config.json\n(publisher-written)", '#f6d4d4', ec=RED)
ax.text(1.65, 6.62, "untrusted input", fontsize=5.2, color=RED, ha='center')
box(0.15, 4.55, 3.0, 0.80, "5 weight signals", '#dce9f5')
arr(1.65, 8.30, 1.65, 5.42)                       # weights -> the signals computed from them
ax.text(1.80, 6.05, "", fontsize=5.2)

# ---- verifier -----------------------------------------------------------
box(3.55, 7.10, 3.0, 0.80, "MFI tier gate", '#e8e8e8')
arr(3.20, 7.45, 3.50, 7.48)                       # config decides the tier
box(3.55, 5.70, 3.0, 0.72, "tier 1: pin 1.0", '#f6d4d4', ec=RED, bold=True)
box(3.55, 4.80, 3.0, 0.72, "tier 2: cap 0.9", '#fae6cc')
box(3.55, 3.90, 3.0, 0.72, "tier 3: use $\\sigma_{id}$", '#dce9f5')
# route the three outcomes down a trunk to the right of the boxes, branching in.
# Drawing them straight down would pass BEHIND the tier-1 box and read as stubs.
TRUNK = 6.68
ax.plot([5.05, TRUNK], [7.06, 7.06], color=GREY, lw=0.8, zorder=2)
ax.plot([TRUNK, TRUNK], [4.26, 7.06], color=GREY, lw=0.8, zorder=2)
for ye in (6.06, 5.16, 4.26):
    arr(TRUNK, ye, 6.60, ye)
arr(3.20, 4.90, 3.50, 4.30, c=BLUE)               # signals reach only tier 3
ax.text(0.15, 4.18, "consulted only at tier 3", fontsize=5.2, color=BLUE)

# ---- consumer -----------------------------------------------------------
box(7.05, 7.10, 2.85, 0.80, "verdict", '#eeeeee')
arr(6.60, 7.50, 7.00, 7.50)
box(7.05, 5.70, 2.85, 0.90, "ML-BOM\npedigree.ancestors", '#dce9f5', fs=5.5)
arr(8.47, 7.06, 8.47, 6.65)
box(7.05, 4.20, 2.85, 0.90, "OPA admission gate", '#dce9f5')
arr(8.47, 5.66, 8.47, 5.15)
box(7.05, 2.70, 2.85, 0.90, "licence, advisory,\nquarantine", '#e6f0e6')
arr(8.47, 4.16, 8.47, 3.65)

# ---- findings, one per line, full width so nothing collides -------------
ax.plot([0.15, 9.9], [2.15, 2.15], color='0.85', lw=0.6)
rows = [("§5    same recipe, no shared weights ⇒ confirmed match", RED),
        ("§5.5  the gate overrides contradicting weight evidence", RED),
        ("§6    no zero-compute transform evades the score", BLUE),
        ("§8    one absent key ⇒ no verdict at all", RED),
        ("§9    in our pipeline, 8 of 32 admitted despite inheriting", RED),
        ("§10   alignment separates the two classes", GREEN)]
for i, (t, c) in enumerate(rows):
    ax.text(0.15, 1.80 - i*0.345, t, fontsize=5.3, color=c, ha='left', va='center')

fig.tight_layout(pad=0.05)
fig.savefig(OUT, dpi=300)
print("wrote", OUT)
