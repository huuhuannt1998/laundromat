# OBSOLETE (2026-10-04): wrote only into the removed [READY] (S&P 2027) folder; the TMLR manuscript has no displacement figure (its figures come from '[READY] (TMLR) LAUNDROMAT - P6/floats/make_floats.py').
"""Figure 3 -- identity score vs embedding displacement, over the eleven true
derivatives. Regenerates the fit reported in Sec. pricing directly from the
frozen sweep files, so figure and text cannot drift."""
import json, pathlib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

plt.rcParams.update({'font.size':7,'font.family':'serif'})
ROOT=pathlib.Path(_ROOT)
# excluded: two from-scratch models (N-FAM, not derivatives) and Bio_ClinicalBERT,
# whose documented parent is bert-base-CASED via BioBERT, not the uncased model compared.
EXCL={"nlpaueb/legal-bert-base-uncased",
      "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
      "emilyalsentzer/Bio_ClinicalBERT"}
pts=[]
for f in ("M5/sweep_bert.jsonl","M5/corpus_distance.jsonl"):
    for line in (ROOT/f).read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line); d=r.get("displacement",{}); s=r.get("scores",{})
        if d.get("embedding") is not None and s.get("identity_score") is not None \
           and r["child"] not in EXCL:
            pts.append((d["embedding"], s["identity_score"]))
x=np.array([p[0] for p in pts]); y=np.array([p[1] for p in pts])
b,a=np.polyfit(x,y,1); pred=a+b*x
r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
xmax=x.max()
fig,ax=plt.subplots(figsize=(3.4,2.2))
ax.axvspan(xmax,1.35,color='0.92',zorder=0)
ax.text((xmax+1.35)/2,0.98,"extrapolated",fontsize=5.6,ha='center',color='0.45')
xs=np.linspace(0,1.35,50)
ax.plot(xs,a+b*xs,color='#1f4e79',lw=1.0,zorder=2)
ax.scatter(x,y,s=16,c='#1f4e79',zorder=3,lw=0)
for t,lab,ls in ((0.65,'0.65 weak-match line',':'),(0.5298,'0.5298 evasion bar','--')):
    ax.axhline(t,color='0.4',lw=0.7,ls=ls,zorder=1)
    ax.text(1.34,t+0.012,lab,fontsize=5.4,ha='right',color='0.4')
    xc=(t-a)/b
    if xc<=1.35: ax.plot([xc],[t],marker='v',ms=3.5,color='#c00000',zorder=4)
ax.set_xlabel(r"embedding displacement $d_{\mathrm{emb}}$",fontsize=7)
ax.set_ylabel(r"identity score $\sigma_{\mathrm{id}}$",fontsize=7)
ax.set_xlim(-0.03,1.35); ax.set_ylim(0.45,1.03)
ax.text(0.03,0.50,rf"$\sigma_{{\mathrm{{id}}}}={a:.4f}{b:+.4f}\,d$,  $R^2={r2:.3f}$,  $n={len(x)}$",fontsize=5.8)
for sp in ('top','right'): ax.spines[sp].set_visible(False)
ax.tick_params(length=2,labelsize=6)
fig.tight_layout(pad=0.15)
fig.savefig(ROOT/"[READY] (S&P 2027) LAUNDROMAT - P6"/"figures"/"fig3_displacement.pdf",dpi=300)
print(f"n={len(x)} fit sigma={a:.4f}{b:+.4f}d R2={r2:.4f} last_obs={xmax:.4f} "
      f"cross065={(0.65-a)/b:.4f} cross05298={(0.5298-a)/b:.4f}")
