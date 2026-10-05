"""Figure 4 -- the design rule, measured. Alignment computed on MLP input
projections vs on the QK circuit, under the adaptive attacks of M4."""
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
qk=json.loads((ROOT/"M4"/"adaptive_qk.json").read_text())
def get(rows,key):
    out=[]
    for r in rows:
        d=r["defence"]
        out.append((r["variant"], d.get(key)))
    return out
labels=["control\n$s=0$","$s=0.1$","$s=0.3$"]
mlp=[qk[0]["defence"]["LAP_on_MLP"], qk[2]["defence"]["LAP_on_MLP"], qk[1]["defence"]["LAP_on_MLP"]]
attn=[qk[0]["defence"]["LAP_on_ATTN"], qk[2]["defence"]["LAP_on_ATTN"], qk[1]["defence"]["LAP_on_ATTN"]]
xs=np.arange(3)
fig,ax=plt.subplots(figsize=(3.4,2.05))
ax.plot(xs,mlp,marker='o',ms=4,lw=1.1,color='#1f4e79',label=r"align on MLP $W_{\mathrm{in}}$ (permutation-only)")
ax.plot(xs,attn,marker='s',ms=4,lw=1.1,color='#c00000',label=r"align on $W_QW_K^{\top}$ (bilinear freedom)")
for x,v in zip(xs,mlp):  ax.annotate(f"{v:.4f}",(x,v),textcoords="offset points",xytext=(0,5),fontsize=5.4,ha='center',color='#1f4e79')
for x,v in zip(xs,attn): ax.annotate(f"{v:.4f}",(x,v),textcoords="offset points",xytext=(0,-10),fontsize=5.4,ha='center',color='#c00000')
ax.axhline(0.822,color='0.45',lw=0.7,ls=':',zorder=1)
ax.text(2.02,0.836,"negative ceiling",fontsize=5.4,ha='right',color='0.45')
ax.set_xticks(xs); ax.set_xticklabels(labels,fontsize=6)
ax.set_xlabel("adaptive QK-invariance strength",fontsize=7)
ax.set_ylabel("alignment score",fontsize=7)
ax.set_ylim(-0.02,1.12); ax.set_xlim(-0.25,2.25)
ax.legend(fontsize=5.4,loc='center left',frameon=False)
for sp in ('top','right'): ax.spines[sp].set_visible(False)
ax.tick_params(length=2,labelsize=6)
fig.tight_layout(pad=0.15)
fig.savefig(ROOT/"manuscripts/laundromat/figures/fig4_symmetry.pdf",dpi=300)
print("MLP:",mlp); print("ATTN:",attn)
