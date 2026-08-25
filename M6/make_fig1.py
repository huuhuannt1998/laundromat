"""Figure 1 -- the decision path. Schematic; the numbers in the caption come from
Table 'fp'/'fam' and Sec. soundness, not from this script."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pathlib
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
plt.rcParams.update({'font.size':7,'font.family':'serif'})
OUT=_ROOT/"manuscripts"/"laundromat"/"figures"/"fig1_pipeline.pdf"
fig,ax=plt.subplots(figsize=(3.4,2.35)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,7)
def box(x,y,w,h,txt,fc,ec='0.25',fs=6.2,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",fc=fc,ec=ec,lw=0.7,zorder=3))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,zorder=4,
            fontweight=('bold' if bold else 'normal'))
def arr(x1,y1,x2,y2,style='-|>',c='0.3',lw=0.8,ls='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,color=c,lw=lw,
                                 linestyle=ls,mutation_scale=7,zorder=2))
box(0.1,5.4,3.0,1.0,"config.json\n(publisher-written)",'#f6d4d4',ec='#c00000')
ax.text(1.6,4.98,"untrusted input",fontsize=5.4,ha='center',color='#c00000')
box(4.0,5.4,2.3,1.0,"MFI tier\ntest",'#e8e8e8'); arr(3.15,5.9,3.95,5.9)
box(7.0,6.0,2.9,0.85,"tier 1: pipeline := 1.0",'#f6d4d4',ec='#c00000',bold=True)
box(7.0,5.0,2.9,0.85,"tier 2: cap 0.9",'#fae6cc')
box(7.0,3.5,2.9,0.85,"tier 3: use $\\sigma_{id}$",'#dce9f5')
arr(6.35,6.1,6.95,6.4); arr(6.35,5.9,6.95,5.42); arr(6.35,5.6,6.95,3.95)
box(0.1,2.2,3.0,1.0,"5 weight signals\n$\\rightarrow \\sigma_{id}$",'#dce9f5')
arr(3.15,2.7,6.95,3.75)
ax.text(4.6,2.72,"consulted only at tier 3",fontsize=5.4,color='#1f4e79')
box(4.0,0.5,3.0,0.9,"verdict",'#e8e8e8')
arr(8.45,5.98,8.45,1.45,c='#c00000',lw=1.0); arr(8.45,3.48,8.45,1.45)
arr(6.95,0.95,7.05,0.95,style='-')
ax.text(9.6,3.0,"11 of 14\npooled\ncomparisons",fontsize=5.4,color='#c00000',ha='center')
box(0.1,0.5,3.0,0.9,"no verdict",'#f6d4d4',ec='#c00000')
arr(1.6,5.35,1.6,1.45,c='#c00000',lw=0.9,ls=(0,(2.5,1.5)))
ax.text(0.28,3.3,"key absent\n$\\rightarrow$ crash",fontsize=5.4,color='#c00000')
fig.tight_layout(pad=0.1); fig.savefig(OUT,dpi=300); print("wrote",OUT)
