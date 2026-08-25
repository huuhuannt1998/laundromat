"""Predeclared ablations (design 11.5) needing NO new model runs -- pure rescoring of collected
signal vectors. Four of the nine were in this category and I had skipped them."""
import json,pathlib
import numpy as np

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

R=pathlib.Path(_ROOT)
W={"eas":.36,"nlf":.08,"lep":.16,"end":.19,"wvc":.21}
rows=[]
for f in ["M1/results/pairs.jsonl","M1/results/pairs2.jsonl"]:
    for l in (R/f).read_text().splitlines():
        d=json.loads(l); s=d["result"].get("signals"); sc=d["result"].get("scores")
        if s and sc: rows.append({"gt":d["ground_truth"],"sig":s,"tier":sc["mfi_tier"],
                                  "id":sc["identity_score"],
                                  "pair":f"{d['model_a'].split('/')[-1]}|{d['model_b'].split('/')[-1]}"})

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

for f in ["M4/p_tilde_arm.jsonl","M4/family_generality.jsonl"]:
    p=R/f
    if not p.exists(): continue
    for l in [x for x in p.read_text().splitlines() if x.strip().startswith("{")]:
        d=json.loads(l)
        if d.get("class")=="P= identity": continue
        s=d.get("signals"); sc=d.get("scores")
        if not s or not sc: continue
        rows.append({"gt":"unrelated","sig":s,"tier":sc["mfi_tier"],"id":sc["identity_score"],
                     "pair":d.get("a","").split("/")[-1]+"|"+d.get("b","").split("/")[-1]})
def score(sig,w,keys):
    num=den=0.0
    for k in keys:
        v=sig.get(k)
        if v is None: continue
        num+=w[k]*v; den+=w[k]
    return num/den if den else float("nan")
def auc(pos,neg):
    p=np.array(pos,float); n=np.array(neg,float)
    if not len(p) or not len(n): return float("nan")
    return float(((p[:,None]>n[None,:]).sum()+0.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
V={"MPK as shipped (identity)":(W,list(W)),
   "ABLATION flattened weights (1/5)":({k:0.2 for k in W},list(W)),
   "ABLATION WVC only (positional)":(W,["wvc"]),
   "ABLATION structural only (no WVC)":(W,["eas","nlf","lep","end"]),
   "ABLATION EAS only":(W,["eas"])}
npos=sum(1 for r in rows if r["gt"]=="related"); nneg=sum(1 for r in rows if r["gt"]=="unrelated")
print("="*98); print(f"PREDECLARED ABLATIONS — rescoring collected signals. corpus {npos} derived / {nneg} not")
print("="*98)
print(f"{'variant':<38}{'AUC':>8}{'med(derived)':>14}{'med(not)':>12}")
print("-"*98)
for name,(w,keys) in V.items():
    pos=[score(r["sig"],w,keys) for r in rows if r["gt"]=="related"]
    neg=[score(r["sig"],w,keys) for r in rows if r["gt"]=="unrelated"]
    pos=[x for x in pos if x==x]; neg=[x for x in neg if x==x]
    print(f"{name:<38}{auc(pos,neg):>8.3f}{np.median(pos):>14.4f}{np.median(neg):>12.4f}")
print("\n"+"="*98); print("ABLATION: gate forced to tier 3 (ignore MFI, use identity score)"); print("="*98)
def band(x): return "Confirmed" if x==1.0 else "High-Conf" if x>0.75 else "Weak" if x>0.65 else "NotMatched"
gated=[r for r in rows if r["tier"]<=2]
print(f"{'pair':<48}{'gated':>11}{'identity':>10}{'ungated':>12}  ground truth")
for r in sorted(gated,key=lambda r:r["gt"]):
    print(f"{r['pair'][:47]:<48}{'Confirmed':>11}{r['id']:>10.4f}{band(r['id']):>12}  {r['gt']}")
fp=[r for r in gated if r["gt"]=="unrelated"]; fixed=[r for r in fp if r["id"]<=0.75]
print(f"\n-> gate creates {len(fp)} false positives; removing it fixes {len(fixed)}, "
      f"{len(fp)-len(fixed)} survive on weight evidence alone")
