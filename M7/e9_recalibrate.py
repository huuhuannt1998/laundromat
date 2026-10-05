"""E9 -- recalibrate the combination, with the five signals frozen.

The reviewer question: if the same signals were recalibrated using CORRECT labels --
in particular with independently-trained same-recipe models as true negatives -- would
the failure disappear? If yes, the signals suffice and the vendor's benchmark design
caused it. If no, the signals themselves lack lineage specificity. Either answer
sharpens the paper, so this is run to find out rather than to confirm.

Only the combination is refit. No signal is recomputed, no model is downloaded.
Splits are stratified by PARENT FAMILY so the test split contains families never seen
during fitting.
"""
import json, glob, pathlib, itertools, math
import numpy as np

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

R = pathlib.Path(_ROOT)
SIG = ("eas", "wvc", "end", "lep", "nlf")
VENDOR = np.array([0.36, 0.21, 0.19, 0.16, 0.08])          # src/provenancekit/config/constants.py

# ---------- gather signal vectors ----------
recs = {}
def add(par, ch, sg, sc, src):
    if not sg or sg.get("eas") is None or not par or not ch: return
    recs[(par, ch)] = dict(parent=par, child=ch, src=src,
                           **{k: sg.get(k) for k in SIG},
                           sigma=(sc or {}).get("identity_score"),
                           tier=(sc or {}).get("mfi_tier"))
for f, pk, ck, holder in [("M1/results/tier3_positives.jsonl","parent","child",None),
                          ("M1/results/pairs.jsonl","model_a","model_b","result"),
                          ("M1/results/pairs2.jsonl","model_a","model_b","result"),
                          ("M5/sweep_bert.jsonl","parent","child",None),
                          ("M5/corpus_distance.jsonl","parent","child",None)]:
    p = R/f
    if not p.exists(): continue
    for l in p.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l); h = r.get(holder) if holder else r
        add(r.get(pk), r.get(ck), (h or {}).get("signals"), (h or {}).get("scores"), f)
for f in [str(R/"M4"/"p_tilde_arm.jsonl"), str(R/"M4"/"power_fix.jsonl")] + \
         glob.glob(str(R/"M1"/"tier3_topup*.jsonl")) + glob.glob(str(R/"M1"/"results"/"tier3_topup*.jsonl")):
    if not pathlib.Path(f).exists(): continue
    for l in open(f):
        if not l.strip(): continue
        r = json.loads(l)
        add(r.get("parent") or r.get("a"), r.get("child") or r.get("b"),
            r.get("signals"), r.get("scores"), f)

# ---------- labels ----------
man = json.loads((R/"M6"/"gold_lineage_manifest.json").read_text())
LBL, FAM = {}, {}
for r in man["rows"]:
    if r["analysis_set"] == "excluded": continue
    LBL[(r["parent"], r["child"])] = 1 if r["relationship_class"].startswith("D-") else 0
    FAM[(r["parent"], r["child"])] = r["parent"].split("/")[-1]
# pythia same-recipe independents: true NEGATIVES, the class the vendor benchmark mislabels
for a in ("70m","160m","410m","1b","1.4b"):
    for base in (f"EleutherAI/pythia-{a}", f"EleutherAI/pythia-{a}-deduped"):
        for suf in ("-deduped","-v0"):
            k=(base, f"{base}{suf}"); LBL[k]=0; FAM[k]="pythia"
        for suf in ("-deduped","-v0"):
            k=(f"EleutherAI/pythia-{a}", f"EleutherAI/pythia-{a}{suf}"); LBL[k]=0; FAM[k]="pythia"
for k in [("google-bert/bert-base-uncased","nlpaueb/legal-bert-base-uncased"),
          ("google-bert/bert-base-uncased","microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")]:
    LBL[k]=0; FAM[k]="bert-base-uncased"

data=[]
for k, r in recs.items():
    if k not in LBL: continue
    r=dict(r); r["y"]=LBL[k]; r["family"]=FAM.get(k, r["parent"].split("/")[-1]); data.append(r)
comp=[r for r in data if all(r[s] is not None for s in SIG)]
print(f"labelled pairs {len(data)}  (complete 5-signal: {len(comp)})")
print(f"  positives {sum(r['y'] for r in comp)}   negatives {sum(1-r['y'] for r in comp)}")
fams=sorted({r['family'] for r in comp})
print(f"  families: {fams}")

X=np.array([[r[s] for s in SIG] for r in comp]); y=np.array([r["y"] for r in comp])
fam=np.array([r["family"] for r in comp])

def auc(sc, yy):
    p=sc[yy==1]; n=sc[yy==0]
    if len(p)==0 or len(n)==0: return float("nan")
    return sum((1.0 if a>b else .5 if a==b else 0.) for a in p for b in n)/(len(p)*len(n))

def fit_logistic(Xd, yd, iters=4000, lr=0.4):
    w=np.zeros(Xd.shape[1]); b=0.0
    for _ in range(iters):
        z=Xd@w+b; p=1/(1+np.exp(-z)); g=p-yd
        w-=lr*(Xd.T@g)/len(yd) + lr*1e-3*w
        b-=lr*g.mean()
    return w,b

def fit_nnls(Xd, yd):
    """Non-negative weights summing to 1 (the vendor's own functional form), fitted by
    non-negative least squares on the label then renormalised."""
    from scipy.optimize import nnls
    w,_ = nnls(Xd, yd.astype(float))
    return w/w.sum() if w.sum() > 0 else np.ones(Xd.shape[1])/Xd.shape[1]

print("\n=== family-disjoint leave-one-family-out ===")
print(f"{'held-out family':<20}{'n':>4}{'vendor':>9}{'logistic':>10}{'nonneg':>9}{'best-1':>9}")
agg={k:[] for k in ("vendor","logistic","nonneg","best1")}
for f in fams:
    te=fam==f; tr=~te
    if y[te].sum()==0 or (1-y[te]).sum()==0:   # need both classes to score AUC
        print(f"{f:<20}{te.sum():>4}   (single-class held-out split, skipped)"); continue
    a_v=auc(X[te]@VENDOR, y[te])
    w,b=fit_logistic(X[tr],y[tr]); a_l=auc(X[te]@w+b, y[te])
    wn=fit_nnls(X[tr],y[tr]);      a_n=auc(X[te]@wn, y[te])
    best=max(range(5), key=lambda j: auc(X[tr][:,j], y[tr])); a_b=auc(X[te][:,best], y[te])
    for k,v in zip(("vendor","logistic","nonneg","best1"),(a_v,a_l,a_n,a_b)): agg[k].append(v)
    print(f"{f:<20}{te.sum():>4}{a_v:>9.3f}{a_l:>10.3f}{a_n:>9.3f}{a_b:>9.3f}  (best single: {SIG[best]})")
print(f"{'MEAN':<20}{'':>4}" + "".join(f"{np.mean(agg[k]):>9.3f}" if k!='logistic' else f"{np.mean(agg[k]):>10.3f}"
                                       for k in ("vendor","logistic","nonneg","best1")))

# ---------- honest out-of-sample: leave-one-pair-out ----------
print("\n=== leave-one-pair-out cross-validation (honest out-of-sample) ===")
sc_log=np.zeros(len(y)); sc_nn=np.zeros(len(y))
for i in range(len(y)):
    tr=np.ones(len(y),bool); tr[i]=False
    w_,b_=fit_logistic(X[tr],y[tr]); sc_log[i]=X[i]@w_+b_
    wn_=fit_nnls(X[tr],y[tr]);       sc_nn[i]=X[i]@wn_
print(f"  vendor coefficients (no fitting) AUC {auc(X@VENDOR,y):.4f}")
print(f"  logistic, LOO                    AUC {auc(sc_log,y):.4f}")
print(f"  non-negative linear, LOO         AUC {auc(sc_nn,y):.4f}")

# ---------- named family-disjoint split (both classes each side) ----------
TRAIN={"bert-base-uncased","roberta-base"}
tr=np.array([f in TRAIN for f in fam]); te=~tr
print(f"\n=== family-disjoint split: train {sorted(TRAIN)} -> test {sorted(set(fam[te]))} ===")
print(f"  train {tr.sum()} ({y[tr].sum()} pos, {(1-y[tr]).sum()} neg)   "
      f"test {te.sum()} ({y[te].sum()} pos, {(1-y[te]).sum()} neg)")
if y[te].sum() and (1-y[te]).sum():
    w2,b2=fit_logistic(X[tr],y[tr]); wn2=fit_nnls(X[tr],y[tr])
    print(f"  vendor      AUC {auc(X[te]@VENDOR,y[te]):.4f}")
    print(f"  logistic    AUC {auc(X[te]@w2+b2,y[te]):.4f}")
    print(f"  non-negative AUC {auc(X[te]@wn2,y[te]):.4f}   w={np.round(wn2,3)}")

print("\n=== full-corpus refit (in-sample ceiling: what the signals can express at best) ===")
print(f"  vendor coefficients      AUC {auc(X@VENDOR,y):.4f}")
w,b=fit_logistic(X,y);  print(f"  logistic (in-sample)     AUC {auc(X@w+b,y):.4f}   w={np.round(w,3)}")
wn=fit_nnls(X,y);       print(f"  non-negative (in-sample) AUC {auc(X@wn,y):.4f}   w={np.round(wn,3)}")
for j,s in enumerate(SIG): print(f"    single signal {s:<4}     AUC {auc(X[:,j],y):.4f}")
json.dump(dict(n=len(comp), pos=int(y.sum()), neg=int((1-y).sum()), families=fams,
               loo_mean={k: float(np.mean(v)) for k,v in agg.items() if v},
               full_vendor=auc(X@VENDOR,y), full_logistic=auc(X@w+b,y),
               full_nonneg=auc(X@wn,y), nonneg_weights=[float(v) for v in wn],
               vendor_weights=list(VENDOR), signals=list(SIG)),
          open(R/"M7"/"e9_recalibration.json","w"), indent=1)
print("\nwrote M7/e9_recalibration.json")
