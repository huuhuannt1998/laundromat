"""E9b -- freeze every recalibration number the manuscript quotes.

Section V-F quotes six numbers (LOO logistic 0.622, LOO non-negative 0.606, family-disjoint
vendor 0.688 and refit 0.125, best single signal 0.778, nlf 0.222) that e9_recalibrate.py
prints to stdout but did not write to e9_recalibration.json, which holds only the
leave-one-FAMILY-out means. The 2026-09-08 review flagged them as untraceable to a frozen
artifact. This script re-runs the identical computation (same inputs, same estimators; the
fits are deterministic) and writes all of them to M7/e9b_recalibration_frozen.json.
e9_recalibrate.py and e9_recalibration.json are left untouched.

It also records a one-label sensitivity arm. The Bio_ClinicalBERT row is D-CP in the
manifest, but its own evidence quote and model card place it in the bert-base-CASED lineage
(initialised from BioBERT-Base v1.0; vocabulary 28,996 cased entries), so as a child of
bert-base-UNCASED it is not a derivative. The manuscript's numbers keep the manifest labels;
the arm records what relabelling that one pair does.
"""
import json, glob, pathlib
import numpy as np
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))
R = pathlib.Path(_ROOT)
SIG = ("eas", "wvc", "end", "lep", "nlf")
VENDOR = np.array([0.36, 0.21, 0.19, 0.16, 0.08])          # src/provenancekit/config/constants.py
OUT = R/"M7"/"e9b_recalibration_frozen.json"

# ---------- gather signal vectors (identical to e9_recalibrate.py) ----------
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

man = json.loads((R/"M6"/"gold_lineage_manifest.json").read_text())
LBL, FAM = {}, {}
for r in man["rows"]:
    if r["analysis_set"] == "excluded": continue
    LBL[(r["parent"], r["child"])] = 1 if r["relationship_class"].startswith("D-") else 0
    FAM[(r["parent"], r["child"])] = r["parent"].split("/")[-1]
for a in ("70m","160m","410m","1b","1.4b"):
    for base in (f"EleutherAI/pythia-{a}", f"EleutherAI/pythia-{a}-deduped"):
        for suf in ("-deduped","-v0"):
            k=(base, f"{base}{suf}"); LBL[k]=0; FAM[k]="pythia"
        for suf in ("-deduped","-v0"):
            k=(f"EleutherAI/pythia-{a}", f"EleutherAI/pythia-{a}{suf}"); LBL[k]=0; FAM[k]="pythia"
for k in [("google-bert/bert-base-uncased","nlpaueb/legal-bert-base-uncased"),
          ("google-bert/bert-base-uncased","microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")]:
    LBL[k]=0; FAM[k]="bert-base-uncased"

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
    from scipy.optimize import nnls
    w,_ = nnls(Xd, yd.astype(float))
    return w/w.sum() if w.sum() > 0 else np.ones(Xd.shape[1])/Xd.shape[1]

def run(labels):
    data=[]
    for k, r in recs.items():
        if k not in labels: continue
        r=dict(r); r["y"]=labels[k]; r["family"]=FAM.get(k, r["parent"].split("/")[-1]); data.append(r)
    comp=[r for r in data if all(r[s] is not None for s in SIG)]
    X=np.array([[r[s] for s in SIG] for r in comp]); y=np.array([r["y"] for r in comp])
    fam=np.array([r["family"] for r in comp]); fams=sorted(set(fam))
    out=dict(n=len(comp), pos=int(y.sum()), neg=int((1-y).sum()), families=fams,
             pairs=[dict(parent=r["parent"], child=r["child"], y=int(r["y"]), family=r["family"])
                    for r in comp])
    # leave-one-family-out means (what e9_recalibration.json already holds)
    agg={k:[] for k in ("vendor","logistic","nonneg","best1")}
    for f in fams:
        te=fam==f; tr=~te
        if y[te].sum()==0 or (1-y[te]).sum()==0: continue
        w,b=fit_logistic(X[tr],y[tr]); wn=fit_nnls(X[tr],y[tr])
        best=max(range(5), key=lambda j: auc(X[tr][:,j], y[tr]))
        for k,v in zip(("vendor","logistic","nonneg","best1"),
                       (auc(X[te]@VENDOR,y[te]), auc(X[te]@w+b,y[te]), auc(X[te]@wn,y[te]), auc(X[te][:,best],y[te]))):
            agg[k].append(v)
    out["leave_one_family_out_mean"]={k: float(np.mean(v)) for k,v in agg.items() if v}
    # leave-one-pair-out (the numbers section V-F quotes)
    sc_log=np.zeros(len(y)); sc_nn=np.zeros(len(y))
    for i in range(len(y)):
        tr=np.ones(len(y),bool); tr[i]=False
        w_,b_=fit_logistic(X[tr],y[tr]); sc_log[i]=X[i]@w_+b_
        wn_=fit_nnls(X[tr],y[tr]);       sc_nn[i]=X[i]@wn_
    out["auc_vendor_coefficients"]=auc(X@VENDOR,y)
    out["auc_loo_logistic"]=auc(sc_log,y)
    out["auc_loo_nonneg"]=auc(sc_nn,y)
    # named family-disjoint split: train BERT+RoBERTa, test the rest
    TRAIN={"bert-base-uncased","roberta-base"}
    tr=np.array([f in TRAIN for f in fam]); te=~tr
    fd=dict(train_families=sorted(TRAIN), test_families=sorted(set(fam[te])),
            n_train=int(tr.sum()), train_pos=int(y[tr].sum()), train_neg=int((1-y[tr]).sum()),
            n_test=int(te.sum()), test_pos=int(y[te].sum()), test_neg=int((1-y[te]).sum()))
    if y[te].sum() and (1-y[te]).sum():
        w2,b2=fit_logistic(X[tr],y[tr]); wn2=fit_nnls(X[tr],y[tr])
        fd.update(auc_vendor=auc(X[te]@VENDOR,y[te]), auc_logistic=auc(X[te]@w2+b2,y[te]),
                  auc_nonneg=auc(X[te]@wn2,y[te]), nonneg_weights=[float(v) for v in wn2])
    out["family_disjoint_split"]=fd
    # in-sample ceiling and single signals
    w,b=fit_logistic(X,y); wn=fit_nnls(X,y)
    out["auc_insample_logistic"]=auc(X@w+b,y); out["auc_insample_nonneg"]=auc(X@wn,y)
    out["insample_logistic_weights"]=[float(v) for v in w]; out["insample_nonneg_weights"]=[float(v) for v in wn]
    single={s: auc(X[:,j],y) for j,s in enumerate(SIG)}
    out["auc_single_signal"]=single
    out["best_single_signal"]=max(single, key=single.get)
    out["auc_best_single_signal"]=single[out["best_single_signal"]]
    return out

main=run(LBL)
BIO=("google-bert/bert-base-uncased","emilyalsentzer/Bio_ClinicalBERT")
alt=dict(LBL); alt[BIO]=0
sens=run(alt)
res=dict(note="Manuscript numbers are 'manifest_labels'. 'bio_clinicalbert_relabelled' relabels one pair "
              "(bert-base-uncased -> Bio_ClinicalBERT) from derivative to non-derivative on card evidence; "
              "reported as a sensitivity arm only.",
         signals=list(SIG), vendor_weights=[float(v) for v in VENDOR],
         manifest_labels=main, bio_clinicalbert_relabelled=sens)
OUT.write_text(json.dumps(res, indent=1))
for name,o in (("manifest labels",main),("Bio_ClinicalBERT relabelled",sens)):
    print(f"\n=== {name}: n={o['n']} ({o['pos']} pos / {o['neg']} neg) ===")
    print(f"  vendor coefficients      AUC {o['auc_vendor_coefficients']:.4f}")
    print(f"  logistic, LOO            AUC {o['auc_loo_logistic']:.4f}")
    print(f"  non-negative, LOO        AUC {o['auc_loo_nonneg']:.4f}")
    print(f"  in-sample logistic       AUC {o['auc_insample_logistic']:.4f}")
    fd=o['family_disjoint_split']
    print(f"  family-disjoint          vendor {fd.get('auc_vendor',float('nan')):.4f}  "
          f"logistic {fd.get('auc_logistic',float('nan')):.4f}  nonneg {fd.get('auc_nonneg',float('nan')):.4f}  "
          f"(test {fd['n_test']}: {fd['test_pos']} pos / {fd['test_neg']} neg)")
    print(f"  best single signal       {o['best_single_signal']} AUC {o['auc_best_single_signal']:.4f}")
    print("  single signals           " + "  ".join(f"{k} {v:.4f}" for k,v in o['auc_single_signal'].items()))
print(f"\nwrote {OUT.relative_to(R)}")
