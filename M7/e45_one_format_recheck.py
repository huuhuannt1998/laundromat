"""E45 -- the E9 recalibration, the E10 threshold sweep and the E6 gate counterfactual, re-run on
ONE-FORMAT signals (safetensors tensor order on both sides), beside the as-served numbers.

Why (CORRECTIONS.md #34, E35-E40): the verifier's NLF and WVC signals concatenate tensors in file
order, so identical weights score differently as .bin and as safetensors. E9/E10/E6 were computed
on as-served vectors, mixing both orders. The PI approved (2026-10-02, Core jrn_01M3YCC6VCBKPTMQYB8ME94KHA)
running the one-format re-check before claims are drafted.

Two one-format variants (set E45_VARIANT):
  canonical (default; M7/e45_one_format_recheck.json): both sides re-written by safetensors'
      save_file, so the verifier's loader returns tensors in the same name order on both sides
      (M9/e44e_canonical_order.jsonl). E44c/E44d showed this is what "one fixed format" has to mean:
      the loader's tensor order is set by the file's writer and the load path, not by the format.
  child_resaved (M7/e45_one_format_recheck_child_resaved.json): the convention of E36/E39/E40/E41/E44
      (re-save the child's .bin, read the parent as served; pairs served as safetensors on both sides
      taken unchanged). Kept for comparison; E44d finds several of its pairs order-mismatched, and
      each basis line carries that audit note.
  As-served inputs are the files M7/e9b_recalibrate.py reads, M6/pooled_lineage_corpus_31row.json
  (E6/E10), M7/e5_same_recipe.jsonl and M9/e23_scale.jsonl. A pair with no reading in the chosen
  variant is reported as missing, never filled in.

Estimators. E9: the exact code of M7/e9b_recalibrate.py (copied, not imported, because importing
it would overwrite its frozen output). The as-served run must reproduce M7/e9b_recalibration_frozen.json
to the last digit; the script asserts it. E6 and E10 had no committed script (they were computed
from /tmp/corpus.json on 2026-08-24, M6/mock_review_experiments.md); the rules below are
reconstructions, and the script asserts that they reproduce the published as-served tables
(E6: 21/6/1/3, 21/5/2/3, 21/5/2/3, 21/5/1/3 + 1 abstention; E10: no threshold gives recall 1 with
same-recipe FPR 0, and at the first threshold with FPR 0, t = 0.80, recall is 10/24).

Writes NEW files only (see the variants above). Nothing existing is modified.
"""
import json, glob, pathlib, hashlib, os
import numpy as np
_ROOT = pathlib.Path(os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in pathlib.Path(__file__).resolve().parents if (p / "MANIFEST-dataintegrity.txt").exists())))
R = _ROOT
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))
VARIANT = os.environ.get("E45_VARIANT", "canonical")   # canonical | child_resaved
OUT = R/"M7"/("e45_one_format_recheck.json" if VARIANT == "canonical" else "e45_one_format_recheck_child_resaved.json")
SIG = ("eas", "wvc", "end", "lep", "nlf")
VENDOR = np.array([0.36, 0.21, 0.19, 0.16, 0.08])
def sha16(p): return hashlib.sha256((R/p).read_bytes()).hexdigest()[:16]

# ---------------------------------------------------------------- as-served vectors (as e9b) ---
recs = {}
def add(par, ch, sg, sc, src):
    if not sg or sg.get("eas") is None or not par or not ch: return
    recs[(par, ch)] = dict(parent=par, child=ch, src=src, **{k: sg.get(k) for k in SIG},
                           sigma=(sc or {}).get("identity_score"), tier=(sc or {}).get("mfi_tier"))
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
        add(r.get("parent") or r.get("a"), r.get("child") or r.get("b"), r.get("signals"), r.get("scores"), f)

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
BIO = ("google-bert/bert-base-uncased","emilyalsentzer/Bio_ClinicalBERT")

# ---------------------------------------------------------------- one-format re-reads -----------
ONE, ONE_SRC = {}, {}
def take(f, pk, ck, ok):
    for l in (R/f).read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        if not ok(r) or not r.get("scores"): continue
        key = (r[pk], r[ck])
        if key in ONE: continue                     # first source wins (order below)
        ONE[key] = dict(**{k: r["signals"].get(k) for k in SIG}, sigma=r["scores"]["identity_score"],
                        tier=r["scores"]["mfi_tier"], verdict=r["scores"]["provenance_decision"],
                        pipeline=r["scores"]["pipeline_score"])
        ONE_SRC[key] = f + " [" + (r.get("format") or r.get("arm")) + "]"
take("M9/e36_format_sensitivity.jsonl", "parent", "child", lambda r: r["format"] == "safetensors-from-bin")
take("M9/e39_format_bert_sweep.jsonl", "parent", "child", lambda r: r["format"] == "safetensors-from-bin")
take("M9/e40_format_roberta_sweep.jsonl", "parent", "child", lambda r: r["format"] == "safetensors-from-bin")
take("M9/e35_serialisation.jsonl", "parent", "child", lambda r: r["format"] == "safetensors")
take("M9/e44_one_format_remaining.jsonl", "parent", "child", lambda r: r.get("arm", "").startswith("one-format"))
E44B = R/"M9"/"e44b_one_format_same_recipe.jsonl"
if E44B.exists() and E44B.read_text().strip():
    # MultiBERTs only: the 6.9B arm of E44b re-saved one side in name order against a hub side that the
    # loader reads in module order (E44c), so it is order-mismatched, not one-format.
    take("M9/e44b_one_format_same_recipe.jsonl", "a", "b",
         lambda r: r.get("arm", "").startswith("one-format") and r.get("suite") == "multiberts")
E44C = R/"M9"/"e44c_6p9b_diagnostic.json"
if E44C.exists():
    diag = json.loads(E44C.read_text())
    if diag["pairs"].get("hub_6.9b | hub_6.9b-deduped", {}).get("same_norm_key_order"):
        p69 = next(json.loads(l) for l in (R/"M9"/"e23_scale.jsonl").read_text().splitlines()
                   if l.strip() and json.loads(l)["a"] == "EleutherAI/pythia-6.9b" and json.loads(l)["b"] == "EleutherAI/pythia-6.9b-deduped")
        ONE[("EleutherAI/pythia-6.9b", "EleutherAI/pythia-6.9b-deduped")] = dict(
            **{k: p69["signals"][k] for k in SIG}, sigma=p69["scores"]["identity_score"], tier=p69["scores"]["mfi_tier"],
            verdict=p69["scores"]["provenance_decision"], pipeline=p69["scores"]["pipeline_score"])
        ONE_SRC[("EleutherAI/pythia-6.9b", "EleutherAI/pythia-6.9b-deduped")] = (
            "M9/e23_scale.jsonl [as served; E44c: both sides load through AutoModel in the same module order]")
# load-order audit (E44d): a pair whose two sides load in different tensor orders has no one-order reading
AUDIT = {}
E44D = R/"M9"/"e44d_load_order_audit.json"
if E44D.exists():
    for a in json.loads(E44D.read_text())["pairs"]:
        AUDIT[(a["parent"], a["child"])] = a

# canonical order (E44e): both sides re-written by save_file -> same (name) order on both sides
CANON, CANON_SRC = {}, {}
E44E = R/"M9"/"e44e_canonical_order.jsonl"
if E44E.exists():
    for l in E44E.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        if not r.get("scores"): continue
        k = (r["parent"], r["child"])
        CANON[k] = dict(**{s_: r["signals"].get(s_) for s_ in SIG}, sigma=r["scores"]["identity_score"],
                        tier=r["scores"]["mfi_tier"], verdict=r["scores"]["provenance_decision"],
                        pipeline=r["scores"]["pipeline_score"])
        CANON_SRC[k] = "M9/e44e_canonical_order.jsonl [both sides re-written by save_file]"
ONE_MAP = CANON if VARIANT == "canonical" else ONE

def served_safetensors(mid):
    d = HUB / ("models--" + mid.replace("/", "--"))
    try: rev = (d/"refs"/"main").read_text().strip()
    except OSError: return None
    names = {p.name for p in (d/"snapshots"/rev).iterdir()}
    return ("model.safetensors" in names) or ("model.safetensors.index.json" in names)

def one_format(key, served):
    """Return (vector/scores dict, basis) or (None, reason)."""
    a = AUDIT.get(key)
    note = "" if a is None else f"; E44d order audit of this convention: norm {a['same_norm_order']}, within-layer {a['same_within_layer_order']}"
    if VARIANT == "canonical":
        if key in CANON: return CANON[key], CANON_SRC[key]
        return None, "MISSING: no canonical-order (E44e) reading"
    # child_resaved: the E36/E39/E40/E41/E44 convention (re-save the child, read the parent as served),
    # reported ungated but annotated with the E44d audit
    if key in ONE: return ONE[key], "re-read: " + ONE_SRC[key] + note
    pa, ch = key
    sp, sc = served_safetensors(pa), served_safetensors(ch)
    if sp and sc: return served, "as served; main serves safetensors for both (cache)" + note
    return None, f"MISSING: no re-read and served safetensors parent={sp} child={sc}"

# ---------------------------------------------------------------- E9 estimators (verbatim e9b) ---
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
def run(labels, vectors):
    data=[]
    for k, r in vectors.items():
        if k not in labels: continue
        r=dict(r); r["y"]=labels[k]; r["family"]=FAM.get(k, r["parent"].split("/")[-1]); data.append(r)
    comp=[r for r in data if all(r[s] is not None for s in SIG)]
    X=np.array([[r[s] for s in SIG] for r in comp]); y=np.array([r["y"] for r in comp])
    fam=np.array([r["family"] for r in comp]); fams=sorted(set(fam))
    out=dict(n=len(comp), pos=int(y.sum()), neg=int((1-y).sum()), families=fams)
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
    sc_log=np.zeros(len(y)); sc_nn=np.zeros(len(y))
    for i in range(len(y)):
        tr=np.ones(len(y),bool); tr[i]=False
        w_,b_=fit_logistic(X[tr],y[tr]); sc_log[i]=X[i]@w_+b_
        wn_=fit_nnls(X[tr],y[tr]);       sc_nn[i]=X[i]@wn_
    out["auc_vendor_coefficients"]=auc(X@VENDOR,y)
    out["auc_loo_logistic"]=auc(sc_log,y)
    out["auc_loo_nonneg"]=auc(sc_nn,y)
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
    w,b=fit_logistic(X,y); wn=fit_nnls(X,y)
    out["auc_insample_logistic"]=auc(X@w+b,y); out["auc_insample_nonneg"]=auc(X@wn,y)
    out["insample_logistic_weights"]=[float(v) for v in w]; out["insample_nonneg_weights"]=[float(v) for v in wn]
    single={s: auc(X[:,j],y) for j,s in enumerate(SIG)}
    out["auc_single_signal"]=single
    out["best_single_signal"]=max(single, key=single.get)
    out["auc_best_single_signal"]=single[out["best_single_signal"]]
    return out

alt = dict(LBL); alt[BIO] = 0
served_main = run(LBL, recs); served_bio = run(alt, recs)
frozen = json.loads((R/"M7"/"e9b_recalibration_frozen.json").read_text())
for mine, theirs in ((served_main, frozen["manifest_labels"]), (served_bio, frozen["bio_clinicalbert_relabelled"])):
    for k in ("n","pos","neg","auc_vendor_coefficients","auc_loo_logistic","auc_loo_nonneg","auc_insample_logistic",
              "auc_insample_nonneg","auc_single_signal","family_disjoint_split","leave_one_family_out_mean"):
        assert mine[k] == theirs[k], ("as-served E9 does not reproduce e9b", k, mine[k], theirs[k])

e9_pairs = [k for k in recs if k in LBL and all(recs[k][s] is not None for s in SIG)]
one_vec, basis, missing = {}, {}, []
for k in e9_pairs:
    v, why = one_format(k, recs[k])
    basis["|".join(k)] = why
    if v is None: missing.append(why); continue
    one_vec[k] = dict(parent=k[0], child=k[1], **{s: v[s] for s in SIG}, sigma=v["sigma"])
e9 = dict(pairs=len(e9_pairs), one_format_missing=missing, basis=basis,
          vectors={"|".join(k): dict(as_served={s: recs[k][s] for s in SIG} | {"sigma": recs[k]["sigma"]},
                                     one_format={s: one_vec[k][s] for s in SIG} | {"sigma": one_vec[k]["sigma"]}
                                     if k in one_vec else None, y_manifest=LBL[k]) for k in e9_pairs},
          as_served_reproduces_e9b_frozen=True,
          as_served=dict(manifest_labels=served_main, bio_clinicalbert_relabelled=served_bio),
          one_format=None if missing else dict(manifest_labels=run(LBL, one_vec), bio_clinicalbert_relabelled=run(alt, one_vec)))
# vendor-weighted sum check: one-format sigma equals the weighted sum of its signals (to 4 dp rounding)
e9["one_format_sigma_check_max_abs_dev"] = max(abs(float(np.array([one_vec[k][s] for s in SIG])@VENDOR) - one_vec[k]["sigma"])
                                              for k in one_vec) if one_vec else None

# ---------------------------------------------------------------- 31-row corpus: E6 and E10 -------
corpus = json.loads((R/"M6"/"pooled_lineage_corpus_31row.json").read_text())
def corpus_key(r):
    if r["source"] == "Table-I":
        size = r["child"].split("/")[0].replace("pythia-", "")
        return (f"EleutherAI/pythia-{size}", f"EleutherAI/pythia-{size}-deduped")
    return (r["parent"], r["child"])
rows = []
for r in corpus:
    k = corpus_key(r)
    served = dict(sigma=r["sigma_id"], tier=r["tier"], verdict=r["verdict"], pipeline=r["pipeline"])
    v, why = one_format(k, served)
    rows.append(dict(parent=k[0], child=k[1], rel=r["rel"], source=r["source"], served=served,
                     one_format=None if v is None else dict(sigma=v["sigma"], tier=v["tier"], verdict=v["verdict"],
                                                            pipeline=v["pipeline"]), basis=why))
SR = {"SAME_RECIPE_INDEPENDENT"}
NEG_E6 = {"nlpaueb/legal-bert-base-uncased", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"}
def lbl_original(row):      # the 24/7 split E6/E10 used (M6/mock_review_experiments.md)
    return 0 if (row["rel"] in SR or row["child"] in NEG_E6) else 1
def lbl_audited(row):       # CORRECTIONS #23/#25: BERT-Tiny DISPUTED (excluded); Bio_ClinicalBERT cross-lineage
    if row["child"] == "nreimers/BERT-Tiny_L-2_H-128_A-2": return None
    if row["child"] == "emilyalsentzer/Bio_ClinicalBERT": return 0
    return lbl_original(row)

def e6(rows, which, label):
    res = {}
    for pol in ("vendor", "weight_only", "conservative_and", "conflict_aware"):
        tp=fp=tn=fn=ab=0; detail=[]
        for r in rows:
            y = label(r)
            if y is None: continue
            s = r[which]
            ven = s["verdict"] not in ("Not Matched", None)
            wo = s["sigma"] is not None and round(s["sigma"]*10000) >= 6500
            pred = {"vendor": ven, "weight_only": wo, "conservative_and": ven and wo,
                    "conflict_aware": ven if ven == wo else None}[pol]
            if pred is None: ab += 1; detail.append((r["child"], "abstain")); continue
            if pred and y: tp+=1
            elif pred and not y: fp+=1; detail.append((r["child"], "FP"))
            elif not pred and y: fn+=1; detail.append((r["child"], "FN"))
            else: tn+=1
        res[pol] = dict(TP=tp, FP=fp, TN=tn, FN=fn, abstain=ab, errors=detail)
    return res
def e10(rows, which, label, negatives):
    pos = [r[which]["sigma"] for r in rows if label(r) == 1]
    neg = [s for s in negatives(rows, which)]
    sweep = []
    for ti in range(0, 101):
        rec = sum(1 for s in pos if round(s*10000) >= ti*100); fpr = sum(1 for s in neg if round(s*10000) >= ti*100)
        sweep.append(dict(t=ti/100, recall=rec/len(pos), recall_n=rec, sr_fpr=fpr/len(neg), sr_fp_n=fpr))
    both = [x["t"] for x in sweep if x["recall_n"] == len(pos) and x["sr_fp_n"] == 0]
    first0 = next(x for x in sweep if x["sr_fp_n"] == 0)
    return dict(n_pos=len(pos), n_same_recipe=len(neg), thresholds_with_recall1_and_fpr0=both,
                first_t_with_fpr0=first0["t"], recall_at_first_t_with_fpr0=first0["recall"],
                recall_n_at_first_t_with_fpr0=first0["recall_n"],
                max_same_recipe_sigma=max(neg), min_positive_sigma=min(pos), sweep=sweep)
def sr5(rows, which): return [r[which]["sigma"] for r in rows if r["rel"] in SR]

# extended same-recipe negatives: the twelve pairs (five Pythia 70m-1.4b, 6.9B, six MultiBERTs)
e5 = [json.loads(l) for l in (R/"M7"/"e5_same_recipe.jsonl").read_text().splitlines() if l.strip()]
e23 = [json.loads(l) for l in (R/"M9"/"e23_scale.jsonl").read_text().splitlines() if l.strip()]
p69 = next(r for r in e23 if r["a"] == "EleutherAI/pythia-6.9b" and r["b"] == "EleutherAI/pythia-6.9b-deduped")
twelve_served = sr5(rows, "served") + [p69["scores"]["identity_score"]] + [r["identity"] for r in e5]
twelve_one, twelve_missing = sr5(rows, "one_format") if all(r["one_format"] for r in rows if r["rel"] in SR) else [], []
for k in [("EleutherAI/pythia-6.9b", "EleutherAI/pythia-6.9b-deduped")] + [(r["a"], r["b"]) for r in e5]:
    if k in ONE_MAP: twelve_one.append(ONE_MAP[k]["sigma"])
    else: twelve_missing.append("|".join(k))

corpus_missing = [r["child"] + ": " + r["basis"] for r in rows if r["one_format"] is None]

# ---------------------------------------------------------------- E13/E41 regression, this variant ----
# same eleven derivatives and exclusions as M7/e13_regression.py / M7/e41_regression_format.py
EXCL = {"nlpaueb/legal-bert-base-uncased", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "emilyalsentzer/Bio_ClinicalBERT"}
reg_pts = []
for f in ("M5/sweep_bert.jsonl", "M5/corpus_distance.jsonl"):
    for line in (R/f).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line); dd = r.get("displacement", {}); sc_ = r.get("scores", {})
        if dd.get("embedding") is not None and sc_.get("identity_score") is not None and r["child"] not in EXCL:
            v, why = one_format((r["parent"], r["child"]), dict(sigma=sc_["identity_score"]))
            reg_pts.append(dict(child=r["child"], d_emb=dd["embedding"], served=sc_["identity_score"],
                                one_format=None if v is None else v["sigma"], basis=why))
def regfit(ys):
    x = np.array([p["d_emb"] for p in reg_pts]); y = np.array(ys)
    c = np.polyfit(x, y, 1); pred = np.polyval(c, x); r2 = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
    return dict(slope=float(c[0]), intercept=float(c[1]), r2=float(r2),
                d_at_065=float((0.65-c[1])/c[0]), d_at_05298=float((0.5298-c[1])/c[0]))
reg_missing = [p["child"] for p in reg_pts if p["one_format"] is None]
regression = dict(n=len(reg_pts), points=reg_pts, served=regfit([p["served"] for p in reg_pts]),
                  one_format=None if reg_missing else regfit([p["one_format"] for p in reg_pts]), missing=reg_missing)
out = dict(
    note=__doc__.split("\n\n")[0], variant=VARIANT,
    inputs={p: sha16(p) for p in ["M7/e9b_recalibration_frozen.json", "M6/gold_lineage_manifest.json",
            "M6/pooled_lineage_corpus_31row.json", "M9/e35_serialisation.jsonl", "M9/e36_format_sensitivity.jsonl",
            "M9/e39_format_bert_sweep.jsonl", "M9/e40_format_roberta_sweep.jsonl", "M9/e44_one_format_remaining.jsonl",
            "M7/e5_same_recipe.jsonl", "M9/e23_scale.jsonl"] + [p for p in ("M9/e44b_one_format_same_recipe.jsonl", "M9/e44c_6p9b_diagnostic.json", "M9/e44d_load_order_audit.json", "M9/e44e_canonical_order.jsonl", "M9/e44e_canonical_order.py") if (R/p).exists()]},
    E9=e9,
    corpus31=dict(rows=rows, one_format_missing=corpus_missing),
    E13_regression=regression,
    E6=dict(rule="vendor: verdict is not Not Matched; weight_only: identity >= 0.65 (gate ignored); "
                 "conservative_and: both; conflict_aware: abstain when the two disagree",
            original_labels=dict(as_served=e6(rows, "served", lbl_original),
                                 one_format=None if corpus_missing else e6(rows, "one_format", lbl_original)),
            audited_labels=dict(as_served=e6(rows, "served", lbl_audited),
                                one_format=None if corpus_missing else e6(rows, "one_format", lbl_audited))),
    E10=dict(rule="predict related iff identity >= t, t = 0.00..1.00 step 0.01; recall over derivatives; "
                  "false-positive rate over same-recipe independent pairs",
             original_labels_5pythia=dict(as_served=e10(rows, "served", lbl_original, sr5),
                                          one_format=None if corpus_missing else e10(rows, "one_format", lbl_original, sr5)),
             audited_labels_5pythia=dict(as_served=e10(rows, "served", lbl_audited, sr5),
                                         one_format=None if corpus_missing else e10(rows, "one_format", lbl_audited, sr5)),
             audited_labels_twelve_same_recipe=dict(
                 as_served=e10(rows, "served", lbl_audited, lambda rr, w: twelve_served),
                 one_format=None if (corpus_missing or twelve_missing) else
                            e10(rows, "one_format", lbl_audited, lambda rr, w: twelve_one),
                 one_format_missing=twelve_missing)))
# reconstruction checks against the published as-served E6/E10 numbers
ck = out["E6"]["original_labels"]["as_served"]
assert [(ck[p]["TP"], ck[p]["FP"], ck[p]["TN"], ck[p]["FN"], ck[p]["abstain"]) for p in
        ("vendor","weight_only","conservative_and","conflict_aware")] == \
       [(21,6,1,3,0),(21,5,2,3,0),(21,5,2,3,0),(21,5,1,3,1)], ("E6 reconstruction fails", ck)
c10 = out["E10"]["original_labels_5pythia"]["as_served"]
assert c10["thresholds_with_recall1_and_fpr0"] == [] and c10["first_t_with_fpr0"] == 0.80 \
       and c10["recall_n_at_first_t_with_fpr0"] == 10 and c10["n_pos"] == 24, ("E10 reconstruction fails", c10)
out["reconstruction_checks"] = "passed: E9 as served equals e9b frozen; E6 and E10 as served equal the published tables"
OUT.write_text(json.dumps(out, indent=1, default=float))

def p9(tag, o):
    fd = o["family_disjoint_split"]
    print(f"  {tag:<34} n={o['n']} ({o['pos']}/{o['neg']})  vendor {o['auc_vendor_coefficients']:.4f}  "
          f"LOO-logit {o['auc_loo_logistic']:.4f}  LOO-nonneg {o['auc_loo_nonneg']:.4f}  in-sample-logit "
          f"{o['auc_insample_logistic']:.4f}  FD vendor {fd.get('auc_vendor',float('nan')):.4f} logit "
          f"{fd.get('auc_logistic',float('nan')):.4f} nonneg {fd.get('auc_nonneg',float('nan')):.4f}  best "
          f"{o['best_single_signal']} {o['auc_best_single_signal']:.4f}  nlf {o['auc_single_signal']['nlf']:.4f}")
print("E9 recalibration"); print("  missing one-format:", e9["one_format_missing"])
for lab in ("manifest_labels", "bio_clinicalbert_relabelled"):
    p9("as served / " + lab, e9["as_served"][lab])
    if e9["one_format"]: p9("one format / " + lab, e9["one_format"][lab])
print("  max |sigma - weighted sum| (one format):", e9["one_format_sigma_check_max_abs_dev"])
print("E6 gate counterfactual (TP/FP/TN/FN/abstain)"); print("  corpus missing:", corpus_missing)
for lab in ("original_labels", "audited_labels"):
    for w in ("as_served", "one_format"):
        x = out["E6"][lab][w]
        if x: print(f"  {lab:<16}{w:<11}" + "  ".join(f"{p} {x[p]['TP']}/{x[p]['FP']}/{x[p]['TN']}/{x[p]['FN']}/{x[p]['abstain']}" for p in x))
print("E10 threshold sweep")
for arm in ("original_labels_5pythia", "audited_labels_5pythia", "audited_labels_twelve_same_recipe"):
    for w in ("as_served", "one_format"):
        x = out["E10"][arm][w]
        if x: print(f"  {arm:<36}{w:<11} pos {x['n_pos']} sr {x['n_same_recipe']}  t(recall1&fpr0)={x['thresholds_with_recall1_and_fpr0']}  "
                    f"first t fpr0={x['first_t_with_fpr0']} recall {x['recall_n_at_first_t_with_fpr0']}/{x['n_pos']}  "
                    f"max SR {x['max_same_recipe_sigma']}  min pos {x['min_positive_sigma']}")
    if out["E10"][arm].get("one_format_missing"): print("   twelve missing:", out["E10"][arm]["one_format_missing"])
print("E13/E41 regression: served", {k: round(v,4) for k,v in regression["served"].items()},
      "| one-format", None if regression["one_format"] is None else {k: round(v,4) for k,v in regression["one_format"].items()},
      "| missing", regression["missing"])
print("wrote", OUT.relative_to(R))
