"""E41 -- the section VII displacement regression with every identity score read in one format.

E36/E39/E40 re-read each of the eleven true derivatives of section VII after re-saving its own
pytorch_model.bin as safetensors (squad-v1 already ships safetensors and is taken as published).
E35 supplies twitter-roberta-base, read from the hub's own safetensors conversion. This refits the same linear model E13 fits (identity on embedding displacement, eleven points,
same exclusions) on those scores and on the published ones, and reports both. Pure
recomputation over frozen files. Writes a NEW file, M7/e41_regression_format.json.
"""
import json
import numpy as np
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
R = _ROOT
EXCL = {"nlpaueb/legal-bert-base-uncased",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "emilyalsentzer/Bio_ClinicalBERT"}                       # as in M7/e13_regression.py
pts = []
for f in ("M5/sweep_bert.jsonl", "M5/corpus_distance.jsonl"):
    for line in (R/f).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line); d = r.get("displacement", {}); s = r.get("scores", {})
        if d.get("embedding") is not None and s.get("identity_score") is not None and r["child"] not in EXCL:
            pts.append(dict(child=r["child"], d_emb=d["embedding"], published=s["identity_score"]))
conv = {}
for f in ("M9/e35_serialisation.jsonl", "M9/e36_format_sensitivity.jsonl", "M9/e39_format_bert_sweep.jsonl", "M9/e40_format_roberta_sweep.jsonl"):
    for line in (R/f).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r["format"] in ("safetensors", "safetensors-from-bin") and r.get("scores"):
            conv[r["child"]] = r["scores"]["identity_score"]
SHIPS_SAFETENSORS = {"csarron/bert-base-uncased-squad-v1"}
for p in pts:
    p["one_format"] = p["published"] if p["child"] in SHIPS_SAFETENSORS else conv.get(p["child"])
missing = [p["child"] for p in pts if p["one_format"] is None]
def fit(y):
    x = np.array([p["d_emb"] for p in pts]); y = np.array(y)
    c = np.polyfit(x, y, 1); pred = np.polyval(c, x)
    r2 = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
    return dict(slope=float(c[0]), intercept=float(c[1]), r2=float(r2),
                d_at_065=float((0.65-c[1])/c[0]), d_at_05298=float((0.5298-c[1])/c[0]))
out = dict(n=len(pts), missing=missing, points=pts,
           published=fit([p["published"] for p in pts]),
           one_format=None if missing else fit([p["one_format"] for p in pts]))
(R/"M7"/"e41_regression_format.json").write_text(json.dumps(out, indent=1))
for k in ("published", "one_format"): print(k, out[k])
for p in pts: print(f"  {p['child']:<50} d_emb {p['d_emb']:.3f}  published {p['published']}  one-format {p['one_format']}")
