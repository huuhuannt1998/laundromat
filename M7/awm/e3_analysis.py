"""Paired AWM vs studied-tool comparison on identical pairs (review E3 metrics)."""
import json, pathlib, itertools

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

ROOT=pathlib.Path(_ROOT)
awm={json.loads(l)["label"]: json.loads(l)
     for l in (ROOT/"M7"/"awm"/"e3_awm_results.jsonl").read_text().splitlines() if l.strip()}
lad={r["step"]: r for r in (json.loads(l) for l in
      (ROOT/"M6"/"e12_pythia_ladder.jsonl").read_text().splitlines() if l.strip())}

# Cisco identity scores on exactly the same pairs.
CISCO = {
 "gpt2 -> distilgpt2":            0.8910,   # M1/results/tier3_positives.jsonl
 "SmolLM2 -> Instruct":           0.9992,   # M2/results/gate_HuggingFaceTB--SmolLM2-135M-Instruct.jsonl
 "Qwen2.5-0.5B -> Instruct":      0.9994,   # M2/results/gate_Qwen--Qwen2.5-0.5B-Instruct.jsonl
 "pythia-160m step1k->final":     lad[1000]["identity_score"],
 "pythia-160m step16k->final":    lad[16000]["identity_score"],
 "pythia-160m step128k->final":   lad[128000]["identity_score"],
 "pythia-70m vs -deduped":        0.7856,   # Table fp
 "pythia-160m vs -deduped":       0.6745,
 "pythia-410m vs -deduped":       0.7810,
 "pythia-1b vs -deduped":         0.7262,
}
def auc(pos,neg):
    if not pos or not neg: return None
    w=sum((1.0 if p>n else 0.5 if p==n else 0.0) for p in pos for n in neg)
    return w/(len(pos)*len(neg))

rows=[]
for lab,r in awm.items():
    if r.get("status")!="OK": continue
    rows.append(dict(label=lab, cls=r["cls"], awm=r["awm_wqwk"], z=r["awm_z"],
                     cisco=CISCO.get(lab)))
print(f"{'class':<9}{'pair':<30}{'AWM':>8}{'z':>8}{'Cisco sig_id':>14}")
for r in sorted(rows,key=lambda r:(r["cls"],-(r["awm"] or 0))):
    c = f"{r['cisco']:.4f}" if r["cisco"] is not None else "  --"
    print(f"{r['cls']:<9}{r['label']:<30}{r['awm']:8.4f}{(r['z'] or 0):8.1f}{c:>14}")

matched=[r for r in rows if r["cisco"] is not None]
D  =[r for r in matched if r["cls"].startswith("D-")]
NSR=[r for r in matched if r["cls"]=="N-SR"]
print(f"\n--- paired, matched pairs only: {len(D)} true-descent vs {len(NSR)} independent same-recipe ---")
print(f"AWM   : descent {min(r['awm'] for r in D):.4f}-{max(r['awm'] for r in D):.4f}   "
      f"N-SR {min(r['awm'] for r in NSR):.4f}-{max(r['awm'] for r in NSR):.4f}   "
      f"AUC={auc([r['awm'] for r in D],[r['awm'] for r in NSR]):.3f}")
print(f"Cisco : descent {min(r['cisco'] for r in D):.4f}-{max(r['cisco'] for r in D):.4f}   "
      f"N-SR {min(r['cisco'] for r in NSR):.4f}-{max(r['cisco'] for r in NSR):.4f}   "
      f"AUC={auc([r['cisco'] for r in D],[r['cisco'] for r in NSR]):.3f}")
ov_awm=[r['label'] for r in D if r['awm']  < max(x['awm']  for x in NSR)]
ov_cis=[r['label'] for r in D if r['cisco']< max(x['cisco'] for x in NSR)]
print(f"\ntrue descendants scoring below the top independent pair:")
print(f"  AWM  : {ov_awm}")
print(f"  Cisco: {ov_cis}")
# AWM's own significance reading of the negatives
print(f"\nAWM z-scores on pairs that share NO weights (N-SR): "
      f"{[round(r['z'],1) for r in NSR]}")
print(f"AWM z-scores on genuinely unrelated pairs        : "
      f"{[round(awm[l]['awm_z'],2) for l in awm if awm[l].get('cls')=='N-UNREL' and awm[l].get('status')=='OK']}")
