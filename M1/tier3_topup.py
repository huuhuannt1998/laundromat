"""Gate G0 top-up: 2+ more MFI tier-3 positives (criterion is >=12, we have 10).

Tier 3 means the metadata gate did NOT decide -- the verdict rests on weights.
Those are the only pairs that test the primitive rather than the publisher's own
config.json.  All candidates are public, ungated, small.
"""
import os, sys, json, subprocess, pathlib
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

REPO = pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
OUT = pathlib.Path(str(_ROOT) + "/M1/results/tier3_positives.jsonl")

CANDS = [
    ("google-bert/bert-base-uncased", "huawei-noah/TinyBERT_General_4L_312D"),
    ("facebook/bart-large-cnn",       "sshleifer/distilbart-cnn-12-6"),
    ("google-bert/bert-base-uncased", "prajjwal1/bert-tiny"),
    ("google-bert/bert-base-uncased", "google/bert_uncased_L-4_H-256_A-4"),
    ("FacebookAI/roberta-base",       "sentence-transformers/all-distilroberta-v1"),
]

have = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        if l.strip().startswith("{"):
            r = json.loads(l); have.add((r.get("parent"), r.get("child")))
n3 = sum(1 for l in OUT.read_text().splitlines() if l.strip().startswith("{")
         and json.loads(l).get("scores", {}).get("mfi_tier") == 3) if OUT.exists() else 0
print(f"starting from {n3} tier-3 positives; criterion is >=12", flush=True)

f = OUT.open("a")
for parent, child in CANDS:
    if (parent, child) in have:
        print(f"  cached  {child}", flush=True); continue
    p = subprocess.run(["uv", "run", "provenancekit", "compare", parent, child, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    if i < 0:
        print(f"  ERROR   {child}: {(p.stderr or '')[-160:]}", flush=True); continue
    r = json.loads(p.stdout[i:])
    sc, sg = r.get("scores", {}), r.get("signals", {})
    rec = dict(parent=parent, child=child, scores=sc, signals=sg,
               note="gate G0 tier-3 top-up")
    f.write(json.dumps(rec) + "\n"); f.flush()
    if sc.get("mfi_tier") == 3:
        n3 += 1
    print(f"  tier={sc.get('mfi_tier')} sig_id={sc.get('identity_score')} "
          f"decision={sc.get('provenance_decision')}  {child}   [tier3 total now {n3}]", flush=True)
    if n3 >= 12:
        print(f"  CRITERION MET: {n3} >= 12", flush=True); break
f.close()
print(f"DONE-TIER3 total={n3} {'PASS' if n3>=12 else 'STILL SHORT'}", flush=True)
