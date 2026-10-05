"""Gate G0 top-up, round 2.

Tier 3 requires BOTH the arch_hash and the family_hash to miss, so the verdict
falls through to the weight signals.  Round 1 mostly produced tier-1 matches
(same architecture string) or hit MPK's architectures:null crash.  These
candidates are cross-architecture-class derivatives, which is the pattern that
actually reached tier 3 in the existing set (distilbert<-bert, distilgpt2<-gpt2,
distilroberta<-roberta, xtremedistil<-bert).
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
CRASH = pathlib.Path(str(_ROOT) + "/M1/results/mpk_crashes.jsonl")

CANDS = [
    ("google-bert/bert-base-uncased", "distilbert/distilbert-base-uncased-distilled-squad"),
    ("facebook/bart-large-xsum",      "sshleifer/distilbart-xsum-12-3"),
    ("google-bert/bert-base-uncased", "Intel/dynamic_tinybert"),
    ("google-bert/bert-base-uncased", "cross-encoder/ms-marco-MiniLM-L6-v2"),
    ("FacebookAI/roberta-base",       "cross-encoder/stsb-roberta-base"),
    ("google-bert/bert-base-uncased", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
    ("facebook/bart-large-cnn",       "sshleifer/distilbart-cnn-6-6"),
]

seen = set()
rows = []
if OUT.exists():
    for l in OUT.read_text().splitlines():
        if l.strip().startswith("{"):
            r = json.loads(l); rows.append(r); seen.add((r.get("parent"), r.get("child")))
n3 = sum(1 for r in rows if r.get("scores", {}).get("mfi_tier") == 3)
print(f"starting from {n3} tier-3 rows in file; criterion >=12", flush=True)

f = OUT.open("a"); cf = CRASH.open("a")
for parent, child in CANDS:
    if (parent, child) in seen:
        print(f"  cached  {child}", flush=True); continue
    p = subprocess.run(["uv", "run", "provenancekit", "compare", parent, child, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    if i < 0:
        err = (p.stderr or "")[-400:]
        # separate MPK's architectures:null defect from ordinary fetch failures
        kind = "MPK_DEFECT_architectures_null" if "list_type" in err else "fetch_or_other"
        cf.write(json.dumps(dict(parent=parent, child=child, kind=kind, err=err)) + "\n"); cf.flush()
        print(f"  ERROR[{kind}]  {child}", flush=True); continue
    r = json.loads(p.stdout[i:])
    sc, sg = r.get("scores", {}), r.get("signals", {})
    f.write(json.dumps(dict(parent=parent, child=child, scores=sc, signals=sg,
                            note="gate G0 tier-3 top-up r2")) + "\n"); f.flush()
    if sc.get("mfi_tier") == 3:
        n3 += 1
    print(f"  tier={sc.get('mfi_tier')} sig_id={sc.get('identity_score')} "
          f"{sc.get('provenance_decision')}  {child}   [tier3 now {n3}]", flush=True)
    if n3 >= 12:
        print("  CRITERION MET", flush=True); break
f.close(); cf.close()
print(f"DONE-TIER3R2 total={n3} {'PASS' if n3 >= 12 else 'STILL SHORT'}", flush=True)
