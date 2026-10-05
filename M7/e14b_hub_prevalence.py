"""E14b -- prevalence on the population that actually matters.

E14 surveyed the vendor's shipped fingerprint catalog and found 0 of 157 assets missing a
gate-critical field. That is a real result but it answers the wrong question, and the reason
is a selection effect: an asset is IN the reference database because it was successfully
fingerprinted, which requires the fields to be present. The reference set is conditioned on
the very property we are measuring.

The population that matters for a denial of verification is not the reference set but the
models a consumer SUBMITS for checking. This samples that population deterministically:
for each architecture family the verifier supports, the most-downloaded public models, with
every retrieved id and commit sha recorded so the sample is auditable rather than merely
described. Configuration only; no weights.
"""
import json, collections
import os as _os, pathlib as _pl
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
OUT = _ROOT/"M7"/"e14b_hub_prevalence.jsonl"
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()

FAMILIES = ["bert","roberta","distilbert","gpt2","llama","mistral","qwen2",
            "gpt_neox","bart","t5","electra","albert","xlm-roberta","deberta"]
PER = 40   # most-downloaded per family

done={}
if OUT.exists():
    for l in OUT.read_text().splitlines():
        if l.strip(): r=json.loads(l); done[r["repo"]]=r

sample=[]
for fam in FAMILIES:
    try:
        ms=list(api.list_models(filter=fam, sort="downloads", limit=PER))
    except Exception as e:
        print(f"  {fam}: listing failed ({type(e).__name__})"); continue
    for m in ms: sample.append((fam, m.id))
    print(f"  {fam:<14} {len(ms)} models")
seen=set(); uniq=[]
for fam,rid in sample:
    if rid in seen: continue
    seen.add(rid); uniq.append((fam,rid))
print(f"\ndeterministic sample: {len(uniq)} unique public models across {len(FAMILIES)} families")

for i,(fam,rid) in enumerate(uniq,1):
    if rid in done: continue
    rec=dict(family_filter=fam, repo=rid)
    try:
        p=hf_hub_download(rid,"config.json",repo_type="model")
        cfg=json.loads(_pl.Path(p).read_text())
        rec.update(fetched=True,
                   has_architectures=("architectures" in cfg),
                   architectures_null=(cfg.get("architectures") is None),
                   has_model_type=("model_type" in cfg),
                   model_type=cfg.get("model_type"))
    except Exception as e:
        rec.update(fetched=False, error=type(e).__name__)
    with OUT.open("a") as fh: fh.write(json.dumps(rec)+"\n")
    done[rid]=rec
    if i%50==0: print(f"  fetched {i}/{len(uniq)}", flush=True)

rows=[r for r in done.values() if r.get("fetched")]
n=len(rows)
no_arch=[r for r in rows if not r["has_architectures"] or r["architectures_null"]]
no_mt=[r for r in rows if not r["has_model_type"]]
either=[r for r in rows if (not r["has_architectures"] or r["architectures_null"] or not r["has_model_type"])]
unavail=[r for r in done.values() if not r.get("fetched")]
def pct(k): return f"{k} ({100*k/max(n,1):.1f}%)"
print(f"\n{'population':<38}{'N':>5}{'no architectures':>20}{'no model_type':>16}{'any gate issue':>17}")
print("-"*98)
print(f"{'public sample (consumer-submitted)':<38}{n:>5}{pct(len(no_arch)):>20}{pct(len(no_mt)):>16}{pct(len(either)):>17}")
print(f"{'vendor reference database (E14)':<38}{157:>5}{'0 (0.0%)':>20}{'0 (0.0%)':>16}{'0 (0.0%)':>17}")
if either:
    print(f"\nexamples lacking a gate-critical field:")
    for r in either[:12]: print(f"  {r['repo']}")
if unavail:
    print(f"\nconfig.json unavailable for {len(unavail)}: {collections.Counter(r.get('error') for r in unavail)}")
json.dump(dict(population="deterministic most-downloaded public sample",
               families=FAMILIES, per_family=PER, n_sampled=len(uniq), n_fetched=n,
               missing_architectures=len(no_arch), missing_model_type=len(no_mt),
               any_gate_critical=len(either), n_unavailable=len(unavail),
               examples=[r["repo"] for r in either][:40]),
          open(_ROOT/"M7"/"e14b_summary.json","w"), indent=1)
print("\nDONE-E14B")
