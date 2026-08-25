"""E14 -- metadata prevalence over the verifier's OWN reference database.

Section VIII reports eight models that produce no verdict and admits the hub was not
surveyed. That leaves the defect looking anecdotal. This turns it into a measurement over
a population the vendor itself defines: every asset in the shipped fingerprint catalog,
184 models across 39 families. If a field the gate requires is missing from the vendor's
own reference ecosystem, that is a much stronger statement than eight encountered examples.

Only configuration is retrieved -- config.json, a few kilobytes each. No weights are
downloaded and no model is executed.

Gate-critical fields, from the metadata path measured in E7:
  architectures  absent or null  -> no verdict
  model_type     absent or unrecognised -> no verdict
Both are recorded separately, along with presence of the shape fields the tier test reads.
"""
import json, pathlib, time, collections
import os as _os, pathlib as _pl
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
DB  = _ROOT/"M1"/"oracle"/"model-provenance-kit"/"src"/"provenancekit"/"data"/"database"
OUT = _ROOT/"M7"/"e14_prevalence.jsonl"
from huggingface_hub import hf_hub_download

man = json.loads((DB/"catalog"/"manifest.json").read_text())
assets = []
for sh in man["shards"]:
    d = json.loads((DB/sh["shard_path"]).read_text())
    pub = d["family"].get("publisher") or d["family"]["family_id"]
    for a in d["assets"]:
        uri = a.get("source_uri") or ""
        repo = uri.split("huggingface.co/")[-1].strip("/") if "huggingface.co/" in uri else None
        assets.append(dict(family=d["family"]["family_id"], publisher=pub,
                           model_id=a.get("model_id"), repo=repo,
                           bucket=a.get("param_bucket")))
print(f"population: {len(assets)} assets across {len(man['shards'])} families "
      f"(the vendor's shipped catalog)")

done = {}
if OUT.exists():
    for l in OUT.read_text().splitlines():
        if l.strip():
            r = json.loads(l); done[r["repo"]] = r

for i, a in enumerate(assets, 1):
    if not a["repo"] or a["repo"] in done: continue
    rec = dict(a)
    try:
        p = hf_hub_download(a["repo"], "config.json", repo_type="model")
        cfg = json.loads(pathlib.Path(p).read_text())
        rec.update(fetched=True,
                   has_architectures=("architectures" in cfg),
                   architectures_null=(cfg.get("architectures") is None),
                   architectures_empty=(isinstance(cfg.get("architectures"), list)
                                        and len(cfg["architectures"]) == 0),
                   has_model_type=("model_type" in cfg),
                   model_type=cfg.get("model_type"),
                   has_hidden=any(k in cfg for k in ("hidden_size","d_model","n_embd")),
                   has_layers=any(k in cfg for k in ("num_hidden_layers","n_layer","num_layers")))
    except Exception as e:
        rec.update(fetched=False, error=f"{type(e).__name__}")
    with OUT.open("a") as fh: fh.write(json.dumps(rec)+"\n")
    done[a["repo"]] = rec
    if i % 25 == 0: print(f"  {i}/{len(assets)}", flush=True)

rows = [r for r in done.values() if r.get("fetched")]
miss = [r for r in done.values() if not r.get("fetched")]
def pct(k, n): return f"{k} ({100*k/max(n,1):.1f}%)"
n = len(rows)
no_arch = [r for r in rows if not r["has_architectures"] or r["architectures_null"]]
no_mt   = [r for r in rows if not r["has_model_type"]]
either  = [r for r in rows if (not r["has_architectures"] or r["architectures_null"]
                               or not r["has_model_type"])]
print(f"\n{'population':<34}{'N':>5}{'no architectures':>20}{'no model_type':>16}{'any gate issue':>17}")
print("-"*94)
print(f"{'vendor reference database':<34}{n:>5}{pct(len(no_arch),n):>20}{pct(len(no_mt),n):>16}{pct(len(either),n):>17}")
bypub = collections.defaultdict(list)
for r in rows: bypub[r["publisher"]].append(r)
print(f"\nby publisher (only those with any gate-critical issue):")
any_issue = False
for pub, rs in sorted(bypub.items()):
    bad = [r for r in rs if (not r["has_architectures"] or r["architectures_null"]
                             or not r["has_model_type"])]
    if bad:
        any_issue = True
        print(f"  {pub:<28} {len(bad)}/{len(rs)}")
if not any_issue: print("  none")
if miss: print(f"\nconfig.json unavailable for {len(miss)} assets: "
               f"{collections.Counter(r.get('error') for r in miss)}")
json.dump(dict(population="vendor shipped fingerprint catalog", n_assets=len(assets),
               n_fetched=n, n_unavailable=len(miss),
               missing_architectures=len(no_arch), missing_model_type=len(no_mt),
               any_gate_critical=len(either),
               examples=[r["repo"] for r in either][:20]),
          open(_ROOT/"M7"/"e14_prevalence_summary.json","w"), indent=1)
print("\nDONE-E14")
