"""E14c -- recount E14b's composition, and measure the custom-model_type route on the same sample.

Two questions from review round 4, both answerable from configurations E14b already fetched,
so nothing is downloaded:

  (1) Composition. Section VIII says of the 18 ordinary models lacking `architectures` that
      "thirteen are DeBERTa and eleven are first-party Microsoft releases". This recounts both
      from M7/e14b_classified.json (the 18) joined to M7/e14b_hub_prevalence.jsonl (model_type,
      family filter), by two rules each: DeBERTa by model_type (deberta / deberta-v2) and by the
      family filter the repo was sampled under; Microsoft by the `microsoft/` namespace.

  (2) Reviewer B, W5: renaming model_type to an unrecognised string also returns no verdict
      (E7), and the paper says new architectures do this as a matter of course. How many of the
      545 retrieved configurations carry a model_type the pinned transformers release does not
      recognise? The tool loads configs with AutoConfig and, by default, trust_remote_code=False
      (provenancekit/core/signals/metadata.py), so an unregistered model_type is refused
      exactly as in E7's renamed arm. Registry: transformers CONFIG_MAPPING_NAMES at the
      version pinned in requirements-lock.txt.

Writes a NEW file, M7/e14c_composition.json. E14b's files are read, not modified.
"""
import json, collections
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

rows = [json.loads(l) for l in (_ROOT/"M7"/"e14b_hub_prevalence.jsonl").read_text().splitlines() if l.strip()]
by_repo = {r["repo"]: r for r in rows}
fetched = [r for r in rows if r.get("fetched")]
genuine = json.loads((_ROOT/"M7"/"e14b_classified.json").read_text())["genuine"]
g = [by_repo[x] for x in genuine]

deberta_by_type = [r["repo"] for r in g if r.get("model_type") in ("deberta", "deberta-v2")]
deberta_by_filter = [r["repo"] for r in g if r["family_filter"] == "deberta"]
microsoft = [r["repo"] for r in g if r["repo"].lower().startswith("microsoft/")]

unrec = [dict(repo=r["repo"], model_type=r.get("model_type"),
              has_architectures=r.get("has_architectures"), family_filter=r["family_filter"])
         for r in fetched if r.get("has_model_type") and r.get("model_type") not in CONFIG_MAPPING_NAMES]
no_mt = [r["repo"] for r in fetched if not r.get("has_model_type")]

out = dict(
  transformers_version=transformers.__version__,
  n_fetched=len(fetched), n_genuine_missing_architectures=len(g),
  composition=dict(deberta_by_model_type=len(deberta_by_type), deberta_by_family_filter=len(deberta_by_filter),
                   microsoft=len(microsoft), deberta_by_model_type_repos=deberta_by_type,
                   deberta_by_family_filter_repos=deberta_by_filter, microsoft_repos=microsoft,
                   manuscript_stated=dict(deberta=13, microsoft=11)),
  model_type=dict(unrecognised=len(unrec), unrecognised_rows=unrec, absent=len(no_mt),
                  counts=collections.Counter(r.get("model_type") for r in fetched).most_common()),
  note="The sample draws the 40 most-downloaded models per family the tool supports, so it "
       "cannot measure how often genuinely new architectures carry an unregistered model_type.")
(_ROOT/"M7"/"e14c_composition.json").write_text(json.dumps(out, indent=1))
print(f"transformers {transformers.__version__}; {len(fetched)} configurations; {len(g)} genuine lacking architectures")
print(f"DeBERTa by model_type {len(deberta_by_type)}, by family filter {len(deberta_by_filter)}; Microsoft {len(microsoft)}")
print(f"unrecognised model_type: {len(unrec)} {unrec}; model_type absent: {len(no_mt)}")
print("wrote M7/e14c_composition.json")
