# Code and result files (anonymized mirror)

This mirror holds the code and result files behind the submission "Same Recipe, Not Same Weights: What a
Public Lineage Verifier Measures". Everything runs on one laptop (Apple M4, 24 GB) with no GPU, no
credentials, no hosted service and no access request to any vendor. Cold-start disk use is dominated by
the Hugging Face cache (about 15 GB if you run every arm).

## What runs at what cost

| Claim class | Cost | Entry point |
|---|---|---|
| Analyses over frozen results | seconds | `make reproduce-tables`, `make reproduce-figures`, `make reproduce-stats`, `make reproduce-recalibration` |
| Verifier behaviour on public models | minutes to hours | `make smoke`, `make reproduce-deployment` |
| Full measurement campaigns | hours, network-bound | `make reproduce-ladder`, `make reproduce-awm`, `make reproduce-substrate`, `make reproduce-defence` |

`MANIFEST-dataintegrity.txt` lists the frozen result files with their md5 hashes; `make verify-data`
checks them (94 files). Every reproduction script writes new files and never overwrites a frozen one.
`make help` lists every target.

## The object under study

- Verifier: `https://github.com/cisco-ai-defense/model-provenance-kit`, commit
  `a75007d5ca4aaf8df2e3f055f318557a965b93aa` ("Bump version to 1.1.0", 2026-08-12), Apache-2.0, with its
  shipped parquet fingerprint database. No patch is applied; the shipped CLI is called with
  `--json --no-cache`. `make check-version` reports whether the pinned commit is still upstream HEAD.
- Second substrate: the black-box tester of Nikolic et al. (`M1/oracle/model_provenance_testing`).
- Third substrate: AWM (`github.com/LUMIA-Group/AWM` at `bc20ff8e63cec57f5da422ae065686ced275e76d`).
- The three substrates are not vendored here; clone them at the commits above into
  `M1/oracle/model-provenance-kit`, `M1/oracle/model_provenance_testing` and `M7/awm/repo`.

## Layout

- `M1/` verifier runs on public pairs, crash capture (`results/`).
- `M2/` laundering transforms and signal floors; `M2/work/` holds the modified tokenizer and configs of
  the vocabulary-remap attack. They are evidence listed in the manifest, not scratch.
- `M3/` hub-scale survey of declared lineage.
- `M4/` consequence of laundering and the laundering-resistant (alignment-based) design, with its statistics.
- `M5/` sweeps, corpus distances and the locally repaired configurations (`M5/repaired/`, configs and
  tokenizers only).
- `M6/` gold lineage manifest (`M6/build_gold_manifest.py`), Pythia training ladder, data figures.
- `M7/` numbered experiments (E3 AWM, E9 recalibration, E13 regression, E15 transform matrix, E17/E22 rank
  and runtime, E18 defence, E31/E32/E34/E38 alignment checks, E45 one-format re-check, ...), each script
  beside its result file.
- `M8/` deployment testbed: verdict to CycloneDX 1.6 ML-BOM (`emit_bom.py`, validated against the pinned
  `bom-1.6.schema.json`) to an Open Policy Agent admission gate (`policy/supplychain.rego`); needs `opa`
  on PATH (tested with 1.17.1) and `jsonschema`.
- `M9/` same-recipe false positive at 6.9B, merging with the corrected instrument, metadata adversary.
- `analysis/` table printers; `STATISTICS.md` the statistical procedures.

## Ground truth

`M6/gold_lineage_manifest.json` carries, per pair, the relationship class, adaptation type, ground-truth
source, a verbatim supporting quotation, derivation depth, measured tier, identity score, verdict, the
signals, and whether the configuration was modified. Labels are `GOLD` (stated in the child's own model
card) or `SILVER` (release convention, or documented only in an associated publication); one pair is
`DISPUTED` and excluded from both sets. `M7/model_revisions.json` pins each measured model to the Hugging
Face commit it was measured against.

## Seeds and determinism

Transform seeds are recorded in each result row (`seed`, `frac`, `overlap`); randomisation arms use
`numpy.random.default_rng(17)` unless a row says otherwise. Not deterministic: Hugging Face download order
and the black-box tester's prompt sampling. The verifier's scores are deterministic for a given pair of
files.

## Known reproduction hazards

1. The black-box tester is unstable at low prompt counts; reproduce at 2000 prompts
   (`make reproduce-substrate`).
2. AWM needs a causal-LM state dict: saving through `AutoModel` strips the key prefix its regex requires,
   and AWM then reports "no recognizable attention weights". Use `AutoModelForCausalLM`.
3. AWM does not support encoders (BERT, RoBERTa, BART).
4. Five public models lack an `architectures` key and were repaired locally to be verifiable at all; the
   value written determines the tier and therefore the verdict. Repaired rows are marked in the manifest.
5. `M4/defence_expanded.json` matches layers index to index and exists only to enlarge the arms for the
   statistical contrast. The ordering and margin results come from `M4/defence_final_stats.json`, which
   stride-matches distillation pairs; `M4/defence_distill.json` records both readings side by side, and
   `M7/e18_defence.py` reproduces the stride values.
6. Saving a merged model through `AutoModel` drops the language-model head and rewrites `architectures`;
   `M9/e29b_merging.py` saves with the head preserved.
7. MLP extraction is architecture-specific (DistilBERT names its up-projection `ffn.lin1.weight`).
8. Attack arms logged without a fidelity record are reported as `unmeasured` by
   `M7/e15_transform_matrix.py` and are not counted as capability-preserving.

## Anonymization

This mirror is a curated copy for anonymous review. It leaves out the project's internal notes, review
logs and correspondence drafts, the manuscript sources, and three figure scripts for figures that are not
in the submission. Absolute home-directory paths in two log files (`M9/e47/e47_fetch_paths.json`,
`M1/results/mpk_crashes_deberta.jsonl`) were replaced by `~` or `<artifact-root>`; neither file is in
the integrity manifest, and no measured value changed. Internal record identifiers
(`prj_...`, `mis_...`, `jrn_...`) that appear in some notes and docstrings point to the authors' private
research log and carry no identity.

## Ethics

Everything here derives from public Apache-2.0 source, a published fingerprint database and public
checkpoints. No system other than public model hosting was touched. The transforms are evaluated against
a verifier and are not packaged as a laundering tool.
