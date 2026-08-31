# LAUNDROMAT — artifact

Everything below runs on one laptop (Apple M4, 24 GB) with no GPU, no credentials, no hosted
service, and no access request to any vendor. Total cold-start disk use is dominated by the
Hugging Face cache (~15 GB if you run every arm).

## 0. What this artifact supports

The paper makes three kinds of claim. They have different reproduction costs and we separate
them so a reviewer can spend effort where it matters.

| Claim class | Cost | Entry point |
|---|---|---|
| Analyses over frozen results (every table, every figure) | seconds | `make reproduce-tables`, `make reproduce-figures` |
| Verifier behaviour on public models | minutes to hours | `make reproduce-gate`, `make reproduce-crash` |
| Full measurement campaigns | hours, network-bound | `make reproduce-ladder`, `make reproduce-awm`, `make reproduce-substrate` |

Frozen results are recorded in `MANIFEST-dataintegrity.txt` (80 files, md5). `make verify-data`
checks them. Every reproduction script writes **new** files; none overwrites a frozen one.

## 1. The object under study

- Repository: `https://github.com/cisco-ai-defense/model-provenance-kit`
- Commit: `a75007d5ca4aaf8df2e3f055f318557a965b93aa` ("Bump version to 1.1.0", 2026-08-12)
- License: Apache-2.0. Ships a 184-file parquet fingerprint database.
- No patch is applied. We call the shipped CLI with `--json --no-cache`.
- Verified 2026-08-24: this commit was still `origin/main`, so the pinned and current versions
  are the same object. **Re-check before submission** — if the tip has moved, §IV's attestation
  must be re-run.

Local clone: `M1/oracle/model-provenance-kit`.

Second and third substrates:
- Black-box tester (`M1/oracle/model_provenance_testing`), from Nikolic et al.
- AWM (`M7/awm/repo`), `github.com/LUMIA-Group/AWM` @ `bc20ff8e63cec57f5da422ae065686ced275e76d`.

## 2. Ground truth

`M6/gold_lineage_manifest.json` (schema `laundromat.gold-lineage-manifest/v1`) carries, per pair:
relationship class, adaptation type, ground-truth source, a **verbatim supporting quotation**,
derivation depth, measured tier, identity score, verdict, the five signals, displacement, and
whether we modified the configuration.

- Negatives are not pooled. `N-SR` (independent run, same recipe, same size) is distinct from
  `N-FAM` (same architecture family, independently pretrained).
- Labels are graded `GOLD` (stated in the child's own model card) or `SILVER` (release
  convention, or documented only in an associated publication). Headline numbers use GOLD.
- One pair is `DISPUTED` and excluded from both sets; see Appendix "Corpus label audit".

Rebuild with `python3 M6/build_gold_manifest.py`.

## 3. Where did Table X / Figure Y come from?

| Artifact | Generator | Frozen input |
|---|---|---|
| Fig. 1 pipeline | `M6/make_fig1.py` | schematic |
| Fig. 2 score by relationship | `M6/make_fig2.py` | `M6/gold_lineage_manifest.json`, `M2/results/null_frozen.json` |
| Fig. 3 displacement fit | `M6/make_fig3.py` | `M5/sweep_bert.jsonl`, `M5/corpus_distance.jsonl` |
| Fig. 4 design rule | `M6/make_fig4.py` | `M4/adaptive_qk.json` |
| Tab. related work | hand-authored | literature |
| Tab. same-recipe false positives | §V | `M1/results/pairs.jsonl` |
| Tab. family generality | appendix | `M1/results/scans.jsonl` |
| Tab. weight-decided derivatives | appendix | `M1/results/tier3_positives.jsonl` |
| Tab. no-verdict models | appendix | `M1/results/mpk_crashes.jsonl` |
| Tab. additivity | appendix | `M2/results/decomp_transfer.jsonl` |
| Tab. alignment variants | appendix | `M4/defence_expanded.json`, `M4/lap2.json` |
| Tab. benign children | appendix | `M5/corpus_distance.jsonl` |
| Tab. training ladder | appendix | `M6/e12_pythia_ladder.jsonl` |
| Tab. second substrate | appendix | `M1/oracle/model_provenance_testing/runs/*.csv` |
| AWM baseline | §V | `M7/awm/e3_awm_results.jsonl` |
| Recalibration (E9) | §V-F | `M7/e9_recalibrate.py` → `M7/e9_recalibration.json` |
| Regression CIs (E13) | §VII | `M7/e13_regression.py` → `M7/e13_regression.json` |
| Rank + runtime (E17/E22) | §VI, §X | `M7/e17_rank_e22_runtime.py` → `M7/e17_e22.json` |
| Consolidated transform matrix (E15) | §VI | `M7/e15_transform_matrix.py` → `.csv` / `.md` |
| Deployment testbed (E25) | §III, §VIII-B | `M8/run_testbed.py` → `M8/e25_results.json` |

Figures 2–4 regenerate from frozen data, so figure and text cannot drift. `make_fig3.py` prints
the fit it produces; it must read `sigma = 0.8692 - 0.3327 d, R2 = 0.6927, last_obs = 0.8295`.

## 3b. Model revisions

`M7/model_revisions.json` pins every model the paper measures to the Hugging Face commit sha the
measurement was made against — 66 ids, 61 resolved from the local cache. A model id alone is not
a pin: repositories are mutable, and Pythia checkpoints in particular are fetched at a named
revision (`stepN`) rather than `main`. We record revisions only for models we touch, not for the
thousands listed in the vendor's fingerprint database, whose revisions are not ours to attest.

## 3c. Anonymity

Scripts derive the repository root from their own location. An earlier state embedded an absolute
path containing the author's username in 43 scripts, which would have de-anonymised the artifact;
`grep -rl "/Users/"` over the tracked sources now returns nothing. The paper carries
`\author{Anonymous Submission}` with no affiliation, e-mail or acknowledgements.

## 3d. The deployment testbed

`M8/` holds the consumer pipeline of §III: `emit_bom.py` turns a verdict into a CycloneDX 1.6
ML-BOM validated against the pinned official schema (`M8/bom-1.6.schema.json`),
`policy/supplychain.rego` is the OPA admission gate, and `run_testbed.py` drives the corpus
through both consumer wirings. Requires `opa` on PATH (tested with 1.17.1) and `jsonschema`.
Model licences in `M8/model_licences.json` are the models' real licence tags, fetched from the
hub, not assigned by us.

## 4. Seeds and determinism

Transform seeds are recorded in each result row (`seed`, `frac`, `overlap`). The randomisation
arms use `numpy.random.default_rng(17)` unless a row says otherwise. Two independent seeds are
reported wherever a floor is claimed, because one of them moved a floor from 0.5013 to 0.4987 and
the paper quotes the lower.

Not deterministic: Hugging Face download order, and the black-box tester's prompt sampling. The
tester's verdicts are **sample-dependent at 300 prompts** — see §5.

## 5. Known reproduction hazards, stated rather than discovered

1. **The black-box tester is unstable at low prompt counts.** Across two 300-prompt samples,
   three of four common pairs flipped outcome. Reproduce at 2000 prompts; at that power the
   verdicts are stable and three of four same-recipe pairs are false positives.
2. **AWM needs a causal-LM state dict.** Saving a checkpoint via `AutoModel` strips the
   `transformer.`/`model.` prefix its key regex requires, and it will silently report
   "no recognizable attention weights" on models it does support. Use `AutoModelForCausalLM`.
3. **AWM does not support encoders.** BERT/RoBERTa/BART match none of its patterns. This is a
   property of the released method, not of our harness.
4. **Five public models lack an `architectures` key** and were repaired locally to be verifiable
   at all. The value written determines the tier and therefore the verdict. Repaired rows are
   marked in the manifest and in the paper.
5. **Two defence files exist and they are not interchangeable.** `M4/defence_expanded.json`
   matches layers index-to-index and exists only to enlarge the arms for the *statistical
   contrast* (see its docstring). Read alone it appears to show the defence failing: with
   index matching, distilgpt2 scores 0.2337 against a same-recipe negative at 0.2679, a margin
   of **-0.0342**. The paper's ordering and margin claims come from `M4/defence_final_stats.json`,
   which **stride-matches** distillation pairs (child layer i against parent layer i x stride) --
   the correspondence DistilBERT actually has, since it initialises from every other teacher
   layer. `M4/defence_distill.json` records both readings side by side (`lap_index` 0.2900 vs
   `lap_stride` 0.5255 for bert to distilbert) so the choice is auditable. `M7/e18_defence.py`
   reproduces the stride values exactly (0.5987 distilroberta, 0.5148 distilgpt2).
6. **MLP extraction is architecture-specific.** DistilBERT names its up-projection
   `ffn.lin1.weight`, not `intermediate.dense.weight`. An extractor missing that key returns an
   empty layer list and a silent `null` score rather than an error.
7. **Absence of a fidelity record is not evidence of capability preservation.** Some attack arms
   were logged without a fidelity block. `M7/e15_transform_matrix.py` reports those as
   `unmeasured` and excludes them from the preserving tally. Counting them as preserved
   manufactures an evasion that was never demonstrated — the one arm below the evasion bar is an
   anchor-noise injection at eight times the relative scale, i.e. the model-destroying family.
8. **Pagination is non-monotone.** Trimming LaTeX text sometimes increases overflow.

## 5b. Clean-room reproduction — performed 2026-08-25

The artifact was staged into a fresh directory outside the working tree, with all caches,
worktrees and the vendor clones excluded, and every analysis target re-run there. All headline
numbers reproduced exactly: 80/80 integrity, four figures regenerated, gold manifest 31 rows
(19 primary / 11 secondary), displacement fit `0.8692 -0.3327 d, R2 0.6927`, recalibration
`vendor 0.600 / LOO 0.622`, defence `margin +0.0428, AUC 1.0000`, transform matrix 49 arms.

It found two real defects first, both now fixed:

1. **Scripts hardcoded an absolute repo root.** Run from any other checkout they silently read
   and wrote the author's original tree rather than the reviewer's. The first clean-room attempt
   therefore partly measured the wrong directory and would have reported a false pass. Every
   script now derives the root from its own location (`LAUNDROMAT_ROOT` overrides).
2. **Four manifest entries live under `M2/work/`.** They are the vocabulary-remap attack artifact
   — the modified `tokenizer.json` and its configs — not measurements. A reviewer excluding
   working directories from a copy sees `verify-data` fail on four files. Do not exclude
   `M2/work/`; it holds evidence, not scratch.

## 6. Smoke test (about two minutes)

```
make verify-data        # 80 frozen md5s
make smoke              # one gate comparison + one figure regeneration
```

`make smoke` must print a `Confirmed Match` for the self-comparison and reproduce the Fig. 3 fit
constants above.

## 6b. Submission freeze

Per the S&P 2027 CFP, artifacts "must not be updated after the paper deadline has passed" and must
remain anonymised. This artifact therefore freezes at the 2026-11-17 paper deadline. The one
perishable claim inside it is the version attestation of §1: re-run `make check-version`
immediately before the freeze, and if upstream has moved, re-run the replication and update §IV
rather than letting the attestation go stale.

## 7. Ethics

Everything here derives from public Apache-2.0 source, a published fingerprint database, and
public checkpoints. No vendor was contacted; no system other than public model hosting was
touched. The transforms are evaluated against a verifier and are not packaged as a laundering
tool.
