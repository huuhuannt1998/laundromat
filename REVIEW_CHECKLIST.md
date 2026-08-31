# S&P 2027 mock-review compliance tracker

Audited against `manuscripts/feedback/SP2027_Recipe_or_Descent_Master_Review_and_Revision_Plan.md`
(§42 priority matrix, §43 minimum package, §49 final checklist).

**Last audited: 2026-08-25** (E5, E7, E9, E11, E13, E14, E15, E16, E17, E19, E22 and A1 closed same day). Update this file whenever an item moves.

Legend: **DONE** · **PARTIAL** (something real exists, but short of what §43/§49 asks) ·
**OPEN** · **N/A** (PI decision, or blocked by a standing constraint)

> **Status 2026-08-25: the checklist is closed.** All 44 boxes of the review's §49 final
> checklist are ticked, all ten §43 mandatory experiments are complete, and both P0 blockers are
> resolved. The two items not executed (E23, E24) are P2/P3, neither appears in the §43 minimum
> package, and both are blocked by the standing no-new-hardware constraint rather than by effort;
> E24's actual stated requirement — cite and discuss — is met.

---

## Verified against the actual CFP (sp2027.ieee-security.org/cfpapers.html, checked 2026-08-25)

Previously I worked from the review document's paraphrase. The CFP itself says:

- **"Submitted papers may include up to 13 pages of text and up to 5 pages for references and
  appendices, totaling no more than 18 pages."**
- **"LaTeX submissions using the IEEE templates must use IEEEtran.cls version 1.8b with options
  'conference,compsoc'."** — ours is verified `IEEEtran.cls V1.8b`, options exact.
- **Ethics Considerations does NOT count toward page limits** — at submission it is a HotCRP
  *field*; the section is added to the manuscript at camera-ready, "where it will not count
  toward page limits." (New information; I had been counting it as body.)
- Cycle 2: **abstract 2026-11-10, paper 2026-11-17**, notification 2027-03-05, camera-ready
  2027-04-08.
- Anonymity: no names/affiliations on the title page, avoid revealing identity in text, cite own
  prior work in third person. Verified compliant; no first-person self-citation found.
- **Artifacts "must not be updated after the paper deadline has passed"** and must remain
  anonymised. Operationally important: the artifact freezes at the deadline.

**Open constraint:** body text currently runs about one column into p14 against the 13-page text
limit. Total is 18 and references+appendices are 5, both compliant. Closing the last column needs
a content decision rather than further prose tightening — reflow absorbs small compressions.

## P0 — blockers

| ID | Item | Status | Evidence / note |
|---|---|---|---|
| P0-1 | Vendor disclosure | **N/A — PI decision** | PI: *"I do not want to send anything to anybody."* §XIII states plainly that a document exists, has **not** been transmitted, and that notification is not part of this work. §49 asks disclosure status be *accurately stated* — that box is satisfied; the disclosure box itself is closed by PI direction, not by us. |
| P0-2 | S&P template + anonymity | **DONE** | `\documentclass[conference,compsoc]{IEEEtran}` verified mandatory; `\author{Anonymous Submission}`; no author/affiliation/acknowledgement strings; no e-mail or institution in source. |
| A1 | Clean-room artifact reproduction | **DONE** | Staged to a fresh tree 2026-08-25; all headline numbers reproduced. Caught and fixed two defects: (a) 21 scripts hardcoded an absolute repo root and would have read the author's tree from a reviewer's checkout; (b) four manifest entries live under `M2/work/` (the vocab-remap attack artifact) so that path must not be excluded. |

## Mandatory experiments (§43 "do not submit without")

| ID | Item | Status | Evidence |
|---|---|---|---|
| E1 | Latest-version replication | **DONE** | `git fetch` 2026-08-24: `origin/main` == pinned `a75007d5…`, zero commits since. Stated in §IV. **Perishable — recheck before 17 Nov.** |
| E2 | Gold lineage corpus | **DONE** | `M6/gold_lineage_manifest.json`, 31 rows, 19 GOLD / 11 SILVER / 1 excluded. Negatives split N-SR vs N-FAM. Found and excluded one label defect (BERT-Tiny). §V-B + `app:corpus`. |
| E3 | Official AWM comparison | **DONE** | `LUMIA-Group/AWM` @ `bc20ff8`, 14 causal pairs. AUC 0.833 vs 0.750; z 40–106 on zero-shared-weight pairs. §V `sec:awm`. |
| E4 | Expand second verifier | **DONE** | 6×6 at 300 and 2000 prompts. 3/4 same-recipe pairs false-positive at 2000. `tab:substrate`, `app:substrate`. |
| E6 | Gate counterfactual | **DONE** | Four policies over the 31-pair corpus. §V-F. |
| E8 | Signal ablation | **DONE** | Leave-one-out + concept groups. §V-F, `M6/mock_review_experiments.md`. |
| E10 | Threshold sensitivity | **DONE** | Full sweep. §V-F. |
| E12 | Training-progress experiment | **DONE** | Nine pythia-160m checkpoints, exact token axis. §VII `sec:e12`, `tab:ladder`. |
| E18 | Expand alignment corpus | **DONE** | 17/18 → 30 scored pairs; same-family class added; margin 0.2469 → **0.0428** reported. §IX, `tab:defence`. |
| E20 | Adaptive alignment attacks | **DONE** | QK/OV invariance attacks; design rule. §IX, `fig:sym`. |

**All ten mandatory experiments are complete.**

## Remaining P1 experiments (§42 — not in the §43 minimum set)

| ID | Item | Priority | Status | What it needs |
|---|---|---|---|---|
| E9 | Recalibrate with correct negatives | Extremely high | **DONE** | `M7/e9_recalibrate.py`. 28 audited pairs (18 derived / 10 hard negatives). Vendor AUC 0.600 → refit LOO 0.622; in-sample ceiling 0.822; family-disjoint refit **inverts** to 0.125. Conclusion: signals lack lineage specificity — recalibration does not fix it. §V-F. |
| E5 | Same-recipe negative expansion | Very high | **DONE** | `M7/e5_same_recipe.py`. Written inclusion rule applied before collecting; **Google MultiBERTs** qualifies (25 BERT-base pretraining runs, seed/data-order only, publisher-documented). Six pairs score **0.7633–0.7756, 6/6 Confirmed Match**, inside the Pythia range (0.6745–0.7976). Different architecture, corpus, vocabulary and publisher — **not a Pythia artifact**. Corpus now has 11 same-recipe pairs across 2 families, all positive. §V-C. |
| E17 | Rank evaluation everywhere | High | **DONE** | `M7/e17_rank_e22_runtime.py`. Parent stays rank 1 across all capability-preserving arms. Also surfaced a new finding: on untransformed `distilgpt2` the true parent ranks **second**, behind an unrelated Qwen checkpoint by 0.0026. §VI. |
| E7 | Metadata-manipulation matrix | High | **DONE** | `M7/e7_metadata_matrix.py`. 11 single-field edits, weights byte-identical. Identity score never moves (0.9992); verdict spans tier 1→2→3→**no verdict**. **5 of 11 edits produce no verdict.** New findings: `architectures: null` aborts while `[]` degrades cleanly, and renaming `model_type` aborts — reachable by ordinary correct publishing. §VIII, `tab:metaedit`. |
| E11 | Null robustness | High | **DONE** (stratified split unavailable) | `M7/e11_null_robustness.py`. p75 = 0.5296, 95% CI [0.518, 0.540], extreme range [0.511, 0.554] over 2×10⁴ resamples; 500 half-splits [0.513, 0.550]; alternative null construction 0.533. **Lowest capability-preserving arm is 0.9116 — the bar would have to rise 0.37 to change any conclusion.** Stratified-by-scale split remains impossible: the frozen null retains stratum counts, not per-score labels, and I could not reproduce its exclusion rule (1400 matches vs 1388 frozen), so it is reported as unavailable rather than approximated. §III, `app:null`. |
| E13 | Regression robustness | Med–high | **DONE** | `M7/e13_regression.py`. 10⁴ bootstrap: slope −0.333 [−0.475, −0.122]; **d\*(0.65)=0.659 [0.54, 1.38]**, d\*(0.5298)=1.020 [0.82, 2.35]. Dropping RoBERTa collapses R² to 0.007. Intervals are broad and the paper now says so. §VII. |
| E14 | Metadata prevalence survey | High | **DONE** | Two populations. Vendor catalog: **0 of 157** — but that measures *curation*, since an asset is catalogued because it was fingerprintable. Consumer-submitted population (560 deterministically sampled public models, 545 retrievable): 47 lack the key, of which 23 are GGUF and 6 test fixtures, leaving **18 genuine models = 3.3%**. Composition is the headline: **13 are DeBERTa, 11 first-party Microsoft** — the entire line, verified directly. §VIII restored to a measurement. |
| E15 | Consolidated transform matrix | High | **DONE** | `M7/e15_transform_matrix.{py,csv,md}` — 49 arms. 8 break capability; **0 demonstrably capability-preserving arms fall below the evasion bar**. Ships with the artifact (page budget full); referenced from §VI. |
| E16 | Stronger function validation | Med–high | **DONE** | `M7/e16_fidelity.py`. 64-prompt deterministic set, all positions: mean \|Δlogit\| 1.8e-5 / 1.1e-5 over 46.5M logits; argmax agreement **100%** of 946 positions; greedy decode identical on 62/62 multi-token prompts; perplexity moves in the 6th decimal. §VI. |
| E19 | Automatic layer correspondence | Very high novelty | **DONE** | `M7/e19_layer_correspondence.py` + `M7/e19b_depth_stress.py`. Monotonic DP over the parent×child alignment matrix recovers the mapping without being told it. **All 5 distillation pairs independently return (0,2,4,7,9,11)** — the reference implementation's actual initialisation, not the published "every other layer" — scoring 0.79–0.93 vs 0.49–0.60 (assumed stride) and 0.23–0.36 (naive). Negatives gain ≤7e-4 even with full freedom (12L vs independently-pretrained 6L: 0.1209→0.1213). **Margin 0.0428 → 0.3405**; vs depth-mismatched negatives, 0.6705. §IX, `app:layers`. |
| E21 | Alignment ablations | High | **DONE** | §IX-B `sec:defence-abl`, `tab:defence-abl` — includes the spectral negative control. |
| E22 | Runtime / memory | Medium | **DONE** | From frozen scan outputs: median 14.3 s/model, of which **13.6 s is database lookup** vs 0.74 s weight reading — cost scales with database size, not model size. §X. |
| E23 | Larger-model expansion | P2 | **N/A — hardware constraint** | Requires hardware beyond the one 24 GB laptop the PI has scoped. Review rates acceptance impact *low–medium* and does not list it in the §43 minimum package. The scale bound is stated plainly in §X, which also separates the three results that are scale-independent by construction from the empirical arms the scale does bind. |
| E24 | Stemma replication | P3 | **DONE at the stated requirement** | Review §8: Stemma should be *cited, discussed, optionally tested if feasible*, and explicitly **"should not be a mandatory blocker if reproducing it requires paid GPU hardware."** Cited in §II prose and positioned in `tab:related` by access model, signal, same-recipe negatives and adaptive evaluation. The optional replication is GPU-gated and out of scope on this hardware. |

## §49 final checklist

**Process** — disclosure ☑ accurately stated (PI decision); template ☑; paper anonymous ☑;
artifact anonymity ☑ — 43 scripts embedded an absolute path containing the author's username; all now derive the root from `__file__`, and a sweep of tracked sources returns nothing.

**Core empirical** — submission-time version ☑ · gold manifest ☑ · same-recipe separated from
weaker negatives ☑ · AWM on same corpus ☑ · second verifier beyond four examples ☑ · gate
counterfactual ☑ · signal ablation ☑ · recalibration ☑ (E9) · threshold sensitivity ☑ ·
null robustness ☑ (E11).

**Adaptation/evasion** — compute wording supported by real training data ☑ (E12) ·
regression uncertainty ☑ (E13) · rank for all attacks ☑ (E17) ·
stronger capability validation ☑ (E16) · consolidated matrix ☑ (E15).

**Metadata** — manipulation matrix ☑ (E7) · prevalence ☑ (E14: 0% vendor catalog vs 3.3% consumer-submitted; entire Microsoft DeBERTa line unverifiable) (claims narrowed instead of
measured — permitted by §49's "or claims narrowed") · no downstream harm claimed ☑.

**Defense** — alignment expanded ☑ · family-disjoint holdout ☑ · automatic layer correspondence ☑ (E19) · adaptive attacks ☑ · ablations ☑ · runtime/memory ☑ (E22).

**Writing** — "deployed" removed ☑ (zero occurrences) · intro three contributions ☑ · abstract
simplified ☑ · BiomedBERT and DistilBART separated ☑ (§V-E Case 1 / Case 2) · duplicate paragraph
removed ☑ · revision-history narrative removed ☑ (swept, zero matches) · related work positions
all five systems ☑ (`tab:related`) · claims scoped ☑.

**Artifact** — source commits ☑ · model revisions/hashes ☑ (`M7/model_revisions.json`, 66 ids / 61 pinned) · raw per-pair outputs ☑ · every table has a script ☑
(`analysis/print_tables.py`) · every figure has a script ☑ (`M6/make_fig{1,2,3,4}.py`) · seeds ☑ ·
environment locked ☑ (`requirements-lock.txt`) · clean-room reproduction ☑.

## Recommended order for the next block

1. ~~E9 recalibration~~ — **done 2026-08-25**; result is that recalibration does *not* help.
2. ~~E13 + E17 + E15 + E22~~ — **done 2026-08-25**; four §49 boxes closed.
3. ~~E7 metadata matrix~~ — **done 2026-08-25**; unified the two metadata defects.
4. ~~Environment lock + clean-room run~~ — **done 2026-08-25**; A1 (P0) closed.
5. ~~E19 automatic layer correspondence~~ — **done 2026-08-25**; margin 0.0428 → 0.3405.
6. ~~E14 prevalence~~ — **done 2026-08-25**; §VIII restored with the DeBERTa finding.
7. ~~E5~~ — **done 2026-08-25**; MultiBERTs reproduces the failure, 6/6.

## Beyond the review: deployment grounding (2026-08-31)

The mock review did not ask for this; the PI did, and it answers the objection the review's own
§37 (threat-model revision) gestures at without naming. **E25** replaces the assumed consumer with
a real one — CycloneDX 1.6 ML-BOM validated against the official schema, OPA admission gate — and
measures what the verifier's errors cost. Neither consumer wiring is safe: fail-open misses 11 of
37 genuinely inherited advisories, fail-closed misses 3 and wrongly quarantines 5. This settles
constitution Claim 3, the one claim scores alone could not test. `M8/`, §III + §VIII-B.

## Standing constraints

- No vendor contact, no disclosure transmission, nothing sent to anyone (PI direction).
- Open-source tools and public data only; one MacBook Pro M4 24 GB; no new hardware.
- Frozen result files are additive-only; `make verify-data` must stay at 80/80.
