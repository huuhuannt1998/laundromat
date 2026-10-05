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
| E7 | Metadata-manipulation matrix | High | **DONE** | `M7/e7_metadata_matrix.py`. 10 edits (9 single-field) plus the unmodified row, weights byte-identical. Identity score never moves (0.9992); verdict spans tier 1→2→3→**no verdict**. **5 of 10 edits produce no verdict.** New findings: `architectures: null` aborts while `[]` degrades cleanly, and renaming `model_type` aborts under the default (no model-hosted code); 1 of 545 sampled configs carries an unregistered value (`M7/e14c_composition.json`). §VIII, `tab:metaedit`. |
| E11 | Null robustness | High | **DONE** (stratified split unavailable) | `M7/e11_null_robustness.py`. p75 = 0.5296, 95% CI [0.518, 0.540], extreme range [0.511, 0.554] over 2×10⁴ resamples; 500 half-splits [0.513, 0.550]; alternative null construction 0.533. **Lowest arm verified capability-preserving is 0.9116; the remap∘permutation composition on a true derivative reaches 0.8864, so the bar would have to rise > 0.35 to change any conclusion** (CORRECTIONS.md #35). Stratified-by-scale split remains impossible: the frozen null retains stratum counts, not per-score labels, and I could not reproduce its exclusion rule (1400 matches vs 1388 frozen), so it is reported as unavailable rather than approximated. §III, `app:null`. |
| E13 | Regression robustness | Med–high | **DONE** | `M7/e13_regression.py`. 10⁴ bootstrap: slope −0.333 [−0.475, −0.122]; **d\*(0.65)=0.659 [0.54, 1.38]**, d\*(0.5298)=1.020 [0.82, 2.35]. Dropping RoBERTa collapses R² to 0.007. Intervals are broad and the paper now says so. §VII. |
| E14 | Metadata prevalence survey | High | **DONE** | Two populations. Vendor catalog: **0 of 157** — but that measures *curation*, since an asset is catalogued because it was fingerprintable. Consumer-submitted population (560 deterministically sampled public models, 545 retrievable): 47 lack the key, of which 23 are GGUF and 7 test fixtures, leaving **17 ordinary models = 3.1%**. Composition is the headline: **11 are DeBERTa, 9 first-party Microsoft**, run end to end in `M1/results/mpk_crashes_deberta.jsonl`. *(Corrected 2026-09-25 from 6 fixtures / 18 / 3.3% / 13 / 11; CORRECTIONS.md #32–#33.)* §VIII restored to a measurement. |
| E15 | Consolidated transform matrix | High | **DONE** | `M7/e15_transform_matrix.{py,csv,md}` — 49 arms. 8 break capability; **0 demonstrably capability-preserving arms fall below the evasion bar**. Ships with the artifact (page budget full); referenced from §VI. |
| E16 | Stronger function validation | Med–high | **DONE** | `M7/e16_fidelity.py`. 64-prompt deterministic set, all positions: mean \|Δlogit\| 1.8e-5 / 1.1e-5 over 46.5M logits; argmax agreement **100%** of 946 positions; greedy decode identical on 62/62 multi-token prompts; perplexity moves in the 6th decimal. §VI. |
| E19 | Automatic layer correspondence | Very high novelty | **DONE** | `M7/e19_layer_correspondence.py` + `M7/e19b_depth_stress.py`. Monotonic DP over the parent×child alignment matrix recovers the mapping without being told it. **All 5 distillation pairs independently return (0,2,4,7,9,11)** — the reference implementation's actual initialisation, not the published "every other layer" — scoring 0.79–0.93 vs 0.49–0.60 (assumed stride) and 0.23–0.36 (naive). Negatives gain ≤7e-4 even with full freedom (12L vs independently-pretrained 6L: 0.1209→0.1213). **Margin 0.0428 → 0.3405**; vs depth-mismatched negatives, 0.6705. §IX, `app:layers`. |
| E21 | Alignment ablations | High | **DONE** | §IX-B `sec:defence-abl`, `tab:defence-abl` — includes the spectral negative control. |
| E22 | Runtime / memory | Medium | **DONE** | From frozen scan outputs: median 14.3 s/model, of which **13.6 s is database lookup** vs 0.74 s weight reading — cost scales with database size, not model size. §X. |
| E23 | Larger-model expansion | P2 | **DONE** | The N/A was an inherited assumption, not a measurement. MPK extracts each model to a feature record and scores the records, pinning fp16, so peak is ONE model; the real blocker was our own `M4/p_tilde_arm.py` holding both models in fp32 (51.5 GB at 6.9b). Rewritten to spill one model at a time, **pythia-6.9b vs -deduped ran in 6.6 min at 10.9 GB peak** and is called **Confirmed Match, tier 1, identity 0.8211** — the highest on the ladder. Ladder now 70m–6.9b (100×). 3 of 4 published labels failed weight verification and were excluded, incl. pythia-2.8b-deduped being a near-copy of pythia-2.8b (cos 0.996). See `M9/RESULTS.md`, `jrn_01M1CAYP7EBXD7ZCZBRQEYBVEZ`. |
| E24 | Stemma replication | P3 | **DONE at the stated requirement** | Review §8: Stemma should be *cited, discussed, optionally tested if feasible*, and explicitly **"should not be a mandatory blocker if reproducing it requires paid GPU hardware."** Cited in §II prose and positioned in `tab:related` by access model, signal, same-recipe negatives and adaptive evaluation. The optional replication is blocked by ABSENT CODE, not hardware: no public implementation of the method exists (GitHub search returns nothing). Corrected 2026-08-31 — it had been recorded as GPU-gated. |

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

**Metadata** — manipulation matrix ☑ (E7) · prevalence ☑ (E14: 0% vendor catalog vs 3.1% consumer-submitted; entire Microsoft DeBERTa line unverifiable) (claims narrowed instead of
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

## S&P 2027 cycle-2 external review, applied 2026-09-08

Source: `sp2027-cycle2-reviews/review_laundromat.md`. Verdict was "close to submittable on the
science, not yet submittable as an S&P paper", blocking on disclosure, two ground-truth
inconsistencies, and an unevaluated defence. Every item below was applied to the manuscript and,
where it moved a number, logged in `CORRECTIONS.md` (#22-#31).

**Critical**

- C1 disclosure ☑ **partially** — `disclosure/DISCLOSURE_REPORT.md` and `disclosure/EMAIL_DRAFT.md`
  written; channel taken from the vendor's own `SECURITY.md` at the pinned commit (GitHub private
  vulnerability reporting, fallback `oss-security@cisco.com`, escalation `psirt@cisco.com`). The
  Ethical Considerations section is rewritten to the CFP's second option: a report exists, the
  channel is named, it has **not** been transmitted, and transmission is scheduled before the
  rebuttal deadline. **☐ NOT SENT. Transmission needs PI authorisation.** Until it is sent and
  dated, the paper states a plan, which the CFP permits, and not a completed disclosure.
- C2 BERT+RoBERTa ☑ (#22) — all four locations; both tables' captions say they jointly form the
  pooled fourteen; the count 11/14 was correct and is unchanged.

**Major**

- M1 Table 3 ground truth ☑ (#24, #25) — rebuilt from `M8/e25b_results.json` "consumer"
  population, n 37 → 32, fail-open missed 11 → 8, fail-closed quarantined 5 → 8; per-row appendix
  with a label grade on every row (21 primary, 11 secondary, none inferred). Bio_ClinicalBERT
  settled as cross-lineage on its own card quote.
- M2 "two of eleven" ☑ (#23) — corpus-wide 3 of 12; the distillation-only 2 of 11 kept and
  labelled as a sub-population.
- M3 defence evaluation ☑ (#27, #28) — Table 4 gains a DP column and the two rejected
  derivatives; `M7/e31` shows alignment does **not** recover either, and §10.4 bounds the claim to
  equal-width children; `M7/e32` adds the ladder column to Table 11. ☐ **Still open**: no
  compute-spending adaptive adversary. Stated as a limitation, not closed.
- M4 merging capability ☑ (#29) — instrument was invalid twice over (headless BERT child; an
  `AutoModel` round-trip that both stripped the LM head and rewrote `architectures`). Re-run in
  `M9/e29b`; the 11-16 nat costs and the 0.5349 score were round-trip artifacts.
- M5 missing prior work ☑ (#30) — eleven entries added, every field verified before writing.
- M6 recalibration reproducibility ☑ (#26) — `M7/e9b_recalibration_frozen.json`; all seven numbers
  reproduce exactly, plus a sensitivity arm.
- M7 denominator drift ☑ — §5.1 defines the four populations once and every rate names one.
- M8 threat model ☑ (#31) — `M9/e30` runs the M1 adversary in both directions; §3.1 names the M1
  publisher as the adversary of record; §3.4 gains a "what it buys" column.
- M9 appendices ☑ **partially** — 19 appendix sections down to 12, Figure 4 dropped, pointer-only
  appendices merged. ☐ Table 6 and Table 10 are **still in the appendix**: the body sits at exactly
  13 pages with no slack on p13, and promoting them would mean deleting measured evidence. Their
  load-bearing rows are now named in body prose instead.

**Minor** — Table 4 caption fixed ☑ · [7] now AAAI 2026 ☑ · [10] DOI verified and pages added ☑ ·
[12] DOI verified ☑ · constitution cited by file path at the pinned commit ☑ · Fernandez et al.
cited for the QK/OV invariance in §10.3 ☑ · O0 recovery reduced to one appendix sentence ☑.

**Prose** — em-dashes in the body 24 → 0 · four meta-voice occurrences removed · narrated revision
history removed · eight slogans replaced by declaratives · "rather than" 39 → 30 · abstract
rewritten with a population on every rate.

**Still open for the PI**

1. Authorise sending the disclosure, then fill the date into the report, the e-mail and the ethics
   section. Nothing has been sent.
2. Decide whether to spend the compute for one κ > 0 adaptive attack on the alignment signal
   (review §8, one 125M-160M model, one family). It is the only requested experiment not run.
3. Re-run `make check-version` on 2026-11-16 and update §4 if upstream has moved past `a75007d`.
4. Decide whether the Ethical Considerations section should be marked as an appendix. It prints on
   p13-14; the CFP excludes it from the page limit but also asks that text past p13 be marked.

## Standing constraints

- No vendor contact and no disclosure transmission without PI authorisation. The report and the
  covering e-mail exist in `disclosure/` and have **not** been sent.
- Open-source tools and public data only; one MacBook Pro M4 24 GB; no new hardware.
- Frozen result files are additive-only; `make verify-data` must stay at 94/94 (was 80/80 when this
  line was written, then 88; six artifacts were appended 2026-09-08).
