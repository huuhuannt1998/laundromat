# LAUNDROMAT — experiment checklist

Every item is marked against an artifact on disk, not from memory. `[x]` = run, artifact
verified. `[ ]` = not run. `[~]` = run but superseded or scoped.

Sources: `research_design_detailed.md` (the pre-registered design) for items 1–7;
adversarial review rounds 1–4 for item 8. Constraint honoured throughout: no access
requested from anyone, public models and open-source tools only, one MacBook Pro M4, CPU.

---

## 1. Stage 0 — substrate and gate G0

- [x] **Acquire the oracle** — MPK v1.1.0 @ `a75007d` + 1.4 GB fingerprint DB (184 parquet, 39 families) · `M1/RESULTS.md`
- [x] **Determinism check** — 8/8 byte-identical repeats · `M2/RESULTS.md`
- [x] **Baseline pair corpus** — 21 pairs, ground-truth labelled · `M1/results/pairs.jsonl`, `pairs2.jsonl`
- [x] **Database-wide scans** — top-k ranking behaviour · `M1/results/scans.jsonl` (7)
- [x] **Gate G0: ≥12 tier-3 positives** — **13** · `M1/results/tier3_positives.jsonl` (15 rows)
- [x] **Gate G0: ≥8 same-recipe P~ pairs** — 11 · `M4/p_tilde_arm.jsonl`, `family_generality.jsonl`
- [x] **Gate G0: ≥3 architecture families** — 5 · `M4/family_generality.jsonl`
- [x] **Frozen null distribution** — n=1388, p75 = 0.5298 (the evasion bar, fixed before any transform) · `M2/results/null_frozen.json`

## 2. Stage 1 — T1 soundness (the primitive before any adversary)

- [x] **False positives, same-recipe independents** — 4/4 Confirmed Match · `M4/p_tilde_arm.jsonl`
- [x] **Cross-family generality** — 5 families; signal fails in 4, verdict fails in 1 · `M4/family_generality.jsonl`
- [x] **False negatives, weight-decided derivatives** — 2/12 rejected, 4 weak · `M1/results/tier3_positives.jsonl`
- [x] **Gate-override case** — BiomedBERT from-scratch → Confirmed at tier 1 · `M5/sweep_bert.jsonl`
- [x] **Vendor benchmark labelling defect** — diagnostic pair labelled positive · `M1/RESULTS.md`

## 3. Stage 2 — gate G1, the κ=0 frontier

- [x] **X1a** MLP intermediate permutation · `M2/results/gate_*.jsonl`
- [x] **X1b** GQA-aware head permutation · same
- [x] **X2** RMSNorm ↔ linear rescaling · same
- [x] **X3** OV-circuit invariance · `M4/adaptive_ov.json`
- [x] **X7** QK invariance · `M4/adaptive_qk.json`
- [x] **M1t** config metadata edit (arch_hash) · `M2/results/gate_*.jsonl`
- [x] **M3t** tokenizer-config edit (family_hash) · same
- [x] **M2t** vocabulary remap — exact to 0.00e+00, 99.96% of ids moved · `M2/results/m2t_attack.jsonl`
- [x] **Q1** quantize round-trip (int8/int4) · `M2/results/q1_baseline.jsonl`
- [x] **N1emb** calibrated embedding noise · `M2/results/sweep_embed_*.jsonl`
- [x] **X5** anchor-targeted attack at O1 · `M2/results/anchor_attack_*.jsonl`
- [x] **Composition** M2t ∘ X1a ∘ M1t ∘ M3t · `M2/results/m2t_composed.jsonl`
- [x] **Exactness harness** — all exact transforms verified in fp32 · `M2/results/exactness_*.json`
- [x] **Gate G1 resolved → EXIT 3** — no transform reaches the bar

## 4. Stage 3 — the derived frontier (replaces exhaustive enumeration)

- [x] **Permutation-fraction sweep in the gate arm** — 7 points, R² 0.9999 · `M2/results/frac_sweep_*.jsonl`
- [x] **Signal decomposition closes** — σ_id slope = w_WVC × WVC slope, ratio 0.9998
- [x] **Single-signal destruction bounds** — WVC→0: ≥0.7898; EAS→0: ≥0.6393; both: 0.4299
- [x] **Non-negativity caveat quantified** — negative-EAS floor 0.2793 stated as an assumption
- [~] **Alg-2 exhaustive lattice** — *deliberately not run*. Superseded by the decomposition: any composition's score follows from component deltas, so enumeration was never the right argument.

## 5. Stage 4 — transfer (reframed)

- [x] **Decomposition transfer** — 3 architecture families, 10× scale, residuals ≤1e-4 · `M2/results/decomp_transfer.jsonl`
- [x] **Non-gated MLP support** — GPT-NeoX, exactness re-verified at 2.563e-03 · `M2/pythia_arm.py`
- [~] **Original Stage 4 (≤20 frozen compositions @1.4B)** — *not run as designed*. The decomposition answers the same question with 4 points instead of 20 and explains why rather than showing more failures.

## 6. Stage 5 — cross-substrate

- [x] **Second oracle (arXiv 2502.00706), full run** — 4 trials: FP, TN, FN, TP · `M1/oracle/.../runs/tester_1787192127.csv`

## 7. Stage 6 — T3 defence

- [x] **LAP + column normalisation** · `M4/lap2.json`
- [x] **Expanded, bug-corrected evaluation** — 17 pairs · `M4/defence_expanded.json`
- [x] **Statistics under project convention** — n=5v9, margin 0.5908, two-sided p=0.000999 · `M4/defence_expanded_stats.json`
- [x] **Adaptive attack, QK circuit** — exact, drives QK alignment to 0.216 while MLP alignment holds at 1.000 · `M4/adaptive_qk.json`
- [x] **Adaptive attack, OV circuit** — approximate (2.5–3.7% ppl), corroborating only · `M4/adaptive_ov.json`
- [x] **Laundering invariance** — 0.99688 clean vs 0.99688 laundered; 0.8807 without column norm · `M4/lap2.json`

## 8. κ>0 pricing (M5) — not in the original design

- [x] **RoBERTa family sweep** — 7 children · `M5/corpus_distance.jsonl`
- [x] **BERT family replication** — **7 of 7 complete** (last 2 recovered 2026-08-21) · `M5/sweep_bert.jsonl`
- [x] **From-scratch displacement anchor** — 4 independent Pythia runs · `M5/displacement_reference.json`
- [x] **Verified lineage per model** — evidence recorded per child · `M5/lineage.json`
- [x] **Pooled regression + confusion** — recall 8/9, specificity 1/3 · `M5/analyze_all.json`
- [x] **2 lost BERT rows** (`imdb`, `squad`) — **RECOVERED 2026-08-21.** Both downloaded on
      retry. `imdb` σ_id 0.8248 tier 2, `squad` σ_id 0.9998 tier 2, both Confirmed Match, both
      true positives. As predicted when they were dropped, neither changes a conclusion — but
      they **double the positive class (2 → 4)**, so the BERT arm's asymmetry is now measured on
      twice the evidence: **sensitivity 4/4 = 100%, specificity 1/3 = 33%** (TP 4, FP 2, TN 1,
      FN 0). **Completing the arm *weakens* the displacement predictor, and I first recorded
      the opposite.** On the paper's actual quantity — the pooled regression over *true
      derivatives only*, across both parent families — R² falls from **0.780 (n=9) to 0.6927
      (n=11)**. The 0.8366 I first wrote was the correlation across all seven BERT rows
      *including non-derivatives*, a different population; see `CORRECTIONS.md` #15. The
      qualitative claim survives (embedding 0.693 still beats overall 0.560) and the
      load-bearing number barely moves: the rejection crossing goes from 56.1% to **54.7%** of
      a from-scratch run. The fit *quality* is genuinely lower and the paper now says so.
      · `M5/sweep_bert.jsonl` (7/7), `M5/sweep_bert_final.log` · `jrn_01M0KMV0YCYYFWXE0V709Q9HWA`

## 9. Statistics and ablations

- [x] **Preregistered contrasts C1–C5 + Holm** · `analysis/prereg_final.json`
- [x] **Shape-stratified C5, powered** · `analysis/c5_final.json`
- [x] **TOST equivalence for C4** — FAILS at the predeclared δ=0.05
- [x] **Mixed-effects model** — ICC 0.26; strengthened rather than weakened the contrast · `analysis/mixed_effects_data.csv`
- [x] **M-5 capability battery** — LAMBADA/ARC-E/HellaSwag · `analysis/capability.json`
- [x] **Signal ablations** — flattened, WVC-only, structural-only, gate-forced tier-3 · `analysis/ablations.py`
- [x] **fp32 vs bf16 sensitivity**
- [~] **χ_comp necessity test** — vacuous: no SUCCESSes to certify
- [~] **O1→O2→O3 ablation** — closed analytically; the bound holds regardless of knowledge

## 10. Denial of verification

- [x] **Structured crash evidence** — 10 models, 8 confirmed defects + 2 controls · `M1/results/mpk_crashes.jsonl`

## 11. M3 — hub-scale claimed vs derived lineage

- [x] **Claim survey** — 3000 models · `M3/claim_survey.jsonl`
- [x] **Declared-parent verification** — 24 claims, 21 gate-decided · `M3/verify_claims.jsonl`

---

## STILL OPEN

### Would change what the paper can claim

- [x] **1. O0 weight recovery — DONE 2026-08-21.** Norm-preserving randomisation of a known
      fraction *p* of non-embedding, non-norm projection rows isolates the positional signal
      exactly: at every *p* the other four signals read 1.0000 to four decimals while the
      positional signal tracks 1−*p*. Recovered **w = 0.2097 ± 0.0004** against a published
      0.21, **0.14% relative error**, from scores alone. OLS σ_id = 0.9999 − 0.2093·p,
      R² = 0.999998. Upgrades the paper's conjecture to a demonstrated O0 attack.
      · `M2/o0_weight_recovery.py`, `M2/results/o0_weight_recovery.jsonl`, `o0_recovery_stats.json`
      · paper: new §VI-D, plus abstract, §I, §III, §XI
- [x] **2. Held-out defence split — DONE 2026-08-21.** Applied the pre-registered split by
      parameter scale: develop ≤160M (9 pairs), hold out ≥360M (5 pairs). Midpoint threshold
      **0.5633** fit on development only (margin +0.5908), applied **unchanged** to held-out:
      **5/5 correct**, 2 derived and 3 negative, held-out margin **+0.7588** — wider out of
      sample than in development. Converts an admitted in-sample weakness into an out-of-sample
      result. · `M4/heldout_split.py`, `M4/heldout_split.json` · paper: §IX

- [x] **11. O0 recovery of the anchor weight, and the range defect it exposed — DONE
      2026-08-21.** Complementary construction to item 1: randomise a fraction *p* of
      **embedding rows only**, norms preserved, so the anchor signal moves alone and the other
      four stay pinned at 1.0000. Recovered **w_EAS = 0.3600 ± 0.0001** against a published
      0.36 — **0.006% error**. With item 1 restated on the same estimator
      (**w_WVC = 0.2100 ± 0.0001, 0.002% error**), the attack now recovers **0.57 of the unit
      identity weight** from returned scores alone.

      **The estimator turns out not to need linearity.** Item 1 was analysed by regressing
      σ_id on *p*, which works only because that signal happens to be near-linear in *p*.
      The anchor signal is **not**: at *p* = 0.125 it reads 0.9347 where linearity predicts
      0.8750, and at *p* = 1.0 it reads 0.5013 where linearity predicts 0. The ratio estimator
      w = Δσ_id/Δs is exact at every *p* regardless (s.d. 0.0001) because it never references
      *p*. The adversary therefore need not know or control how a signal responds to the
      perturbation — only isolate it. This is why item 1's estimate improves from 0.14% error
      to 0.002% on identical data.

      **Second, unplanned finding.** Driving the embedding to fully random directions floors
      the anchor signal at **0.5013**, not 0 — the value an affinely-rescaled cosine
      (1+cos)/2 returns as cos → 0 in high dimension. So the *highest-weighted* signal has a
      usable range of [0.501, 1.000], half its nominal range. It contributes an unconditional
      **0.1805** to σ_id for any pair of same-dimension models, related or not: **27.8% of the
      0.65 weak-match bar and 24.1% of the 0.75 high-confidence bar, granted before any
      evidence of derivation is considered.** Additivity holds to 3e-5, reconfirming the M2
      linear model. **Falsification test:** if the floor is real, nothing in the corpus should
      sit below it — scanned every result artifact, **1,170 anchor-signal observations, zero
      below 0.5013**, corpus minimum equal to the floor itself. The floor survives.
      · `M2/o0_eas_recovery.py`, `M2/results/o0_eas_recovery.jsonl`, `M2/O0_RECOVERY.md`
      · `jrn_01M0KMJF6SGX9ZN45D1DKQ3XVQ` · paper: extends §VI-D, abstract, §V

- [x] **12. Signal floors and the true reachable set — DONE 2026-08-21. Corrects a published
      bound in our own favour's opposite direction.** The anchor-signal floor found in item 11
      raised the question the §VI bound depends on: *can* a signal be driven to zero? The bound
      assumed yes for all five. Driving each signal to its floor with a targeted perturbation
      and then applying all five at once:

      | signal | weight | measured floor | reaches 0? |
      |---|---|---|---|
      | embedding anchor | 0.36 | **0.4987** | no |
      | positional | 0.21 | 0.0034 | yes |
      | embedding-norm dist. | 0.19 | 0.0684 | yes |
      | layer-energy profile | 0.16 | **0.4836** | no |
      | norm-layer | 0.08 | **0.6861** | no |

      **Three of five never approach zero.** They carry 0.60 of the weight and leave an
      irreducible **0.3118**. All five at once gives a measured σ_id = **0.3255** against 0.3250
      predicted by summing wᵢ × floorᵢ (residual 1e-5). That is the minimum attainable identity
      score for a model of this shape — measured, not extrapolated to an unreachable limit.

      **This makes our own attack harder.** The paper's bound said destroying the two
      top-weighted signals suffices to clear the 0.5298 evasion bar (floor 0.4299). With real
      floors that pair bottoms out at **0.6102 — it does not evade**. Enumerating all 32
      subsets, **no pair suffices**; the minimal sufficient sets are anchor+positional+norm-dist
      (0.4332) and anchor+positional+layer-energy (0.5276). The bar is still reachable (all-signal
      floor sits 38.6% below it) so §VII's conclusion stands, but at a broader transform than we
      claimed. Logged as `CORRECTIONS.md` #16. A second seed also put the anchor floor at 0.4987,
      below the 0.5013 I had called a corpus minimum an hour earlier — `CORRECTIONS.md` #17.

      **And the gate makes all of it moot.** Every arm, including the all-signal model whose
      σ_id is 0.3255 — *half the weak-match threshold* — returns tier 1, pipeline 1.0, and the
      verdict **Confirmed Match**, the top of the ladder. Randomised embeddings, randomised
      projections, flattened norms, rescaled layer energy, randomised norm layers; on every
      weight signal the tool measures it is barely related to the reference; and the verdict does
      not move, because nothing touched `config.json`. This is the sharpest demonstration of the
      gate defect in the project.
      · `M2/signal_floors.py`, `M2/results/signal_floors.jsonl` · `jrn_01M0KNK6JNPRTVWQ852KPQ67JF`
      · paper: §VI bound subsection rewritten

### Would strengthen, cheap

- [x] **3. `ffn.lin1` selector fix — DONE 2026-08-21.** Added DistilBERT's `ffn.lin1.weight`
      to the matrix selector. Both lost trials recovered: `bert→distilbert` (derived) and
      `distilbert/distilroberta` (unrelated negative), restoring the negative class to full
      strength. · `M4/defence_distill.py`
- [x] **4. Distillation layer correspondence — DONE 2026-08-21, AND IT REVERSED ONE OF OUR OWN
      RETRACTIONS.** Matching child layer *i* to parent layer *2i* (the initialisation these
      models actually use) makes all three distillation pairs separate cleanly: **0.5148–0.5987
      against a negative ceiling of 0.2679**. Under naive index-to-index matching two of three
      fall into the negative range — a property of the matching, not the method.
      **These stride-matched values reproduce `defence_full.json` exactly (diff 0.00e+00).**
      The §IX paragraph retracting them as the output of a row-subsampling bug was therefore
      **wrong, and the retraction is itself now retracted in the paper.** The discrepancy was
      layer correspondence all along.
      Consequences: defence set grows from 14 same-depth pairs to **18 covering all three
      derivation types**; margin +0.2469, AUC 1.000, two-sided p = 4.6e-5; held-out split
      re-run with distillation included gives a tighter threshold (0.3913) and still
      **5/5 correct**. · `M4/defence_distill.py`, `M4/defence_distill.json`,
      `M4/defence_final_stats.json` · paper: §IX rewritten, abstract, §XI

### Already measured — writing only, no runs

- [x] **5. pythia-1.4b/-deduped added to the false-positive table — DONE 2026-08-21.** σ_id
      **0.7976**, EAS 0.9838, WVC 0.1023, tier 1. It is the pair in the vendor's *own published
      benchmark*, and its identity score is above the 0.75 line, so on the example the vendor
      itself chose the weight signals fail independently of the gate. Three of five pairs now
      clear 0.75 on weights alone, not two. · paper: §V-A, Table I
- [x] **6. Same-shape WVC coverage gap reported — DONE 2026-08-21.** The positional signal is
      undefined on **13% of shape-compatible pairs (2/15)** despite matching dimensions, so
      shape mismatch is not the only cause of its absence. Strengthens the argument: the signal
      that tests inheritance is sometimes simply unavailable. · paper: §V-D

### Author-only, not experiments

- [ ] **7. Transmit the coordinated disclosure.** Venue-blocking at IEEE S&P.
      `M4/DISCLOSURE_DRAFT.md` predates four of the eight crash instances and understates the case.
- [x] **8. Verify 12 references against an external index — DONE 2026-08-21.** The skill's
      Stage A-G backends are not importable in this environment, but `manubot` is, so every
      reference was resolved against arXiv/Crossref directly. **10 of 10 arXiv entries resolved;
      author lists and titles match the bib exactly.** The two entries that differed were exactly
      the two with *no* author list, now filled from the resolved records (`mpt2025` -> Nikolic,
      Baluta, Saxena; `modelscodes2024` -> Zhao et al., 9 authors, and upgraded from a bare arXiv
      note to its ASE 2024 proceedings entry with DOI 10.1145/3691620.3695271). The last
      unresolved entry, `hfsupplychain2026`, was located by Crossref bibliographic query:
      TOSEM 2026, doi 10.1145/3776739, Stalnaker et al. **All 12 references now carry verified
      author lists.** Two things worth recording: Crossref splits "Di Penta" as
      family="Penta"/given="Massimiliano Di" and was corrected by hand; and `reef2025` resolves
      to its 2024 arXiv preprint while we cite the ICLR 2025 proceedings version -- an apparent
      year mismatch that is not an error. Manuscript recompiled: 11 pages, 0 errors, no
      undefined citations. · `refs.bib` (audit note in header)

      **Follow-on: the citation gate's BLOCK was diagnosed, not worked around.**
      `verify_citations.py` first reported `manifest_missing`, which was misleading — the RKA
      manifest is intact with all 12 members active. The script reads the manifest from a
      *file*, which `rka writer sync --output` produces; once exported, every structural check
      passes (0 unresolved, 0 case mismatches, 0 unregistered, 0 unused). The **real** and only
      blocker is that no reference carried a persisted validation attestation
      (`validation_not_current` ×12). All 12 were queued through `validate_reference`; a worker
      is live and processing them. Three RKA `lit_` records that still held placeholder author
      strings — `(TOSEM)`, `(see arXiv 2409.09368)`, `(see arXiv 2502.00706)` — were corrected
      in the knowledge base so the authoritative source matches the verified bib, not just the
      `.bib` file. · `.planning/RKA_PROJECTION_SET.json`

      **Outcome of the 12 validation jobs: all completed, and the result is split.**
      `identity_matches` is **TRUE for 12 of 12** — every reference's identity is confirmed,
      agreeing exactly with the independent `manubot` check. But `validation.current` is
      **0 of 12**: 8 jobs ended in `error` and the 4 that completed reached only
      `LOW_CONFIDENCE` (sources confirmed: Crossref, Semantic Scholar, arXiv). So the citation
      gate still returns BLOCK. The honest reading is that **the references are verified and
      the attestation pipeline is degraded in this environment**, not that anything is wrong
      with the bibliography. Two independent methods agree on all 12. This is reported as a
      tooling limitation rather than worked around, and it is the one gate that is not green.

      **RESOLVED 2026-08-22 — root-caused to three stacked defects, 10/12 now VERIFIED.**
      The currency predicate (`manuscript_native.py:690`) needs `status == VERIFIED` *and* a
      completed retraction check. Three separate faults prevented both:

      1. **Title path hangs against a 60s budget.** The validator takes the fast DOI path only
         when the input carries the key `DOI` (uppercase, CSL convention). Otherwise it runs a
         title search against Semantic Scholar's `/paper/search`, which unauthenticated is
         severely rate-limited — measured still running at **6+ minutes**, versus **1.0s** for
         `resolve_doi`. The service caps the subprocess at 60s, so the job dies as `error`.
         Ten of my twelve original calls supplied no DOI. The service itself is correct here;
         the fault was mine.
      2. **`pyalex` missing in the `rka-worker` container.** It has the `academic` extras but
         not `writer-tools`, where `pyalex>=0.21` is declared. OpenAlex was unavailable to every
         job, so no reference could reach the **two concurring indices** VERIFIED requires.
      3. **`manubot` missing in the same container.** Without it Stage F reports `unavailable`
         and the script exits 2; `reference_validation.py:1082` treats any exit outside `{0,1}`
         as a failure and **discards a verdict that had already reached VERIFIED**. Confirmed
         inside the container: EXIT=2 with `STATUS: VERIFIED` in the audit before, EXIT=0 after.

      Fixes: added official DOIs to 9 literature records, installed `pyalex` and `manubot` in
      the worker. **Unapproved citations fell from 12 to 2.** Two metadata upgrades came out of
      it — `fernandez2024invariants` gains its ICASSP 2024 DOI, and **`mpt2025` turns out to be
      published at NeurIPS 2025**, not an arXiv preprint. That one matters: it is the project's
      secondary oracle.

      **The last two are honest limits and were left alone.** `stemma2026` — Semantic Scholar
      *does* hold the paper (paperId `6f1f8c5a…`, externalIds ArXiv 2607.25880) but never
      registered its DataCite DOI, and Stage B resolves by DOI only, so only OpenAlex confirms.
      Forcing it to VERIFIED would mean adding an arXiv-ID fallback to the validator — changing
      what "verified" means in order to approve our own reference. Not done. `cisco2026mpk` —
      a vendor blog post plus Apache-2.0 software, no DOI, in no academic index; correctly
      unverifiable. · `jrn_01M0M41EHM40KX6KDE9MQ731JZ`

      **Durable fix applied in source 2026-08-22** (`dec_01M0M4K8H0ZDH80YWWFQ8ZSFBM`). The
      whole packaging defect was one omission: `Dockerfile:48` installed
      `.[llm,academic,workspace]` and left out **`writer-tools`**, the extra where `pyalex` and
      `manubot` are declared. Added it. Verified by `pip --dry-run` inside the worker: resolves
      cleanly, *"Would install arxiv-3.0.0 … serpapi-1.1.0"*, no conflicts — and it corrects the
      `arxiv` pin (4.0.0 → 3.0.0) in the intended direction as a side effect.

      **Deliberately not rebuilt.** `rka-server` is up 28h healthy, the running containers
      already carry the packages from the live installs, and rebuilding mid-paper-push is
      disruption for zero immediate gain. The change lands on the next rebuild, observably.
      For the same reason the running container's `arxiv 4.0.0` was left alone: `arxiv` is
      queried only on the Stage B *title* path, and all ten VERIFIED references resolved by
      DOI, which never calls it.

### Blocked by tooling, not by us

- [~] **9. Ratify CB5/CB7/CB8 at v2 — REOPENED AND LARGELY RESOLVED 2026-08-22; the original
      diagnosis was wrong.** Not version-gated: **CB5 ratified at v2 today without error**
      (`mra_01M0M515PYCCRSGF8XYT63STH9`). The real guard (`manuscript_native.py:402`) refuses when
      any ratification row for a claim joins an **active** decision, whatever its version; CB5
      passed because its prior ratification pointed at a decision since superseded. The HTTP 500 is
      a separate defect — a `ValueError` precondition mapped to 500 instead of 4xx, hiding the
      message that explains it. `CORRECTIONS.md` #18.

      **The real problem turned out to be wording, not mechanism.** Checking currency before
      ratifying the remaining two:
      - **CB7** — v2 verified **exact** against the artifact of record (8 defects, 7 from 4 major
        orgs + 1 community, both controls). Correct, but blocked by the guard above: its live
        ratification points at `dec_01M0H6T351GTD8WPH7G70FBZHX`, which is active *and holds the
        correct text*, so the service asks us to supersede the very decision we want to keep. The
        escape is a same-wording successor decision — a PI act, not mine, so I stopped.
      - **CB8** — v2 is **stale**: "fourteen same-depth pairs … 0.5908 … no held-out split …
        distillation unresolved". Current: **18 pairs, margin +0.2469, AUC 1.000, p = 4.57e-5,
        held-out 5/5, distillation resolved.** Ratifying it would bind text contradicting §IX. Not
        ratified — deliberately.
      - **CB5** — I ratified it, then found its wording is **also stale**: it calls the O0 recovery
        "conjectured, not demonstrated" and prohibits "recoverable from observed scores alone",
        both falsified by our own result hours earlier. My error, logged as `CORRECTIONS.md` #19.

      Raised to the PI as blocking checkpoint `chk_01M0M57EJJ49YGFMWZHB0R1SX0` with a recommended
      route: author v3 wording for CB5 and CB8, supersede-then-ratify for CB7. **No manuscript text
      is blocked** — the drafted prose already reflects current evidence in all three cases; only
      the claim-spine records lag.

      Worth noting: `manuscript_readiness` reports *whether* a claim is ratified, never whether the
      ratified wording is still true. That gap is what let a stale ratification look like progress.
- [x] **10. Register the 8 unpinned result artifacts — DONE 2026-08-22, and it was not
      bookkeeping.** Auditing `MANIFEST-dataintegrity.txt` against the working tree found the
      manifest pinned **`M1/results/mpk_crashes.jsonl` at `d41d8cd98f00b204e9800998ecf8427e`
      — the MD5 of a zero-byte file.** That is the artifact of record behind the paper's
      denial-of-verification claim, and the manifest was certifying the *broken* version of it.
      The file was regenerated 2026-08-20 (10 models, 8 defects, 2 controls, 7214 bytes) and the
      manifest had never been refreshed. `M5/sweep_bert.jsonl` was likewise stale (5 rows → 7).
      Eight result files produced since — the O0 recovery pair, the signal floors, and the five
      defence artifacts — were absent entirely. Regenerated to **79 entries, 0 mismatches**, with
      the previous version preserved as `MANIFEST-dataintegrity.prev.txt` and a header recording
      exactly what changed, so the update is auditable rather than silent.

---

**Score: 64 of 72 run · 4 deliberately superseded (reason recorded) · 2 open, both PI decisions.**

Everything in the pre-registered design is run or superseded with the reason stated. What
remains is **not experimental**: item 7 is an author-only decision (transmit the disclosure),
item 9 is blocked by a server-side HTTP 500, and item 10 is bookkeeping. **Every experiment in
the pre-registered design has now been run or superseded with the reason stated.** No open item
blocks a claim, and none needs hardware beyond the laptop.


---

## Session log — 2026-08-21

Six open items closed. Two changed what the paper can claim, one reversed a correction we
had made in error, three were writing-only.

| # | Item | Outcome |
|---|---|---|
| 1 | O0 weight recovery | **New result.** w = 0.2097 vs published 0.21, 0.14% error, from scores alone. Conjecture → demonstrated attack. |
| 2 | Held-out defence split | **Preregistered split applied.** 5/5 correct out of sample, margin +0.7588. Removes an admitted in-sample weakness. |
| 3 | `ffn.lin1` selector | Two lost trials recovered, including an unrelated negative. |
| 4 | Distillation layer correspondence | **Reversed our own false retraction.** Stride-matched values reproduce `defence_full.json` to 0.00e+00; the earlier retraction was a misdiagnosis. Defence now covers all three derivation types, n=18. |
| 5 | Vendor benchmark pair in Table I | σ_id 0.7976, above threshold — the weight signals fail on the vendor's own example. |
| 6 | Same-shape WVC coverage gap | Positional signal undefined on 13% of shape-compatible pairs. |

Gates after all six: 11/13 pages, 0 undefined refs/cites, layout 0 BLOCK, provenance 0 BLOCK,
AI-tic 0 BLOCK. Manuscript rebuilt and copied to `LAUNDROMAT-draft.pdf`.


---

## Session log — 2026-08-21 (continued)

Three more items. One is a new result, one closed a submission blocker, one is still running.

| # | Item | Outcome |
|---|---|---|
| 11 | O0 recovery of the anchor weight | **New result.** w_EAS = 0.3600 vs published 0.36 (0.006% error); with item 1 restated on the same estimator, **0.57 of the identity weight recovered from scores alone**. Two by-products: the estimator needs **no linearity assumption**, and the highest-weighted signal is shown to have **half the range it appears to have**, granting 27.8% of the weak-match bar unconditionally. Falsified against 1,170 corpus observations — zero below the predicted floor. |
| 8 | Reference verification | **Closed.** All 12 references resolved against arXiv/Crossref via `manubot`; 10/10 arXiv entries match the bib exactly. The 3 previously-unresolved author lists are now filled, including the TOSEM entry located by bibliographic query. Two traps recorded: a Crossref surname split, and a preprint-vs-proceedings year that looks like an error but is not. |
| — | 2 lost BERT rows | **Closed.** Both recovered; BERT arm complete at 7/7. Positive class doubles to 4; **sensitivity 100%, specificity 33%**. The displacement predictor **weakens** on the full arm (R² 0.780 → 0.693 on true derivatives, n=9 → 11) — I first recorded a strengthening by comparing the wrong population (`CORRECTIONS.md` #15). The evasion price is robust: crossing 56.1% → 54.7%. The sharpest row in the paper is now `CP biomed`: largest displacement, lowest σ_id (0.5248, below even the weak-match line), trained from scratch — verdict **Confirmed Match**, because tier 1 pins the pipeline to 1.0 and the identity score is never consulted. |

The item-11 estimator result is the more general of the two: item 1 could be read as depending
on a signal that happens to respond linearly to the perturbation. It does not. Isolation alone
is sufficient, which is what makes the recovery a property of the scoring design rather than of
the particular transform we chose.


---

## Session log — 2026-08-21 (third block)

Two new experiments, one of which corrected a bound in the paper against our own interest.

| # | Item | Outcome |
|---|---|---|
| 12 | Signal floors / true reachable set | **New result, and a self-correction.** Three of five signals never approach zero (anchor 0.4987, layer-energy 0.4836, norm-layer 0.6861), leaving an irreducible 0.3118. Minimum attainable σ_id measured at **0.3255**. The §VI bound had assumed all signals reach 0; with real floors, **destroying the two top-weighted signals no longer evades** (0.6102 vs the 0.5298 bar). No pair suffices — evasion needs ≥3 signals. Conclusion survives, attack is harder. |
| — | Gate defect, sharpest form | The all-signal model — randomised embeddings, projections, norms, layer energy, norm layers, σ_id **0.3255**, half the weak-match line — still returns tier 1, pipeline 1.0, **Confirmed Match**. Destroying every weight signal does not move the verdict. |
| 8 | Reference validation jobs | All 12 completed. **identity_matches 12/12**; `validation.current` 0/12 (8 errors, 4 LOW_CONFIDENCE). References verified by two independent methods; attestation pipeline degraded here. Citation gate remains the one non-green gate. |

Corrections logged this block: **#16** (the two-signal evasion bound was wrong — measured floors
make our attack harder) and **#17** (a 0.5013 "corpus minimum" superseded by a 0.4987 observation
at a different seed within the hour).

Gates after this block: 11/13 pages, 0 errors, provenance 0 BLOCK, AI-tic 0 BLOCK, layout 0 BLOCK
(12 pass / 1 warn, chktex unavailable). Citation gate BLOCK, cause diagnosed above.


---

## Session log — 2026-08-22

| Item | Outcome |
|---|---|
| Citation gate BLOCK | **Root-caused and largely fixed.** Three stacked defects: the validator's title path hangs on Semantic Scholar's rate-limited search endpoint against a 60s subprocess budget; `pyalex` missing from the worker so OpenAlex never contributed a second source; `manubot` missing so Stage F exited 2 and the service discarded already-VERIFIED verdicts. **10/12 references now VERIFIED, current, retraction-checked**; unapproved citations 12 → 2. |
| Bibliography upgrades | `mpt2025` is published at **NeurIPS 2025** (doi 10.52202/085713-1146), not an arXiv preprint — it is our secondary oracle. `fernandez2024invariants` gains its ICASSP 2024 DOI. Nine records gained official DOIs. |
| Deliberately not done | An arXiv-ID fallback in Stage B would make `stemma2026` VERIFIED. Rejected twice over: **measured `s2.paper_by_id` at 133.6s against a 60s subprocess budget**, so it would reintroduce the very timeout failure this work removed — and for *every* arXiv reference, not one. And it means editing the verifier to approve our own citation, in a paper about a verifier that approves what it shouldn't. |
| Durable packaging fix | `Dockerfile:48` omitted the `writer-tools` extra. Added; dry-run verified. Not rebuilt — deferred to the PI so it lands observably rather than mid-push. |

Manuscript rebuilt: 11/13 pages, 0 errors, 0 undefined citations.


---

## Session log — 2026-08-22 (second block)

Took the three remaining "open" items. One was mine to finish, one was mis-diagnosed and is now
mostly solved, one is not mine to take.

| Item | Outcome |
|---|---|
| 7. Disclosure | **Rewritten, still UNSENT.** The old draft carried 4 findings and predated half the evidence. Now 6 findings ordered by operational severity, led by the denial-of-verification defect (8 models, 5 organisations, identical Pydantic error, one clean positive control, one honest exclusion) which was **absent entirely** from the previous version. Adds the gate-override result (σ_id 0.3255 → still `Confirmed Match`), the O0 weight recovery, the EAS range defect, and the measured κ>0 price. Keeps the honest "we did not break it at κ=0" framing first. **Transmission remains a PI decision — I did not send it and will not.** |
| 9. CB5/CB7/CB8 | **Mis-diagnosed, reopened, largely resolved.** Not version-gated — CB5 ratified at v2 first try. Real cause is a guard on any *active* prior ratification, plus a ValueError→500 mapping that hid the message. Then the actual problem surfaced: **CB5 and CB8 wording has drifted behind the evidence.** CB7 verified exact but deadlocked on the guard. Raised as `chk_01M0M57EJJ49YGFMWZHB0R1SX0`. |
| 10. Artifact records | **Done, and it caught a real defect.** The manifest pinned `mpk_crashes.jsonl` at the MD5 of an *empty file* — certifying the broken version of the artifact behind our most quotable claim. Regenerated: 79 entries, 0 mismatches, previous preserved for audit. |

Corrections logged: **#18** (the "confirmed version-gated" blocker that dissolved on one attempt)
and **#19** (I ratified CB5 before checking whether its wording was still true).

The through-line for both: `manuscript_readiness` tells you *whether* a claim is ratified, never
whether the ratified wording is still **true**. A stale ratification reads as progress.


---

## Session log — 2026-08-22 (third block)

| Item | Outcome |
|---|---|
| M4 **T1 dependency chain** | **Written** — `sections/08b-consequence.tex`, §VIII-B, between Denial and Defence. This was an unmet mission acceptance criterion the mission itself called *"the paper's security contribution"*, warning that without it "Oakland will treat it as an attack note". Organising argument: the failure modes are **directional** and the three consumers degrade asymmetrically — over-attachment is visible and contestable, under-attachment is invisible by construction. Three points: the tier-1 gate returns the verdict to self-report (killing the BOM's independence-from-assertion); the undetected region is occupied by benign two-stage adaptations, so licence leakage happens with no adversary; and denial of verification fails *silent*, not closed. Carries an explicit "what we did not establish" subsection and makes **no** claim about any specific vendor product pipeline. |
| Integrity manifest | Regenerated (79 entries, 0 mismatches) after finding it pinned `mpk_crashes.jsonl` at the MD5 of an empty file. |

Gates after: 12/13 pages, 0 errors, provenance **0 BLOCK**, AI-tic 0 BLOCK, layout 0 BLOCK.
Citations BLOCK on the 2 documented exceptions. Readiness BLOCK pending `chk_01M0M57EJJ49YGFMWZHB0R1SX0`.

Recorded caveat: 9 of the new section's 11 provenance markers score LOW_SUPPORT at near-zero token
overlap. Expected for an interpretive section reasoning *from* measurements rather than restating
them — but it means those markers establish **coverage, not entailment**.


---

## Paper-loop — 2026-08-22 (3 rounds, strict gate, maximal scope)

Reviewer verdict progression: **NOT READY → NOT READY → READY** (conditional on the disclosure,
which is an author action outside the loop's authority).

| Round | Found | Resolved how |
|---|---|---|
| 1 | **6 Critical, 12 Major, 2 Minor** | Intro said "four pairs" where Table I has five (C-1). §VI-E asserted the claim it had refuted three paragraphs earlier and stated its central negative result backwards (C-2). Conclusion's "thirteen of the sixteen" was supported by no table and no artifact — replaced with the verified **11/14** (C-3). The additivity-at-the-floor check was **circular** and its residual mis-stated 200× — 10⁻⁵ claimed, 1.0×10⁻² actual (C-4). The floor construction's capability was never measured and "the bar remains reachable" conflated identity with pipeline score — the exact conflation the paper indicts; **ran a new measurement** (`M2/floor_capability.py`): perplexity 164→113,210, max\|Δlogit\| 34.0, capability-preserving **false** (C-5). |
| 2 | **2 Critical, 2 Major, 6 Minor** — all introduced by round 1 | The "real models" column added to *fix* the floor error **repeated that error**: 0.5360 was a rung of our own noise ladder, printed under a heading reading "real published models". Root cause: our scan missed `wide_scans.jsonl` (1,400 real comparisons) because it nests signals under `scores`, not `signals`. Three of five cells corrected against 1,100+ genuine comparisons (N-1). Abstract asserted at O0 what the intro says needs per-signal values (N-2). 11/14 was not reconstructible because `imdb`/`squad` appeared in no table — both added (N-4). |
| 3 | **1 Major, 4 Minor** | A stale `0.5360` survived two inches from the table correcting it. Fixed, plus all four Minors. One of my fixes tripped the AI-tic gate on a PI-banned term (`additionally`) and was rewritten. |

**The reviewer was wrong once and I checked rather than complied:** M-2 alleged Table IV silently
omitted two tier-3 pairs. It does not — there are exactly 12 tier-3 records; the two named are
tier-**1**. The reviewer withdrew it. Following the pointer did surface something better:
`distilbart-cnn-12-6` returns `identity_score: null` and **all five weight signals null**, yet
`Confirmed Match` at tier 1. Now in §V-E as a stronger real-model instance of the gate finding
than any construction.

**Corrections logged this loop: #20 and #21.** #20: three of my five signal "floors" were
artifacts of an all-positive randomisation; only the anchor survives. #21: the table written to
record #20 committed #20's error. The headline survived every recomputation — **no pair of
signals suffices** to clear the bar, verified by a hostile reviewer under five distinct floor
value sets, worst-case margin 7 points.

Final: 13pp (refs p13), 0 errors, 0 overfull, 0 undefined. Provenance/AI-tic/layout **0 BLOCK**.
79 frozen files unchanged. Citations BLOCK on the 2 documented exceptions; readiness BLOCK
pending `chk_01M0M57EJJ49YGFMWZHB0R1SX0`.

---

## External review — ai-cyber-paper-reviewer + manuscript_revision_prompt_template — 2026-08-23

Two independent tools run against the post-loop manuscript, per user request. Findings verified
against the vendor's actual source repo (not taken on faith) before any fix was applied.

**Tool 2 (writing/presentation, Prompt 1 + security add-on)** caught two arithmetic defects the
internal loop missed:
- §VI's "no pair suffices" mechanism was **wrong**: I'd attributed it to the anchor's irreducible
  mass, but the binding pair (positional+embedding-norm, 0.6001) doesn't involve the anchor at
  all. Real mechanism: the two largest available reductions sum to 0.3999 against the 0.4702 the
  bar requires. Conclusion unchanged; stated reason was false and checkable in one minute.
- A stale `0.32` sat next to a table weighting to `0.3357`, printed by name two paragraphs later.
- The `0.01%` in the abstract/intro/conclusion was **never measured** — body gives 0.006% and
  0.002%. Fixed at all four sites.
- EAS/WVC/END/σ_id used in every table header, defined nowhere. Now bound in §IV.
- Three tables never referenced from text (incl. the availability evidence table).

**Tool 1 (ai-cyber-paper-reviewer, full rubric + external verification)** found something the
internal loop could not, because it read the vendor's actual source rather than only the paper's
artifacts:
- **The vendor's own benchmark contains the paper's headline false positive**, correctly labelled
  `"provenance": "none"`, with a challenge string stating the mechanism verbatim — but the shipped
  evaluation notebook scores ground truth from the bucket name (`similar`/`dissimilar`), not the
  `provenance` field, so the pair counts as a correct positive in the vendor's own metric. The
  same 111-pair benchmark calibrated the five signal weights by Cohen's *d* on those labels
  (`eas: d=2.03` vs `wvc: d=1.21` — the exact asymmetry the paper identifies as the root cause).
  **Verified directly against `benchmarks/Benchmark_All.json` and `run_benchmark.ipynb` in the
  vendor repo** before writing a word. Now §V-A's central paragraph.
- **REEF already evaluates one same-recipe-independent pair** in an appendix (CKA 0.9983 vs
  0.7632, no threshold, n=1). The paper's novelty sentence was scoped narrowly enough to survive
  literally but sat beside a defence drawn from REEF's mechanism — narrowed and REEF cited.
- **§IX was reasoning conditionally about unobserved consumers when it could falsify three
  published vendor security claims directly** — verified against `docs/constitution/` in the
  vendor repo: "fabrication... defeated by weight-level analysis, which cannot be faked" (refuted
  by the null-signal Confirmed Match), "metadata-only systems are trivially defeated" (the tool
  is metadata-only on 11/14 of the paper's own pooled corpus), and "under-inclusive definitions
  are acceptable... licensing audits... catch missed cases" (engaged, not ignored — the paper's
  errors are not predominantly under-inclusive, and a missing verdict isn't a fallback that ran).
  §IX rewritten around constructive falsification rather than hedged inference.
- `cisco2026mpk` had a mismatched title/URL, wrong author org, and an unresolvable version
  string. **Verified in the actual repo**: commit `a75007d` ("Bump version to 1.1.0"), owner
  `cisco-ai-defense`, 184 shipped parquet fingerprint files. Bib entry and §IV corrected to name
  the artifact by commit rather than an untagged version string.
- Table III (n=4) compressed to prose (already started last round; tightened further).
- §X's retraction-of-a-retraction narrative moved to a footnote — both reviewers independently
  flagged this as undercutting the section meant to persuade a reviewer a fix exists.
- §X retitled from a double-hedged title to a finding-first one.
- Dropped a frontier-scale extrapolation from Limitations (a section whose job is to bound
  extrapolation, extrapolating).
- 2/12 vs 2/15: three tier-3-excluded pairs were decided correctly by the gate and absent from
  the corpus that produces the abstract's rejection rate. Both denominators now stated.

**Gates after: 0 errors, 0 overfull, 13 total pages (body ends p12, references p13), provenance/
AI-tic/layout 0 BLOCK, 79 frozen files unchanged.**

Six items marked UNVERIFIED by the second tool were left alone rather than "fixed" on faith:
an artifact-absent perplexity anecdote (13.86→5,742), an unlocated 8-run determinism claim, a
possibly-stale permutation-floor figure (0.0157 vs recomputed 0.0159), a compare-vs-scan score
discrepancy on gpt2/gpt2-medium (0.8915 vs 0.8751), the null distribution's undisclosed strata
(89% in the 1–10B bucket vs a ≤1.4B evaluated population), and three proposed experiments (E1:
attack AWM's actual QK pipeline; E2: recompute the Cohen's-d weight calibration under corrected
labels; E4/E5: widen specificity and replicate off-corpus). None were run — they need genuine new
measurement or an admission the paper cannot currently make, and are recorded here rather than
silently dropped.

---

## Mock-review response — 2026-08-24

User provided a full mock IEEE S&P review + 22-experiment revision plan
(`manuscripts/feedback/SP2027_Recipe_or_Descent_Master_Review_and_Revision_Plan.md`). Triaged
honestly: this is a multi-week research program, not a one-session task. What follows is what
was actually done, verified against real sources rather than assumed.

**P0 items:**
- **P0-B (compsoc template)** — verified live against the actual S&P 2027 CFP (fetched
  2026-08-24): mandatory, "subject to rejection without review" if omitted. Applied. In
  verifying it I also **caught and corrected my own error across four prior rounds**: the real
  limit is 13pp main text + up to 5pp references/appendices (18pp total), not 13pp total. I had
  been mutilating the paper's content for three rounds to fit a limit that was never real.
- **P0-C (deployed → publicly released)** — applied, 8 sites + title.
- **P0-A (vendor disclosure)** — **not actioned.** Conflicts directly with the PI's standing
  instruction, "I do not want to send anything to anybody. This is for research purposes."
  Flagged, not overridden.

**Writing fixes applied:** duplicated paragraph (§32.2) removed; introduction's 5-item
contribution list reduced to 3 (§33); abstract simplified 345→293 words, precision pair
(0.14%/0.006%) removed from the abstract (§34).

**Experiments run — E6, E8, E10 (pure recomputation, zero new downloads):**
Built a 31-row pooled lineage-verified corpus from three existing artifacts. All three results
strengthen claims already in the paper by replacing "selected cases" with systematic,
full-corpus evidence:
- **E6 gate counterfactual (4 policies)** — removing the gate fixes exactly 1 of 6 false
  positives (BiomedBERT); the five pythia same-recipe pairs stay false-positive under every
  policy including weight-only, because the identity score itself is unsound on them.
- **E8 signal ablation** — removing WVC collapses specificity to exactly 0; WVC alone gives
  perfect specificity at 0.05 recall; EAS alone is the mirror image. Directly substantiates the
  paper's closing thesis sentence.
- **E10 threshold sweep** — no value in [0,1] simultaneously achieves recall=1.0 and
  same-recipe FPR=0.0; at the point FPR reaches 0, recall has already fallen to 0.42. Stronger
  than "the fixed thresholds happen to misfire" — no calibration of this score works.
- **E11 null robustness (partial)** — bootstrap CI [0.5176, 0.5392] confirms the bar isn't
  knife-edge; the 89%-out-of-population strata composition confirmed. **Blocked honestly**: a
  stratified recompute needs per-item labels the frozen artifact doesn't store.

Added as new §V-F in `sections/05-soundness.tex`. Gates: 0 errors, 0 undefined, provenance/
AI-tic 0 BLOCK, 14 total pages (compliant with the verified 18pp real limit). 80 frozen files,
0 mismatches (additive-only).

**Not attempted, and why:** E1 (latest-commit repro), E2 (gold-corpus manifest), E3 (official
AWM baseline), E4 (expanded second verifier), E5 (same-recipe negative expansion), E9
(recalibration), E12 (training-progress ladder), E18–E22 (defence expansion, automatic layer
correspondence, adaptive attacks, runtime scaling). Each needs new downloads, implementing
another paper's actual pipeline, or multi-hour compute campaigns — none are a one-session task.
Full write-up: `M6/mock_review_experiments.md`.

---

## 2026-08-24 — Figures (review §35, item "four figures"), and layout repair

The mock review's §35 asked for four figures on the grounds that the paper's core
vulnerability is currently only readable in prose. All four are built from data already
frozen in the corpus; none required a new measurement or a download.

**Fig. 1 `fig1_pipeline.pdf` — the decision path** (`sections/04-verifier.tex`,
`\label{fig:pipeline}`). Publisher-written `config.json` selects the MFI tier; tier 1 pins
the pipeline score to 1.0 and tier 2 caps it at 0.9, so the five weight signals are reached
only at tier 3. Marks the 11-of-14 pooled comparisons decided without weight evidence, and
the missing-key edge that aborts verification with no verdict.

**Fig. 2 `fig2_distributions.pdf` — identity score by true relationship**
(`sections/05-soundness.tex`, `\label{fig:dist}`). Strip plot over four classes:
derived n=24 (0.391–1.000), same-recipe independent n=5 (**0.674–0.798**), same-family n=2,
unrelated n=1388 (the frozen null). The same-recipe band sits inside the derived range and
above both decision thresholds — the overlap is the point, and no vertical line separates
the two classes. This is §V-F's "no working threshold" result made visible.

**Fig. 3 `fig3_displacement.pdf` — score vs embedding displacement**
(`sections/07-pricing.tex`, `\label{fig:disp}`). n=11 documented derivatives, fit
sigma_id = 0.8692 − 0.3327·d_emb, R² = 0.693. The region past the last observation
(d_emb = 0.8295) is shaded: the 0.65 crossing is interpolated, the 0.5298 crossing is
extrapolated. The figure carries the honesty of the reframe rather than leaving it to prose.

**Fig. 4 `fig4_symmetry.pdf` — the design rule, measured**
(`sections/09-defence.tex`, `\label{fig:sym}`). MLP input projection 0.9969 → 0.9969 under
the adaptive attack (permutation-only residual symmetry); Q/K circuit 1.0000 → 0.216
(continuous bilinear freedom). One panel showing why the substrate choice is the defence.

Each figure is referenced from body prose (`Figure~\ref{...}`), not left orphaned.

**Provenance-gate defect found while wiring the figures.** Placing the marker anywhere but
immediately above `\includegraphics` produced UNCOVERED; placing it there produced ORPHAN for
Fig. 4 alone. Root cause is in `verify_provenance.py:_next_prose`, which strips
`\cmd[...]{...}` before testing whether the governed block has any prose: Fig. 4's caption
contained no inner braces, so the entire caption was stripped to the empty string and the
marker looked orphaned. Figs. 1–3 escaped only because their captions happen to contain
`\ref`/`\texttt`/`$\sigma_{}$`. Fixed by giving Fig. 4's caption a real `\S\ref` and math
symbols — which the caption wanted anyway.

**Layout repair.** Wiring the floats surfaced seven overfull hboxes, three of them severe
(47.1pt, 45.7pt, 32.4pt) — tables spilling into the margin: `tab:defence-abl`,
`tab:pricing`'s threshold table, and `tab:related`. Fixed by shortening headers and row
labels and tightening `\tabcolsep`; no data cell was altered. Build now reports **0 overfull**.

Gates after this round: build 0 errors, 0 undefined refs, **0 overfull**; provenance
**0 BLOCK** (15 WARN, all LOW_SUPPORT token-overlap advisories); AI-tics **0 BLOCK**;
16 pages total, body ends p13, refs p14, appendix p15–16 — inside the 13 body + 5 back
matter limit. Citation gate still BLOCK on the two known-unverifiable references
(`cisco2026mpk`, `stemma2026`); revalidation re-queued.

---

## 2026-08-24 — E1 closed, E2 built (and it corrected us)

**E1 — submission-time version replication. Closed, no run needed.** The plan asks whether
our finding survives on the vendor's current release. A read-only `git fetch --tags origin`
at 2026-08-24T23:04:32Z returns `origin/main` = `a75007d5ca4aaf8df2e3f055f318557a965b93aa`,
byte-identical to our pin; `git log HEAD..origin/main` is empty; the only remote branch is
`main`; the newest tag is 1.0.0, so the 1.1.0 bump is the untagged tip. Twelve days elapsed
since that commit with no upstream change. There is no newer version to replicate on — the
artifact we measured and the artifact a reader installs today are the same object. Recorded
in §IV as a dated attestation. **Perishable:** re-check immediately before 2026-11-17; if the
tip moves, the replication must actually be run.

**E2 — gold-standard lineage manifest.** `M6/gold_lineage_manifest.json`, 31 rows, built
additively by `M6/build_gold_manifest.py` from data already frozen (80/80 md5s unchanged).
Per row: relationship class, adaptation type, ground-truth source, verbatim supporting
quotation, derivation depth, measured tier, identity score, verdict, five signals,
displacement, config-modified flag. Negatives are **not** pooled — N-SR (independent run,
same recipe, same size, n=5) is kept separate from N-FAM (same architecture family,
independently pretrained, n=2), which is the distinction the whole paper turns on.
Labels are graded: a relationship stated in the child's own model card is **primary**;
one resting on a release convention, a repo name, or only on an associated publication is
**secondary**. Card text was fetched, not assumed. Outcome: 19 primary, 11 secondary,
1 excluded.

**The audit found a defect in our own data.** `M1/results/tier3_positives.jsonl` recorded
`nreimers/BERT-Tiny_L-2_H-128_A-2` as `provenance="distillation"`, `source="model card"`.
That card is 161 bytes and reads, in full: *"This is the BERT-Medium model from Google: <url>.
A BERT model with 2 layers, 128 hidden unit size, and 2 attention heads."* It makes no
distillation claim, and it misnames the model — BERT-Medium — against both the repo name and
its own config (2 layers / 128 hidden is BERT-Tiny). The label was not supported by the source
we cited for it, so the pair is **excluded from both analysis sets** rather than quietly
reclassified. Same defect class we document in the vendor's own benchmark, found in our data
because we went looking.

**Numbers that changed** (all corrected in the manuscript): derivative pairs attempted
15 → **14**; weight-decided tier-3 derivatives 12 → **11**; end-to-end rejection 2/15 → **2/14**;
weak matches among tier-3 4 → **3**; WVC undefined 5-of-12 → **5-of-11**; Fig. 2 derived class
n=24 → **n=23** (range unchanged at 0.3914–0.9998, and all five same-recipe points still fall
inside the derived range, which is the figure's whole point). Fig. 2 is now generated by
`M6/make_fig2.py` directly from the manifest so the figure and the corpus cannot drift apart.

**A qualification that runs against us, now stated in the paper** (new §V-B `sec:corpus`).
Of the eleven weight-decided derivatives, five carry primary labels and **none of those five
is rejected**; both rejections — xtremedistil-l6-h256 at 0.4211 and dynamic_tinybert at
0.3914 — carry secondary labels. Their teachers are named in the publications their cards
cite, so the relationships are not in doubt, but the evidence is one step weaker than for the
five that pass. The appendix table now marks every secondary-label row with a dagger.

**A mechanism I tested and rejected.** Rejection does *not* track hidden-size compression:
xtremedistil compresses hidden 768→256 and BERT-Tiny 768→128, but `dynamic_tinybert` has the
*same* hidden size as its parent and the lowest score in the corpus. No clean geometric
predictor exists in this data, so no such claim is made.

Also resolved three depth conflations (`distilbert-base-uncased-distilled-squad`,
`all-distilroberta-v1`, `dynamic_tinybert`): in each the corpus parent is an ancestor, not the
immediate parent. Derivation is transitive so the rows stay true; depth is now recorded rather
than the two being conflated.

**E12 (Pythia training-progress ladder) is running** — `M6/e12_pythia_ladder.py`, nine
log-spaced checkpoints of pythia-160m compared against step143000, on a token axis calibrated
at 2,097,152 tokens/step (143000 steps = 299.9B tokens, matching Pythia's published budget).

Gates after this round: build 0 errors, 0 undefined refs, 0 overfull; provenance 0 BLOCK;
AI-tics 0 BLOCK; 16 pages (body ends p13); 80 frozen files unchanged. Citation gate remains
BLOCK on `cisco2026mpk` and `stemma2026` — both explained, neither fabricated.

**E12 result (2026-08-25).** Nine checkpoints of `pythia-160m` against step143000 of the same
run, token axis exact at 2,097,152 tokens/step.

| step | after fork | σ_id | had weights decided |
|---|---|---|---|
| 1 000 | 99.3% | 0.4526 | **Not matched** (also below the 0.5298 evasion bar) |
| 2 000 | 98.6% | 0.5378 | **Not matched** |
| 4 000 | 97.2% | 0.7571 | High-confidence |
| 8 000 | 94.4% | 0.8052 | High-confidence |
| 16 000 | 88.8% | 0.7356 | Weak |
| 32 000 | 77.6% | 0.7614 | High-confidence |
| 64 000 | 55.2% | 0.8003 | High-confidence |
| 96 000 | 32.9% | 0.8578 | High-confidence |
| 128 000 | 10.5% | 0.9957 | Confirmed |

Three findings. (1) **All nine are pinned at pipeline score 1.0, tier 1, Confirmed Match** —
`config.json` is byte-identical across revisions, so nine verdicts on a lineage that is literally
one training run are reached without consulting weights, and on two of them the weight evidence
would have said *not matched*. This is the paper's thesis on ground truth nobody can dispute.
(2) **The compute mapping, and it runs against our framing.** Falling below 0.65 by honest
continued training needs ≥98.6% of a full run after the fork, so 55% of from-scratch
*displacement* ≈ 98% of from-scratch *compute*. The units are not interchangeable and on this
axis **the vendor's cost claim is substantially correct** — §VII now says so. (3) **The identity
score is a poor distance proxy:** flat within 0.736–0.858 while training-after-fork varies 3.0×,
and non-monotone (0.8052 at step8000 → 0.7356 at step16000, *falling* as the models grow closer).

Caveats carried into the text: continued training is on the same distribution, so 98.6% is an
upper bound on the honest-training cost; n=1 model at 160M. Verdict labels are derived from
`pipeline_score` via a mapping established empirically over 152 prior comparisons (Confirmed
0.9000–1.0000 at tiers 1–2; High-Confidence 0.7672–0.9688; Weak 0.6502–0.7378; Not Matched
0.3481–0.6378), not read directly. `d_embedding` is reported but **not** pooled with the n=11
regression — the convention normalises by the earlier checkpoint's norm and early embeddings sit
near initialisation, giving values above the √2 anchor.

**Page-limit repair.** Adding §V-B, E12 and four figures pushed the body from 13 pages to ~14.3.
Recovered without losing a result: Fig. 3 and Fig. 4, `tab:fam`, `tab:additivity`, the O0-recovery
detail, the corpus-audit detail and the second-substrate per-pair data all moved to the appendix;
the coefficient-recovery result was **demoted** in body and conclusion per review §29 ("keep the
result, but demote it"); two paragraphs of revision-history narration were cut from §X (the
limitations themselves are unchanged and still stated in full); prose tightened across §V, §VI,
§X, §XII. **Body now ends p13**, Ethical Considerations opens p14, back matter runs p14–16 —
inside 13 + 5.

Also corrected stale counts the E2 audit invalidated: abstract, introduction and conclusion said
"two of twelve genuine derivatives" and now say "two of eleven".

Gates: build 0 errors / 0 undefined / 0 overfull; provenance 0 BLOCK; AI-tics 0 BLOCK; 80/80
frozen files unchanged.

---

## 2026-08-25 — E3 (AWM baseline) and E4 (second substrate)

### E3 — the official AWM baseline. The strongest result of the session.

The review calls this "probably the most important new experiment": a reviewer will ask why we
built our own assignment-based signal instead of testing the actual prior method. Answer: we
tested it, with its authors' code.

**Feasibility, checked first.** The official implementation exists at `LUMIA-Group/AWM`
(confirmed from the arXiv abstract page), cloned at commit
`bc20ff8e63cec57f5da422ae065686ced275e76d`. Training-free, ~5–15 s/pair on CPU.

**Scope, established by testing rather than assumed.** AWM extracts attention weights by key
patterns requiring `.layers.`/`.h.`/`.blocks.` with LLaMA/Qwen/GPT naming. BERT- and
RoBERTa-style encoders (`encoder.layer.N.attention.self.query.weight`) and BART's
`model.encoder.layers.*` match none of them — verified directly: bert-base-uncased vs
distilbert-base-uncased returns *"No recognizable attention weights found"*. **The prior method
cannot score most of our corpus**, which is itself a legitimate answer to the reviewer question.
A version we extended to encoders would be testing our code, not theirs, so we did not.

**Harness bug caught before it became a false claim.** Saving checkpoints via `AutoModel` strips
the `transformer.`/`model.` prefix AWM's regex needs, so GPT-2 initially looked unsupported.
Materialising via `AutoModelForCausalLM` fixes it. Unnoticed, we would have reported that AWM
cannot score GPT-2 — false.

**Result, on matched pairs (6 documented descendants, 4 independent same-recipe):**

| method | descendants | independent same-recipe | AUC |
|---|---|---|---|
| AWM (Wq·Wk) | 0.1012–0.9977 | 0.1163–0.3007 | 0.833 |
| studied tool (σ_id) | 0.4526–0.9994 | 0.6745–0.7856 | 0.750 |

Neither separates the classes. The AUC gap is **two discordant orderings out of 24** and is not
statistically distinguishable at this n — no claim that AWM is better.

**The sharpest number.** AWM gives the four Pythia base-vs-deduped pairs z-scores of **40.2,
66.1, 85.7, 106.0** against its own negative reference, where genuinely unrelated pairs score
**0.12 and 1.41**. Those pairs share *no weights* — separate training runs — and the method reads
them as overwhelmingly related. And one true ancestor (the 0.7% checkpoint from E12) scores
0.1012, **ranking below all four non-relatives**.

**Two points in AWM's favour, both reported:** it separates same-recipe from unrelated by one to
two orders of magnitude, and it tracks training progress monotonically across the E12 ladder
(0.1012 → 0.3824 → 0.9957) where σ_id is non-monotone.

**What this does for the paper.** It converts the claim from *"this vendor's tool confuses recipe
with descent"* into *"weight-similarity lineage verification confuses recipe with descent as a
class, including the current academic method run with its authors' own code."* That removes the
single-target generality objection, which was the most dangerous one on the list. Written into
§V (`sec:awm`) and the conclusion.

Caveats: AWM's z uses a negative-pair std hardcoded at 0.0028 with a source comment saying it
"was not specified", so raw similarity is primary; gpt2 vs gpt2-medium returned UNSUPPORTED
(dimension mismatch); n is small per class.

### E4 — second substrate expanded, and it is unstable at n=300 prompts

Ran the black-box tester over 6 parents × 6 candidates at 300 prompts. Compared against the
earlier 4-parent run, **three of the four overlapping pairs flipped verdict**:

| candidate | run A (4 parents) | run B (6 parents) | |
|---|---|---|---|
| pythia-160m-deduped | FP p=0.0376 | TN p=0.1019 | flipped |
| pythia-410m-deduped | TN p=0.1136 | TN p=0.6775 | |
| distilgpt2 | FN p=0.0825 | TP p=0.0139 | flipped |
| SmolLM2-135M-Instruct | TP p=0.0222 | FN p=0.1912 | flipped |
| pythia-70m-deduped | — | **FP** p=0.0086 | |
| pythia-1b-deduped | — | TN p=0.5520 | |

What replicates is the *error directions*, not the specific pairs: both runs produce a false
positive on an independently-trained same-recipe pair and a false negative on a genuine
derivative. That is what you expect if the classes are not separable and boundary verdicts are
near-random — consistent with the paper's thesis, but it means **no single verdict from this
substrate can be leaned on**. A 2000-prompt run is in flight to test whether verdicts stabilise
at higher power; whatever it shows, the instability at n=300 will be reported.

### Artifact work

Figures 1, 3 and 4 had been built by inline code that was never saved — a real gap for the
artifact requirement "scripts generating each figure". Now reconstructed as
`M6/make_fig1.py`, `make_fig3.py`, `make_fig4.py`, all regenerating from frozen data.
`make_fig3.py` reproduces the paper's fit exactly from `M5/*.jsonl`: σ = 0.8692 − 0.3327·d,
R² = 0.6927, last observation 0.8295, crossings 0.6587 and 1.0199. `make_fig4.py` reads
`M4/adaptive_qk.json` directly (MLP alignment 1.0000 flat; QK 1.0000 → 0.7565 → 0.2160).

Gates: build 0 errors / 0 undefined / 0 overfull; provenance 0 BLOCK; AI-tics 0 BLOCK; 80/80
frozen files unchanged. **Known trivial item: body is 13 pages plus a few lines.** Roughly a
page was recovered today while *adding* E3, E12 and the corpus audit; the remainder is a
pre-submission trim, not a research blocker.

### E4 at adequate power — the instability was underpowering, and the failure is worse than at n=300

Re-ran the black-box tester at **2000 prompts** (6 parents × 6 candidates):

| candidate | class | A (300) | B (300) | **C (2000)** |
|---|---|---|---|---|
| pythia-70m-deduped | N-SR | — | FP 0.009 | **FP 0.0094** |
| pythia-160m-deduped | N-SR | FP 0.038 | TN 0.102 | **FP 0.000097** |
| pythia-410m-deduped | N-SR | TN 0.114 | TN 0.678 | TN 0.193 |
| pythia-1b-deduped | N-SR | — | TN 0.552 | **FP 0.011** |
| distilgpt2 | derived | FN 0.083 | TP 0.014 | TP 0.00075 |
| SmolLM2-135M-Instruct | derived | TP 0.022 | FN 0.191 | TP <1e-7 |

At adequate power the tester identifies **both** genuine derivatives correctly and names a parent
for **three of the four** independently-trained same-recipe pairs, at p down to 9.7e-5. Recall is
perfect; specificity fails precisely on the same-recipe class.

**More statistical power makes the false positives worse, not better** — the opposite of the
natural assumption, and the key point. The output agreement between same-recipe models is
*genuine*, so additional evidence buys more confidence in a relationship that does not exist in
the weights. Underpowering at n=300 concealed the failure; it did not cause it. Both the
instability and the resolution are reported (`tab:substrate`).

### Where this leaves the paper's central claim

Three methods, sharing no mechanism, all confuse recipe convergence with descent:

1. **The deployed tool** — weight signals behind a metadata gate.
2. **AWM** (weight-based, ICLR 2026, authors' own code) — z = 40.2–106.0 on pairs sharing *no
   weights*, against 0.12–1.41 for genuinely unrelated pairs.
3. **The black-box tester** (output agreement, no weights read) — 3 of 4 same-recipe pairs
   false-positive at p down to 9.7e-5.

The claim is now class-level rather than vendor-specific, and it is evidenced on both sides of
the weight/behaviour divide. Carried into the abstract, §V (`sec:awm`, second-substrate
subsection) and the conclusion.

Gates: build 0 errors / 0 undefined / 0 overfull; provenance 0 BLOCK; AI-tics 0 BLOCK; 80/80
frozen files unchanged; 17 pages total, body 13 + a few lines.

---

## 2026-08-25 — E18 (defence expansion) and artifact assembly

### E18 — 30 pairs, the missing negative class, and a family-disjoint holdout

Corpus expanded from 17/18 to **30 scored pairs** (`M7/e18_defence.py`, `M7/e18_defence.json`).
Class coverage: continued-pretraining 7, distillation 5, fine-tune 5, multitask 1, same-recipe
independent 8, **same-family independent 2 (new)**, unrelated 2. The same-family class — models
of the same architecture trained independently from scratch — was entirely absent before and is
the class the review specifically asked for.

**Ordering is still perfect: AUC = 1.0000**, derived 0.4941–0.9998 against negatives
0.1169–0.4513.

**The qualification, and it runs against us.** Adding the same-family class drops the margin from
0.2469 to **0.0428** — roughly six-fold. The binding case is `legal-bert-base-uncased` at 0.4513
against a lowest derivative of 0.4941; every other negative sits at or below 0.2679. The ordering
holds without error, but one harder same-family negative would close the gap. Now stated in §IX,
the conclusion and the abstract rather than quoting a margin measured without that class present.

**Label verified, not assumed.** Because legal-bert became the margin-limiting case I re-read its
card: *"LEGAL-BERT-BASE is the model referred to as LEGAL-BERT-SC ... a model trained from scratch
in the legal corpora ... using a newly created vocabulary"*, and the card notes the
checkpoint-derived LEGAL-BERT-FP variants are released elsewhere. Our N-FAM label stands.

**Family-disjoint holdout**, replacing the parameter-count split the review criticised (that split
put every distillation pair in development). Develop on BERT/cased-BERT/mBERT/RoBERTa — which
retain the hardest negative — giving threshold 0.4727. Applied unchanged to BLOOM, GPT-2, Pythia,
Qwen and SmolLM2: **TP 4, FN 0, TN 8, FP 0**, held-out margin +0.2469. A first split left the
held-out arm with *no negatives*, testing recall only; rebalanced so both arms carry both classes.

**Two harness defects found.** (1) DistilBERT names its MLP up-projection `ffn.lin1.weight`, not
`intermediate.dense.weight`; the extractor inherited from `M4/defence_expanded.py` misses it and
returns a silent `null`, so all three BERT-family distillation pairs scored as unscorable. Fixed —
`bert→distilbert` then returns **0.5255, exactly matching** the paper's stride value in
`M4/defence_final_stats.json`. (2) `gpt2` vs `pythia-160m` is genuinely unscorable: GPT-2's Conv1D
`c_fc` and GPT-NeoX's `dense_h_to_4h` are transposed relative to each other. Recorded as an
architectural limit of the comparator, not a bug.

`tab:defence` regenerated from the E18 data (30 rows) so the table matches the claim.

### Artifact

`ARTIFACT.md` + `Makefile` + `analysis/print_tables.py`. Targets: `verify-data`, `smoke`,
`check-version`, `reproduce-{manifest,figures,tables,ladder,awm,substrate,defence}`.
`print_tables.py` answers "where did Table X row Y come from?" by printing each table's numbers
from its frozen source.

**A reproduction hazard worth the artifact's space.** `M4/defence_expanded.json` matches layers
index-to-index and exists only to enlarge the arms for the *statistical contrast*. Read alone it
appears to show the defence **failing**: index matching puts distilgpt2 at 0.2337 below a
same-recipe negative at 0.2679, a margin of **−0.0342**. The paper's ordering claims correctly
come from the stride-matched `M4/defence_final_stats.json`, and `M4/defence_distill.json` records
both readings side by side (`lap_index` 0.2900 vs `lap_stride` 0.5255). A reviewer opening the
wrong file would reach the opposite conclusion, so ARTIFACT.md spells the distinction out.

Gates: build 0 errors / 0 undefined / 0 overfull; provenance 0 BLOCK; AI-tics 0 BLOCK; 80/80
frozen files unchanged; 17 pages total. **Body is ~1 page over the 13-page limit** — every
addition today was a mandatory review item, and further trimming now means demoting content
rather than tightening prose.

---

## 2026-08-25 — Page-budget decision (PI delegated: "you decide what is best for S&P")

The body had to lose about a page. Rather than pick by taste, I used the review's own ranking of
the contributions.

**Cut 1 — the denial-of-verification section, 76 → 36 lines.** Review §30: *"Keep it, but change
its role... without a prevalence survey, it remains a localized robustness bug."* We did **not**
run E14, the metadata-prevalence survey that would have elevated it, so by the review's own
standard it is the weakest remaining body contribution. Kept in the body: the mechanism, the count
(8 models = 5 distinct publisher-side defects), the sharpest point (all four rungs of the standard
DAPT release are unverifiable as published), the cost comparison, and the honest no-prevalence
caveat. Moved to `app:denial`: the control-row discussion and the cross-harness precision
footnote. One paragraph was **deleted outright** as redundant — it restated the repair caveat that
§X already makes in full.

**Cut 2 — the O0 weight-recovery result, demoted out of the abstract and conclusion.** Review §29:
*"keep the result, but demote it... should not be a headline S&P contribution... remove the
precision numbers from the abstract unless space is abundant."* Space is not abundant. The result
remains in §VI with its mechanism, and its detail is in `app:o0`; it no longer appears in the
abstract or the conclusion.

**Not cut, and why.** The consequence section (§VIII-B) stays — the review identifies it as what
establishes security relevance. E12 and the AWM baseline stay: both are new, both are what turn a
single-vendor bug report into a class-level claim. The defence stays despite its now-thin margin,
because the margin being thin is itself a reported finding.

Net effect: every mandatory review item is in the paper, and what left the body is what the review
itself ranked lowest.

**Final state.** Build 0 errors / 0 undefined / 0 overfull. Provenance 0 BLOCK, AI-tics 0 BLOCK,
80/80 frozen files unchanged. 17 pages: body 13 pages plus two lines, back matter p14–17 (inside
the 5-page allowance). Stale-claim sweep clean — no surviving "eighteen pairs", "two of twelve",
or "held-out arm is five".

Remaining known items, neither blocking: the two-line body overrun (a camera-ready trim), and the
citation gate at 10/12 on `cisco2026mpk` (a GitHub artifact with no DOI, verified absent from
every index) and `stemma2026` (ordinary preprint indexing lag).

---

## 2026-08-25 — E9, E13, E15, E17, E22 (all pure recomputation, no downloads)

### E9 — recalibration does NOT fix it *(the most important of the five)*

Signals frozen, only the combination refit, on 28 audited pairs (18 derivatives vs 10 hard
negatives: 8 independent same-recipe runs, 2 from-scratch same-architecture BERTs).

| fit | AUC |
|---|---|
| vendor coefficients (no fitting) | 0.600 |
| logistic refit, leave-one-pair-out | **0.622** |
| non-negative refit, leave-one-out | 0.606 |
| logistic, in-sample ceiling (biased) | 0.822 |
| family-disjoint split, refit | **0.125** (vendor 0.688) |

Honest recalibration buys 0.022 AUC. On unseen families the refit *inverts*, scoring worse than
chance and far worse than the vendor's own coefficients. Two sharper points: **the best single
signal (0.778) beats the full weighted combination (0.600)**, and nlf is *anti*-correlated with
descent (0.222) — exactly what the clamping defect predicts.

This is the review's second branch: the signals lack lineage specificity. It forecloses the
"you found a calibration bug" rebuttal using the vendor's own signals. §V-F.

*Defect caught en route:* my first non-negative fit hadn't converged and returned uniform weights
(AUC 0.383). I would have reported a broken optimiser as a finding. Replaced with `scipy.nnls`.

### E13 — the displacement fit is real but imprecise

10⁴ bootstrap: slope −0.333, 95% CI [−0.475, −0.122] (excludes zero). But the crossings inherit
wide intervals: **d\*(0.65) = 0.659 [0.54, 1.38]**, d\*(0.5298) = 1.020 [0.82, 2.35] — the latter
reaching ~2× the from-scratch anchor. Theil–Sen slope is shallower (−0.249); **dropping the
RoBERTa family collapses R² to 0.007**, so one of two families carries the relationship. Quadratic
adds only 0.016 R². The review said "if the interval is broad, say so" — it is, and §VII now does,
with the CIs in the threshold table.

### E17 — rank, for every arm

Parent stays rank 1 across all capability-preserving arms, which is why the zero-compute result is
negative. But the scan mode is weaker than that implies: on **untransformed `distilgpt2`** the true
parent ranks **second**, behind an unrelated Qwen checkpoint by 0.0026, and three further targets
(two Pythia, BLOOM) return `gpt2-medium` as top candidate. Cross-family resemblance beats true
descent before any adversary. §VI.

### E15 — consolidated transform matrix

49 arms across all M2 experiment files → `M7/e15_transform_matrix.{csv,md}`. Eight break the
capability criterion; exactly one falls below the evasion bar, an anchor-noise injection at 8×
relative scale whose fidelity was not recorded in that arm. **Zero demonstrably
capability-preserving arms reach the bar** — the paper's central negative claim, confirmed in one
view. Ships with the artifact rather than as a table; the page budget is full.

*Classification bug caught:* my first version treated "no fidelity record" as "capability
preserved", which reported one capability-preserving evasion that does not exist. Absence of a
measurement is not evidence of preservation; such arms are now `unmeasured` and excluded from the
tally. Documented as an artifact hazard.

### E22 — runtime

From the frozen scan outputs: median **14.3 s per model**, of which **13.6 s is database lookup**
and 0.74 s is reading the target's weights — an 18× ratio. Cost scales with database size, not
model size. §X. *(My first summary line inverted this ratio; corrected before it reached the
paper.)*

Gates: build 0 errors / 0 undefined / 0 overfull; provenance 0 BLOCK; AI-tics 0 BLOCK; 80/80
frozen unchanged; **18 pages — exactly at the 13+5 cap**, body ends p13.

`REVIEW_CHECKLIST.md` updated: 18 items DONE, 11 open/partial.

---

## 2026-08-25 (cont.) — E7, E11, E16, A1

### E7 — the metadata matrix unifies the two "separate bugs"

Weights byte-identical; only `config.json` varies, one field at a time. The identity score reads
**0.9992 in every arm that returns anything** — nothing about the weights changed. Everything that
moves is metadata.

| edit | tier | pipeline | verdict |
|---|---|---|---|
| (unmodified) | 1 | 1.0000 | Confirmed |
| `architectures` renamed / empty list / `max_pos`+1 | 2 | 0.9000 | Confirmed |
| `num_hidden_layers`+1 | 3 | 0.9992 | High-confidence |
| `architectures` **absent** / **null** | — | — | **no verdict** |
| `model_type` **renamed** / **absent** | — | — | **no verdict** |
| `hidden_size`+64 | — | — | **no verdict** |

**5 of 11 single-field edits produce no verdict, none touching a weight.** Two new findings:
an *empty* `architectures` list degrades cleanly to tier 2 while absent and `null` both abort — so
the degradation path exists and isn't reached, and null-handling is a defect distinct from the
missing default. And **renaming `model_type` aborts**, which is reachable by ordinary correct
publishing (new architectures set custom `model_type` routinely), not only by omission.
§VIII + `tab:metaedit`.

### E11 — the evasion bar is not threshold engineering

p75 = 0.5296, 95% CI **[0.518, 0.540]**, extreme range [0.511, 0.554] over 2×10⁴ resamples; 500
disjoint half-splits [0.513, 0.550]; an alternative null construction gives 0.533. The
lowest-scoring capability-preserving transform sits at **0.9116** — the bar would have to rise by
**0.37** before any of them counted as an evasion. The conclusion holds across the entire
plausible range.

*Still unavailable:* the stratified-by-scale and same-family-vs-cross-family splits. The frozen
null keeps stratum counts, not per-score labels, and I could not reproduce its exclusion rule
(1400 scan matches against 1388 frozen). Reported as unavailable rather than approximated.

### E16 — exactness, validated harder

64-prompt deterministic probe (prose, code, multilingual, numerals, degenerate), scored at every
position: mean |Δlogit| **1.8e-5** and **1.1e-5** over 46.5M logits, next-token argmax agreement
**100%** of 946 positions, greedy decoding identical on **62/62** multi-token prompts, perplexity
moving in the sixth decimal. *Reporting bug caught:* the first version printed "62/64", which
reads as two mismatches; two prompts were single-token and skipped. Now "62/62 measured
(2 skipped)".

### A1 — clean-room reproduction, and it earned its keep

Staged to a fresh tree outside the working directory; every headline number reproduced exactly.
It caught two real defects first:

1. **21 scripts hardcoded an absolute repo root.** From a reviewer's checkout they would silently
   read and write *the author's* tree. My first clean-room attempt therefore partly measured the
   original directory and would have reported a false pass. All scripts now derive the root from
   `__file__` (`LAUNDROMAT_ROOT` overrides).
2. **Four manifest entries live under `M2/work/`** — the vocabulary-remap attack artifact, not
   scratch. Excluding working directories from a copy makes `verify-data` fail on four files.
   Documented in ARTIFACT.md.

Also added: `requirements-lock.txt` (analysis environment pinned; the verifier keeps its own
uv-managed environment, which we deliberately do not override).

**Checklist: 22 DONE, 7 open/partial.** Gates: 0 errors / 0 undefined / 0 overfull; provenance
0 BLOCK; AI-tics 0 BLOCK; 80/80 frozen; 18 pages, body ends p13.

---

## 2026-08-25 (cont.) — E19, E19b, E5

### E19 — the layer correspondence is recoverable, not something we must be told

The distillation result previously depended on *knowing* child layer *i* came from parent layer
2*i*, which is circular: the detector needs part of the derivation to verify the derivation.
Replaced by a monotonic dynamic program over the full parent×child alignment matrix — layer order
assumed, stride not.

| | index (naïve) | stride (assumed) | **DP (recovered)** |
|---|---|---|---|
| 5 distillation pairs | 0.2337–0.3609 | 0.4941–0.5987 | **0.7918–0.9266** |

**All five independently recover (0,2,4,7,9,11)** — the initialisation the reference
implementation actually performs, not the "every other layer" of its published description. The
method recovers ground truth it was never given.

### E19b — the stress test that makes E19 credible

Maximising over mappings could flatter everything, and every E19 negative was equal-depth where
the search has no freedom. So: three **depth-mismatched** negatives, 12L parent vs 6L child, where
the DP may choose any six of twelve.

| negative | stride | DP | lift |
|---|---|---|---|
| bert-base-uncased vs independently-pretrained 6L BERT (same width) | 0.1209 | 0.1213 | +0.0004 |
| bert-base-uncased vs distilroberta-base | 0.1167 | 0.1172 | +0.0005 |
| roberta-base vs distilbert-base-uncased | 0.1172 | 0.1179 | +0.0007 |

Largest lift any negative receives: **7×10⁻⁴**, against 0.3–0.4 for true derivatives. The search
is discriminative, not merely optimistic. **Margin 0.0428 → 0.3405** (0.6705 against
depth-mismatched negatives).

*Bug caught:* E19 computed all 11 pairs then crashed writing output (`np.int64` not
JSON-serialisable). Results recovered from the log rather than re-running 13 minutes of model
loading; the cast is fixed so it reproduces.

*Stale claim caught:* the conclusion and abstract still quoted the pre-E19 margin of 0.0428.
Both corrected; §IX now states both figures with the condition attached.

### E5 — the same-recipe failure is not a Pythia artifact

Written inclusion rule applied **before** collecting: separate pretraining runs of the same
architecture at the same size under the same procedure, differing only in seed and/or data order,
no shared checkpoint, independence stated by the publisher and never inferred from naming.

Google MultiBERTs qualifies — 25 BERT-base runs differing only in seed and data order. Six pairs
from seeds 0–4:

| identity | verdict |
|---|---|
| 0.7633–0.7756 (mean 0.7698) | **6/6 Confirmed Match**, all tier 1 |

Inside Pythia's range (0.6745–0.7976), on a suite sharing *nothing* with it — different
architecture, corpus, vocabulary, tokenizer, publisher, year. The corpus now holds **eleven
independent same-recipe pairs across two architecture families, every one called a match.**

### Page budget

Each of these pushed the paper to 19 (over the 18 cap) and each was offset by a cut chosen on the
review's own ranking rather than convenience: the O0 weight-recovery subsection reduced in §VI
(review §29 says it "should not be a headline contribution"), the defence-ablation detail and the
gate-counterfactual per-policy numbers moved to the appendix. Back to **18, body ending p13**.

**Checklist: 24 DONE, 5 open.** Gates: 0 errors / 0 undefined / 0 overfull; provenance 0 BLOCK;
AI-tics 0 BLOCK; 80/80 frozen.

---

## 2026-08-25 — E14, and the checklist closes

### E14 — prevalence, measured on two populations

| population | N | no `architectures` |
|---|---|---|
| vendor's own fingerprint catalog | 157 | **0 (0.0%)** |
| consumer-submitted sample (40 most-downloaded × 14 families) | 545 | 47 raw → **18 genuine (3.3%)** |

The vendor's 0% **measures curation, not prevalence** — an asset is in the catalog *because* it
was fingerprintable, which requires the fields being counted. The review proposed that frame as
"defensible" without flagging the conditioning; both populations are reported side by side.

The raw 8.6% is not reported unqualified either: of 47, **23 are GGUF repos** (not the format the
tool reads) and **6 are test fixtures**. Defensible rate 18/545 = 3.3%.

**The composition is the finding.** 13 of 18 are DeBERTa; **11 are first-party Microsoft** —
`deberta-base`, `-large`, both v2 sizes, all four v3 sizes, `mdeberta-v3-base`. Verified directly
against three configs. An entire model line from a major publisher, including one of the
most-downloaded encoders on the hub, cannot be verified. §VIII is now a measurement rather than
eight anecdotes — which is exactly what review §30 said it needed.

*Stale claims caught:* §VIII and §X both still said "we did not survey the hub". Corrected.

*Bug caught:* E14b's first run returned N=0 for all 14 families — `direction=-1` was removed from
`HfApi.list_models` in huggingface_hub 1.x, so every listing threw `TypeError` while the summary
printed "0 (0.0%)", which reads as a finding and was an API error.

### Checklist closed

**26 done · 3 N/A · 0 open.** All 44 boxes of review §49 are ticked; all ten §43 mandatory
experiments complete; both P0 blockers resolved.

The three N/A are constraint-blocked, not skipped: **P0-1** vendor disclosure (PI direction, and
the §49 requirement is that disclosure *status be accurately stated*, which §XIII does);
**E23** larger models (needs hardware beyond the 24 GB laptop; P2, low–medium impact, absent from
the §43 minimum package); **E24** Stemma (review §8 requires *cite and discuss*, both done — the
optional replication is GPU-gated and explicitly non-mandatory).

Three limits stated rather than papered over: the stratified null split is permanently
unavailable (the frozen artifact kept counts, not per-score labels, and its exclusion rule could
not be reproduced — 1400 vs 1388); E9's family-disjoint split has only 2 held-out positives, so
its 0.125 rests on 16 comparisons and the leave-one-out figure is the robust one; E14's 3.3% is a
rate within a most-downloaded sample, not a hub-wide estimate.

### Final state

Build 0 errors / 0 undefined / 0 overfull · provenance 0 BLOCK · AI-tics 0 BLOCK · 80/80 frozen ·
**18 pages, body ends p13** (13 + 5) · no author paths in tracked sources · pinned commit still
equals upstream `origin/main`.

Citation gate stays 10/12: `cisco2026mpk` (GitHub artifact, no DOI, verified absent from every
index — full 40-char SHA pinned rather than a DOI invented) and `stemma2026` (preprint indexing
lag).

---

## 2026-08-31 — E25: a real deployment for the threat model

**The gap.** §III declared an adversary's capabilities and a success criterion but never showed
the verifier *inside* anything. "A consumer runs the verifier and acts on its verdict" was one
sentence, assumed. A reviewer asking "how is this actually launched?" had no answer.

**The setup, built from tools a consumer would really use — not a scenario we invented.**
CycloneDX 1.6 defines `component.pedigree.ancestors` as *"zero or more components in which a
component is derived from"*, with `machine-learning-model` a first-class component type. That is
exactly the field a lineage verdict populates. Pipeline:

    model pair → verifier verdict → CycloneDX 1.6 ML-BOM → OPA/Rego admission gate → admit/deny

Every BOM is validated against the **official pinned schema** (`M8/bom-1.6.schema.json`); the gate
is **OPA 1.17.1**. The obligation instrumented is the one the paper's own abstract names —
scoping inherited vulnerabilities — driven by one realistic consumer action: an advisory is
published against an ancestor, and the gate asks whether each model inherits it.

**The load-bearing modelling choice, made explicit.** A match becomes an ancestor edge, a
not-matched becomes no edge — but the tool can return *nothing*, and the standard cannot represent
that. Only two readings are defensible, so both were evaluated:

| consumer wiring | missed an inherited advisory | wrongly quarantined | correct |
|---|---|---|---|
| fail-open (silence ⇒ unrelated) | **11 / 37** | 5 | 21 |
| fail-closed (silence ⇒ unknown) | **3 / 37** | 5 | 29 |

**Neither wiring is safe.** Fail-open lets eleven models escape an advisory they genuinely inherit,
including all four AllenAI DAPT rungs. Fail-closed clears the eight no-verdict cases but leaves
three — there the verifier returned a *confident wrong answer* no wiring can recover — and both
quarantine five models that inherit nothing, the recipe-convergence false positives now expressed
as blocked deployments.

This settles constitution Claim 3 ("under-inclusion is the safe direction"), the one claim scores
alone could not test because it is a claim about consequences.

**Two defects caught before anything reached the paper.**
1. *The first run measured nothing and looked like good news* — zero wrong admissions under
   fail-open. Cause: our corpus contains **zero licence-violating derivative pairs**, so a licence
   policy can never fire. Rebuilt around advisory scope, which needs no fabricated licence.
2. *A conflation that would have fabricated the failure under study.* The runner treated a
   manifest row with `measured=False` (we did not record the verdict) as "the verifier returned no
   verdict". Those are different things, and the difference is the entire experiment. Verdicts are
   now sourced from the measured result files, and genuinely unknown ones are skipped explicitly.

**A real property of the surface, found by schema validation.** `license.id` is the SPDX enum, and
`bigscience-bloom-rail-1.0` is not in it — the RAIL family, whose use restrictions explicitly bind
derivative works, can only be carried as free text in a conformant ML-BOM. The obligations most in
need of automated propagation are the ones the format expresses least well.

**Paper changes.** §III gained "The pipeline the verdict enters"; §VIII-B was rewritten around the
measurement (`tab:deploy`) and absorbed the weaker "we observed no downstream harm" framing; §VIII
and §X contracted to make room; the SPDX limit, the repair detail, the floor-construction
diagnostic and the statistical elaboration moved to appendices.

Gates: text ends p13 (limit 13), 18 pages (limit 18), abstract 248 words (<250), 0 errors /
0 undefined / 0 overfull, provenance 0 BLOCK, ai-tics 0 BLOCK, 80/80 frozen.
