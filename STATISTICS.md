# Preregistered statistics — final state

Design §11.4 fixed five contrasts, a mixed-effects model, a held-out split and an equivalence
margin **before** any T2 run. This is what happened when they were actually run. Two of the five
turned out not to be inferential at all, one was underpowered by construction until fixed, one
failed its predeclared margin, and the mixed-effects model contradicted the design's own stated
fear. Nothing here is post hoc except where explicitly labelled.

## The five contrasts

| # | contrast | result | status |
|---|---|---|---|
| C1 | S2 vs S1 — composition beats free symmetry? | σ_pipe 0.8924 vs 1.0000 (Δ −0.1076) | **exhaustive**, no p-value |
| C2 | S2 vs M1t-alone — composition beats metadata? | σ_pipe 0.8924 vs 0.9000 (Δ −0.0076) | **exhaustive**, no p-value |
| C3 | κ=0 frontier vs B-3 (scratch at matched κ) | won by construction — B-3 is empty at κ=0 | **vacuous**, no evidential weight |
| C4 | P~ vs P+ on `s_struct` — the masking claim | 0.9022 vs 0.9141, AUC 0.607, p = 0.857 | **retain H₀** (Holm p = 0.857) |
| C5 | P~ vs P+ on `s_raw`/WVC — the separability claim | 0.1633 vs 0.5203, AUC 0.881, p = 0.0239 | **reject H₀** (Holm p = **0.0478**) |

**C1–C3 are not tests.** §11.4 states that a p-value on an exhaustive enumeration is a category
error, so the Holm family is the inferential subset {C4, C5}, m = 2. The literal m = 5 correction is
also reported in `analysis/prereg_stats.json` and changes no conclusion.

**C5 survives Holm by 0.002.** At α = 0.05 with Holm p = 0.0478, a single additional discordant pair
would flip it. It should be read as *established but thin*, not as comfortable.

## C5 was underpowered by construction until fixed

WVC is only defined when parent and child dimensions match, so the contrast is meaningful **only on
the same-shape stratum**. On that stratum the original arms held n = 3 vs 3, where the minimum
attainable two-sided exact permutation p is **0.10** — the preregistered test could not reject at
α = 0.05 regardless of the data. The observed p = 0.1429 was a power floor, not a null result.

Adding same-shape pairs (Pythia `-deduped` / `-v0` independent runs; published fine-tunes) brought
this to n = 7 vs 6, minimum attainable p = **0.00117**. Only then was C5 decidable.

Two defects were found in the original arms and fixed:
1. P~ contained `pythia-160m | pythia-160m` — an **identity pair** (WVC = 1.0) belonging to the control arm.
2. Both arms pooled across shape. WVC ≈ 0.00x whenever dimensions differ — for distillation *and* for
   different-size same-family pairs alike. Near-zero there means "dimensions don't match", not "not derived".

## P+ is not one population — and that is the finding

| derivative type | WVC | in the P~ range? |
|---|---|---|
| fine-tune (SmolLM2, Qwen) | 0.9970, 0.9976 | no |
| fine-tune (BERT-SST2, bloomz) | 0.5472, 0.2460 | no |
| **continued pretraining** (SecureBERT) | **0.1631** | **yes** |
| **continued pretraining** (twitter-roberta) | **0.1711** | **yes** |
| independent same-recipe (P~) | 0.1017 – 0.3362 | — |

Restricted to fine-tuning, C5a gives AUC 0.964, p = 0.0091 — but **C5a is a stratified follow-up, not a
preregistered contrast**, and is reported uncorrected and labelled exploratory.

## The predeclared equivalence test FAILS

C4 is an equivalence claim: masking means the structural signals are *indistinguishable*, and a
failure to reject does not establish that. TOST at the **predeclared margin δ = 0.05**:

- δ = 0.05 (predeclared): TOST p = 0.2224 — **equivalence NOT established**
- δ = 0.10 (post hoc): TOST p = 0.0474 — equivalent, but this margin is ~half the observed range of
  `s_struct` and was chosen after seeing the data. **Exploratory only.**

So the honest statement is: structural signals show **no detectable difference at adequate power**
(AUC 0.607, 90% CI on the difference [−0.0985, +0.0746]), but the predeclared equivalence bar was not
cleared. Wording asserting they are *indistinguishable* must be weakened accordingly.

## Mixed-effects model — the design's stated fear was backwards

§11.4: family clustering is *"the single most likely way to overstate significance here."*
Fitting `σ_id ~ pair_class + (1|parent_family)` (n = 41, 9 parent families):

| | p |
|---|---|
| naive pooled Welch | 0.00281 |
| **mixed-effects, clustered** | **0.00003** |

coef(independent) = −0.2052; group variance 0.005278, residual 0.014889, **ICC = 0.2617**. Accounting
for clustering made the contrast ~100× *stronger*, because a quarter of the variance is family-level
and pooling dumped it into the residual. The design's prediction was wrong here and is corrected
rather than dropped.

## The operational framing that survives

σ_id separates on average (AUC 0.795, p = 3e-5) **and** misclassifies at the threshold Cisco ships:

| | σ_id | verdict |
|---|---|---|
| independent, gpt2 family | 0.836 | above the 0.75 line |
| independent, pythia | 0.713 | |
| derived, bloom | 0.780 | |
| derived, bart | 0.767 | |
| derived, roberta | 0.746 | below the line |
| **derived, twitter-roberta** | **0.5375** | **"Not Matched" — false negative** |

"Separates on average but fails at the deployed threshold" is sharper, more defensible and more
damaging than "cannot distinguish", and it is what the data supports.

## M5 — the κ>0 displacement regression

Seven benign published children of `roberta-base`; dependent variable σ_id (weight-computed, so
independent of the MFI tier — which matters because four verdicts are gate-decided artifacts of our
own `architectures` repair, see `M5/CAVEATS.md`).

| predictor | fit | R² |
|---|---|---|
| overall displacement | σ_id = 0.8382 − 0.2567·d | 0.441 |
| **embedding displacement** | σ_id = 0.8523 − 0.3023·d | **0.730** |

Embedding displacement wins, as the signal weights force: EAS (0.36) + END (0.19) = 0.55 of the
identity score reads only the embedding. WVC stays in 0.163–0.179 across the whole sweep.

**Threshold crossings, with their reliability stated:**

| threshold | disp_emb | vs from-scratch (1.2044) | basis |
|---|---|---|---|
| MPK `Not Matched` (σ_id < 0.65) | 0.676 | **56%** | **interpolated** (max observed 0.8295) |
| predeclared evasion bar (0.5298) | 1.097 | 91% | **extrapolated** — weaker |

Both crossings are fit on the embedding axis and compared to the **embedding** anchor (1.2044). An
earlier draft divided them by the *overall* anchor (0.9533), giving 70% / 112% — an axis mismatch
that flattered C1.

The from-scratch anchor is four independent Pythia runs (embedding displacement mean 1.2044;
analytic uncorrelated bound 1.4142).

**Caution, stated rather than buried:** n = 7, one parent family, R² = 0.730, and both crossings come
from a two-parameter linear fit rather than a measured threshold. The 112% figure extrapolates past
the last data point and should not be quoted without that qualifier. Replicated on a second parent family (BERT, n=5 of 7 attempted; two rows lost to stalled downloads,
see M5/CAVEATS.md). Pooled on verified lineage (n=12): recall 8/9 = 89%, specificity 1/3 = 33%.

## Coverage gap

WVC — the only positional signal, 0.21 of the identity weight — is **undefined on 13% of same-shape
pairs** (2/15: `gpt2 | gpt2-imdb`, `distilbert | distilbert-SST2`), despite matching dimensions.

## Files

`analysis/prereg_stats.py` · `prereg_stats2.py` · `prereg_final.py` · `c5_final.py` ·
`mixed_effects.py` · `model_dims.json` · `prereg_final.json` · `c5_final.json` ·
`mixed_effects_data.csv` · `M4/power_fix.py` · `M4/power_fix.jsonl`
