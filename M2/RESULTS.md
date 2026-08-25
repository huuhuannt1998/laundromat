# LAUNDROMAT M2 — The κ = 0 Laundering Frontier

Mission `mis_01M0E1XPEXWWF82X156K9ZPSYA` · project `prj_01M0E1SKSP6CSEHY9DTT4V9MB2` · 2026-08-19
Hardware: **Apple M4, 10 cores, 24 GB, CPU only. No GPU, no cluster, no cloud.**
Every transform below costs **zero training compute** — pure tensor ops plus file writes.

## Verdict

**Gate G0: GO** (`chk_01M0E7PHNRWEJVKMJEVSY9WH2G`) · **Gate G1: EXIT 3.**

> *"Deployed provenance verification resists free laundering but confuses recipe with descent."*

Predeclared `E(L)=1 ⟺ rank > 5 ∧ π ≤ 75` was **never achieved**. Neither leg was loosened after the fact.

## Stage 0 — substrate and metric validity

Frozen null **|N| = 1388**, 45 strata. `p10 .2601 · p25 .3743 · p50 .4558 · p75 .5298 · p90 .6351 · p95 .6925 · p99 .8230`

`p99 = .8230` independently corroborates M1: ~1 in 100 *unrelated* pairs reaches into the High-Confidence band.

- **Rank is valid.** 140 competitors survive the param filter on every scan (criterion ≥ 20). C4 fired on **0/3** benign targets (criterion < 20%).
- **But difficulty is parent-dependent** — the real bar is the 6th-highest score in the target's own scan:
  SmolLM2 `0.6456` (sparse) · Qwen2.5 `0.7224` (medium) · distilgpt2 `0.8320` (dense) · pythia-160m `0.5555`.
  Every rank result must therefore ship its density profile.
- **Declared deviation:** the `≥10 per stratum` criterion is missed — 41/45 qualify, smallest is 8. Not relaxed; M-3 reads percentiles off the pooled N, so this governs coverage, not estimator reliability.

### Transform exactness — verified, not assumed

fp32, 96 probes. Tolerance `max|Δlogit| ≤ 1e-2`, `rel Δppl ≤ 1e-3`.

| transform | max\|Δlogit\| | rel Δppl | fp32 | **bf16 (published dtype)** |
|---|---|---|---|---|
| X1a MLP permutation | 6.46e-04 | 5.0e-07 | EXACT | **EXACT** (1.76e-04) |
| X1b head permutation | 1.48e-04 | 0.0 | EXACT | **EXACT** (1.07e-04) |
| X2 RMSNorm scaling | 1.99e-04 | 1.2e-06 | EXACT | **✗ NOT EXACT** (2.47e-01) |

**Permutations survive quantized publication; scaling does not.** Permutation only *moves* values; scaling
multiplies and divides, and in bf16 the compensating division does not recover the product. Not reported by
Fernandez et al. or any invariant-watermarking work — a practical constraint on that whole line.

Determinism control: identity transform returns `σ_id = 0.9992`, exactly reproducing M1's published number.

## Gate G1 — the arms

**Arm 1 — exact symmetries.** SmolLM2-135M and Qwen2.5-0.5B agree.

| case | tier | σ_pipe | σ_id | rank | π | E |
|---|---|---|---|---|---|---|
| S0-identity | 1 | 1.0000 | 0.9992 | 1 | 100.0 | 0 |
| X1a | 1 | 1.0000 | 0.9116 | 1 | 100.0 | 0 |
| X1b | 1 | 1.0000 | 0.9334 | 1 | 100.0 | 0 |
| X2 | 1 | 1.0000 | 0.9518 | 1 | 100.0 | 0 |
| X1a∘X1b∘X2 | 1 | 1.0000 | 0.8137 | 1 | 100.0 | 0 |
| M1t (metadata only) | 2 | 0.9000 | 0.9992 | 1 | 100.0 | 0 |
| **S2 flagship** X1a∘X1b∘X2∘M1t | 3 | **0.8137** | 0.8137 | **1** | 98.7 | **0** |

Identity score reconstructed from published weights matches MPK to **2e-5**, so the attribution is exact:
**EAS 0.9997 → 0.9997 and END 0.9999 → 0.9999 — completely unmoved.** Only NLF, LEP, WVC responded.

**Arm 2 — isotropic embedding noise.** At `σ=0.40`: ΔEAS = 0.0028, Δσ_id = 0.0010 — one thousandth of a point.

> **CORRECTED (M-5 battery).** This line previously read "ppl 20.17 → 53.63 (ρ = 0.376) … *total capability
> destruction*", which the predeclared M-5 instrument refutes. On LAMBADA/ARC-E/HellaSwag the same artifact
> retains **ρ = 0.7863** (floor-corrected 0.671; LAMBADA alone, which has no guessing floor, 0.448). Perplexity
> overstated the damage. The transform is *degrading*, not destroying. The conclusion is unchanged and in fact
> strengthened: the σ_id movement is bought at **less** capability cost than claimed, and it is still 0.0010.

**Arm 3 — anchor-targeted attack (strongest available at O1).** EAS carries 0.36 of the weight but reads only
**K=64 rows**, chosen by a deterministic rule hardcoded in the Apache-2.0 repo. Recovered the exact anchor set
by importing MPK's own `get_anchor_ids` — source read, zero queries: **64 of 49,152 rows = 0.130%**.

| σ | EAS | END | σ_id | rank | π | ppl | ρ |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.9997 | 0.9999 | 0.8458 | 1 | 99.7 | 12.59 | 1.000 |
| 0.50 | 0.9948 | 0.9999 | 0.8441 | 1 | 99.7 | 15.89 | 0.793 |
| 1.00 | 0.9751 | 0.9999 | 0.8370 | 1 | 99.4 | 55.52 | 0.227 |
| 2.00 | 0.8663 | 0.8387 | 0.7672 | 1 | 97.6 | 1.91e6 | 0.000 |
| 4.00 | 0.6454 | 0.2685 | 0.5793 | 1 | 83.3 | 2.83e11 | 0.000 |
| 8.00 | 0.5360 | 0.0503 | **0.4985** | **1** | **65.9** | **inf** | 0.000 |

Targeting is 3–5× more efficient per unit capability than isotropic noise — the anchor set is a genuine,
precisely exploitable surface. **It is still nowhere near enough.** The percentile leg *is* eventually
satisfied (σ_id 0.4985 < p75, π 65.9 ≤ 75) — at **infinite perplexity**, and the parent is *still rank 1*.
Classification: **DESTRUCTIVE**, exactly the trivial reading the predeclared taxonomy exists to exclude.

**Arm 4 — tokenizer-mediated anchor shadowing (the last flagged attack).** MPK resolves anchors by calling
`encode()` on the *attacker-published* tokenizer, so this looked like the most promising κ=0 route. It fails:

- `add_tokens` on strings already in vocab maps to **existing ids** (vocab 49152 → 49152); **0/64 anchors moved**.
- Any variant that *would* redirect shreds real text: `"There are other mothers in the theatre."` 8 → 20 tokens,
  `other` → `Ġo|the|r`, `mothers` → `Ġmo|the|rs`.

### Best-case adversary bound — why the verdict does not depend on our attack being optimal

Granting every signal its best observed κ=0 movement, including **perfect** EAS destruction:

`0.36(0.50) + 0.08(0.9170) + 0.16(0.8694) + 0.19(0.9999) + 0.21(0.2447)` = **σ_id ≥ 0.6338**

vs the π\*=75 cut of `0.5298` — **+19.6% above the bar**. Even complete destruction of the highest-weighted
signal, at affordable capability, cannot reach evasion.

*Scope correction:* the earlier `w_EAS + w_END = 0.55` floor applies **only** to embedding-*preserving*
transforms. Once the embedding is attacked, EAS+END fall to 0.37 — below the cut — so the binding argument
in that regime is the bound above, which rests on NLF and LEP being hard to move.

## The finding

**Capability is exhausted long before the linkage signal moves.** ρ 1.00 → σ_id 0.8458; ρ 0.79 → 0.8441;
ρ 0.23 → 0.8370; ρ 0.00 → 0.4985. Constitution **C1 holds at κ = 0, with margin** — and we price it rather
than asserting it.

**The duality.** EAS is simultaneously (a) the signal most robust to laundering and (b) the signal chiefly
responsible for M1's recipe-convergence false positives (EAS = 0.9838 on independently-trained
pythia-1.4b vs -deduped). **These are the same property.** EAS resists perturbation because it measures a
stable structural consequence of the *training recipe* rather than the specific weights — and it
false-positives on independent same-recipe models for exactly that reason.

*You cannot fix the false positives by hardening the signal, and you cannot make it discriminative without
making it attackable.* That is a structural property of the primitive, not an implementation bug, and it is
what M4's laundering-resistant design must actually solve.

## Limitations

1. **Not an optimal attack.** Anchors were perturbed with *random* noise. EAS is differentiable, so an O1
   adversary could optimise the 64×576 anchor block against a perplexity constraint. Untested. The best-case
   bound is why the verdict survives regardless.
2. **Two parents, one architecture family.** SmolLM2-135M (sparse) and Qwen2.5-0.5B (medium), both
   tied-embedding Llama-family decoders ≤ 0.5B. Untied-embedding models admit an exact per-row embedding
   rescaling these do not; untested.
3. **The rank leg never moved**, so this does not exercise the rank metric's discriminative range.
4. **Two method corrections, recorded not buried.** (a) bf16 rounding silently erased sub-4e-3 perturbations,
   making σ=0.003 and σ=0.010 bit-identical — patched to fp32 accumulation; those rows are lower bounds.
   (b) The initial 96-probe perplexity set was too easy — noise *improved* it (ρ > 1). Replaced with a longer
   held-out corpus; any ρ claim in the paper needs the LAMBADA/ARC-E/HellaSwag subsets the design specified.


## Corrections applied after first write-up

Three issues found and fixed; none changes the Exit-3 verdict, all are recorded rather than buried.

### C1 — MFI tier column was contaminated by tokenizer state leakage (anomaly A1, resolved)

MPK is **deterministic** (8/8 identical runs on one fixed artifact). The tier variation came from *my*
harness: `verify_exact` called the tokenizer with `truncation=True`, which mutates a fast tokenizer's
backend state, and `save_pretrained` serialises it into `tokenizer.json`:

```
fresh tokenizer -> sha256 bf346d64…  "truncation": null
after one call  -> sha256 577e6d03…  "truncation": {"direction":"Right","max_length":128,…}
```

MPK folds a hash of the tokenizer backend into `family_hash`, so the probed tokenizer silently demoted
MFI tier 2 → 3. Patched (pristine tokenizer for the artifact save). **Unaffected:** every weight signal,
`identity_score`, rank, percentile, exactness, and both sweeps — the weights were provably identical
(0/272 tensors differing).

**This is also a finding.** `family_hash` is sensitive to a *runtime inference-config* field that has
nothing to do with lineage — accidentally (genuine siblings land in different tiers) and adversarially
(a subtler gate-breaker than editing architecture fields). → disclosure, low severity.

### C2 — corrected gate mechanism

`M1t` alone does **not** reach tier 3; it takes tier 1 → **tier 2** (breaks `arch_hash`, not `family_hash`).
Reaching tier 3 needs a **tokenizer** edit (`M3t`). Clean re-runs:

| case | tier | σ_pipe | σ_id | rank | π | E |
|---|---|---|---|---|---|---|
| FLAGSHIP + M1t only (SmolLM2) | 2 | **0.9000** | 0.8137 | 1 | 100.0 | 0 |
| FLAGSHIP + M1t only (Qwen) | 2 | **0.9000** | 0.8375 | 1 | 100.0 | 0 |
| gate-only (M1t∘M3t) | 3 | 0.9992 | 0.9992 | 1 | 100.0 | 0 |
| **TRUE FLAGSHIP** (X1a∘X1b∘X2∘M1t∘M3t) | 3 | **0.8137** | 0.8137 | **1** | 98.7 | **0** |

Even with the gate fully defeated by two zero-cost metadata edits, the weight evidence still puts the
parent at rank 1. **Exit 3 confirmed on the corrected configuration** — and tier 2 (σ_pipe 0.9) is
*further* from evasion than the contaminated reading suggested.

### C3 — the untied-embedding arm is closed, and my claim about it was wrong

I wrote that untied-embedding models admit an exact per-row embedding rescaling. **They do not.** In a
residual transformer the *unnormalised* embedding enters the residual stream, so scaling is not absorbed:

| model | tie | emb-LN | per-row scale | global scale |
|---|---|---|---|---|
| pythia-160m | False | False | max\|Δlogit\| 6.13 — **NOT EXACT** | 2.20 — **NOT EXACT** |
| bloom-560m | True | True | ppl → **inf** | 0.52 rel — **NOT EXACT** |

Two architectures, two *different* failure reasons, exactly as the residual argument predicts. So
**EAS and END cannot be moved by any function-preserving transform of a residual transformer**, and the
0.55 floor is architectural rather than a tied-embedding artifact. The exception is nameable: an
architecture with an embedding-normalising layer *before* the residual stream **and** an untied head.

**Method lesson:** the Stage-0 determinism control passed precisely because it exercised the one code
path (`ops=[]`) that never called `verify_exact`. Controls must cover the *same* path as the treatment.

## Artifacts
```
M2/transforms/laundry.py     X1a, X1b, X2, M1t, N1emb + verify_exact
M2/anchor_attack.py          X5 anchor-targeted (imports MPK's own get_anchor_ids at O1)
M2/eval_transform.py         Alg-1 EVAL: transform → score → delete artifact
M2/{wide_scan,build_null,sweep_embed,summarize}.py
M2/results/*.jsonl           wide_scans, null_frozen, gate_*, sweep_embed_*, anchor_attack_*
```
