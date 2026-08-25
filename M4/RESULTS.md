# LAUNDROMAT M4 — Consequence and the laundering-resistant design

Project `prj_01M0E1SKSP6CSEHY9DTT4V9MB2` · 2026-08-19 · Apple M4, CPU only, no GPU

## The question M2 left

M2 established a tension that looked structural: the signals that resist laundering are the ones
measuring the training **recipe** — and therefore false-positive on independent same-recipe models —
while the one signal tracking weight inheritance is destroyed by a free permutation. Is that tension
fundamental, or a property of MPK's particular statistics?

**It is the latter, and the fix is already in the literature.**

## Experiment 1 — the tension, measured on one table

Two statistics over the same weight matrices, same pairs:

| class | SVD-spectral (perm-**invariant**) | raw positional (perm-**sensitive**) |
|---|---|---|
| IDENTITY | 1.0000 | 1.0000 |
| P+ derived (SmolLM2) | 1.0000 | 0.9963 |
| P+ derived (Qwen) | 1.0000 | 0.9975 |
| **P~ same-recipe independent** | **0.9924** | **0.1341** |
| P− unrelated (SmolLM2/pythia) | 0.9460 | 0.0003 |
| P− unrelated (gpt2/bloom) | 0.9852 | −0.0032 |
| LAUNDERED (X1a∘X1b∘X2) | 0.9955 | 0.2413 |

The permutation-invariant statistic is laundering-robust (0.9955) and **useless** — unrelated models
score 0.946–0.985, indistinguishable from derived at 1.0000. The permutation-sensitive one discriminates
perfectly (0.996 vs 0.0003) and is **destroyed** by a free permutation (0.996 → 0.2413).

> **Caveat, important.** My spectral statistic is a cosine between normalised top-64 singular-value
> vectors — crude. Trained transformer spectra are all power-law-ish, so high cross-model similarity is
> expected. This does **not** show GhostSpec or Spectral Signatures fail; they use richer shape metrics
> and alignment. It shows *naive* permutation-invariance buys robustness at the cost of discrimination,
> which is the design pressure those papers respond to.

## Experiment 2 — the way out: LAP + column normalisation

Undo the permutation by solving the assignment problem, *then* compare values. Invariances by
construction: column L2 normalisation kills X2's per-column scaling; row L2 normalisation kills any
per-row scaling; LAP over rows kills X1a's permutation. Discrimination survives because it still
compares matched **values**.

| class | LAP + colnorm | LAP only |
|---|---|---|
| IDENTITY | 1.0000 | 1.0000 |
| P+ derived (SmolLM2) | **0.9969** | 0.9969 |
| P+ derived (Qwen) | **0.9969** | 0.9969 |
| **P~ same-recipe independent** | **0.1961** | 0.1942 |
| LAUNDERED perm-only | **0.9969** | 0.9969 |
| LAUNDERED full (SmolLM2) | **0.9969** | 0.8807 |
| LAUNDERED full (Qwen) | **0.9969** | 0.8773 |

**Both properties hold at once.**

- **Sound** — derived 0.9969 vs independent-same-recipe 0.1961, a separation of **0.80**. MPK's EAS
  scores that same P~ pair at 0.9838 against ~0.999 for a fine-tune: separation ≈ **0**. This is exactly
  the false positive that defeats the deployed tool, and LAP resolves it.
- **Robust** — laundered artifacts recover to **0.9969**, the clean value, on *both* models. Column
  normalisation is what closes it: LAP alone leaves 0.8807/0.8773 because it undoes the row permutation
  but not X2's column rescaling.


## Experiment 3 — the defence across ALL derivation types (the stress test)

The same-recipe arm is WVC's *easy* case. The real test is **distillation**, where positional
evidence is legitimately near-zero for a genuine derivation — the non-guarantee predeclared in the
research design. Tested rather than asserted:

| class | pair | LAP + colnorm |
|---|---|---|
| P+ fine-tune | SmolLM2-135M | SmolLM2-135M-Instruct | 0.9969 |
| P+ multitask | bloom-560m | bloomz-560m | 0.9762 |
| P+ distill | bert-base-uncased | distilbert-base-uncased | 0.5255 |
| P+ distill | roberta-base | distilroberta-base | 0.5987 |
| P+ distill | gpt2 | distilgpt2 | 0.5148 |
| P~ same-recipe | pythia-160m | pythia-160m-deduped | 0.1961 |
| P~ same-recipe | pythia-410m | pythia-410m-deduped | 0.2151 |
| P- unrelated | bert-base-uncased | roberta-base | 0.1170 |
| P- unrelated | distilbert-base-uncased | distilroberta-base | 0.1169 |
| P- unrelated | SmolLM2-135M | SmolLM2-360M | nan |

```
SEPARATION  min(derived)=0.5148  max(not-derived)=0.2151  margin=+0.2996
THRESHOLD any value in (0.2151, 0.5148) classifies all 9 pairs correctly
```

**The defence degrades on distillation but does not fail.** Distillation lands at 0.515–0.599 —
well below same-architecture derivation (0.976–0.997) but well *above* independent-same-recipe
(0.196–0.215) and unrelated. The ordering is preserved across every derivation type tested, so a
single threshold separates all pairs.

Compare MPK on the same corpus: its best signal (EAS) separates derived from unrelated by a margin
of **+0.0117**, and gets distillation right only via the same signal that returns "Confirmed Match"
on 4/4 independently-trained pairs. The aligned positional statistic gets **both** regimes right.

### Corrected framing (I overstated this earlier)

I previously wrote that EAS "separates by approximately zero". That is true **only of the
same-recipe arm**. On the full 20-pair corpus EAS *does* separate — margin +0.0117 — and its three
worst unrelated pairs are all same-family (SmolLM2-135M/360M 0.9732, pythia-160m/-deduped 0.9649,
pythia-160m/410m 0.9564) against a derived floor of 0.9849. Cross-family unrelated pairs sit at
0.679–0.899.

The accurate and sharper claim: **EAS and WVC fail in complementary, identifiable regimes** — EAS on
shared recipe, WVC on distillation (0.0070 for roberta→distilroberta; NaN for two others, all genuine
derivations). Neither is universally sound; the deployed pipeline weights them 0.36 to 0.21 in favour
of the one whose blind spot is precisely the case its own Constitution lists as *Independent*.

## Why this is Act III

The fix is **not novel**. LAP alignment is AWM's mechanism (ICLR 2026); CKA invariance is REEF's
(ICLR 2025). Both public well before MPK shipped in April 2026. The contribution is showing that
(a) the deployed stack's statistics cannot be both sound and robust, (b) a known technique achieves
both, and (c) the distance between them is a **deployment** gap, not a research gap — precisely the
framing ratified at the M1 gate.


## Experiment 4 — adaptive attack on the defence (I said I wouldn't claim robustness without this)

### A new exact κ=0 symmetry: **X7, QK invariance**

Attention scores are a **bilinear form**: `scores = (x W_q)(x W_k)ᵀ = x W_q W_kᵀ xᵀ`. So for *any*
invertible `M`, setting `W_q → W_q M` and `W_k → W_k M⁻ᵀ` leaves `W_q M M⁻¹ W_kᵀ = W_q W_kᵀ` unchanged.

This is **richer than anything in the M2 family**. X1a/X1b *relabel* and X2 *rescales*; `M` is an
arbitrary invertible matrix that **mixes** coordinates. Requires no RoPE (which doesn't commute with
arbitrary `M`), so it applies to GPT-2/BERT-style positional schemes but not Llama/Qwen/Pythia.

Exactness on gpt2, fp32, 48 probes:

| log σ | max \|Δlogit\| | rel Δppl | |
|---|---|---|---|
| 0.1 | 6.41e-04 | 2.3e-07 | **EXACT** |
| 0.3 | 2.66e-03 | 7.1e-06 | **EXACT** |
| 1.0 | 2.24e+02 | 1.65 | NOT EXACT — `M` ill-conditioned, `M⁻¹` amplifies error |

### The attack succeeds

| variant | exact | LAP on **MLP** | LAP on **ATTN** |
|---|---|---|---|
| identity (control) | ✓ | 1.0000 | 1.0000 |
| X7, log σ = 0.1 | ✓ | 1.0000 | 0.7565 |
| **X7, log σ = 0.3** | ✓ | **1.0000** | **0.2160** |

LAP alignment on attention matrices **collapses to 0.2160** — at the top of the not-derived band
(threshold boundary 0.2151), i.e. a genuine derivation scored as a non-derivation, at **exact
capability and zero compute**. LAP on MLP is untouched.

**Why:** an assignment problem can undo a *permutation*. It cannot undo a general linear *mixing*.
LAP searches the symmetric group; X7 lives in the general linear group.

### The design rule — the actual contribution here

The defence's robustness is **not a property of LAP**. It is a property of **which matrices LAP reads**.

- **MLP intermediate axis** — the elementwise nonlinearity (SwiGLU/GELU) forbids general mixing; the
  symmetry group is *permutations × positive diagonal*. LAP + column normalisation covers it exactly.
- **Attention head axis** — the QK bilinear form is invariant under the **full general linear group**.
  No alignment over permutations can cover that.

> **Align only on matrices whose symmetry group is permutation-only — i.e. matrices constrained by an
> intervening nonlinearity. Never on matrices related by a bilinear invariance.**

My defence reads MLP `up_proj`, so it satisfies the rule — but I chose MLP for implementation
convenience *before* understanding why it mattered. A naive extension to attention matrices, which an
implementer might reasonably attempt for extra signal, would be defeated by X7.

**On AWM and REEF:** I have not tested either implementation and claim nothing about them. The rule
above is a *hypothesis about the class* — any alignment-based detector reading attention Q/K inherits
this exposure unless it explicitly quotients by the GL invariance. Worth flagging to those authors.

### Caveat on this arm's MPK numbers

The GPT-2 identity control is **not clean**: gpt2 vs a re-saved gpt2 gives σ_id 0.9212 with NLF 0.2221
and WVC 0.0000, where an identical model should give ~1.0 throughout. The save round-trip evidently
changes GPT-2's tensor representation (Conv1D handling or naming). **The MPK columns for this arm are
therefore not a clean measurement of X7's effect on the detector and are not reported as one.** The LAP
result is unaffected — computed on in-memory models, never round-tripped. X7's effect on MPK
specifically remains **unmeasured**, and needs a non-RoPE model with a faithful save round-trip.

## Limitations

1. **Requires matched matrix shapes.** LAP compares equal-shaped matrices, so it cannot compare across
   architectures or sizes. MPK's EAS is *dimension-independent by deliberate design*. This is a real
   capability given up — LAP is a **complementary** check for the matched-shape (common fine-tune) case,
   not a drop-in replacement.
2. **Adaptive attack: run, and partly successful.** X7 (QK invariance) defeats LAP on *attention*
   matrices (1.0000 → 0.2160, exact, κ=0). The defence survives only because it reads *MLP* matrices,
   whose symmetry group is permutation-only. This is now a stated **design constraint**, not an
   untested assumption. No claim of robustness against attacks outside the alignment class.
3. **Soundness rested on n=1** (pythia-160m vs -deduped) — being addressed by the P~ ladder.
4. **O(n³)** in the intermediate dimension; 4864 rows ran fine on CPU, larger untested.
5. **A bug in my own defence, found and fixed.** I first subsampled *rows* before alignment, which
   strides a different subset in the permuted model and destroys correspondence. It made Qwen read
   0.4012 instead of 0.9969 — I would have wrongly concluded the defence is architecture-specific. Rows
   are what the permutation moves and must never be subsampled before alignment; reduce features instead.

## Artifacts
```
M4/defence.py        spectral vs positional (the tension)
M4/lap2.py           LAP + column normalisation (the resolution)
M4/p_tilde_arm.py    the same-recipe ladder, MPK signals + LAP side by side
M4/{defence_signals,lap_defence,lap2}.json · p_tilde_arm.jsonl
```
