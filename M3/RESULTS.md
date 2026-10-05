# LAUNDROMAT M3 — claimed lineage at hub scale

Mission `mis_01M0E1Y9S8CV020Q8PWEKW03QF` · metadata only, no weights downloaded — the one part of
M3 that is feasible without a GPU. 3000 text-generation models per sample.

## Result: a `base_model` adoption curve

| sample | declare `base_model` | name implies derivative | of implied, declared | multi-parent |
|---|---|---|---|---|
| **newest** (by createdAt) | **86.3%** | 45.0% | **93.9%** | 37 |
| **recent** (by lastModified) | **81.4%** | 40.4% | **90.2%** | 47 |
| head (by downloads) | 60.6% | 53.1% | 80.3% | 53 |
| **most-liked** (oldest-skewed) | **44.6%** | 38.9% | **61.6%** | 51 |

Likes accumulate over time, so the most-liked sample skews old; `createdAt` skews new. The ordering
44.6% → 60.6% → 81.4% → 86.3% is therefore a **temporal adoption curve**, not a popularity effect.

## This refutes my own M1 hypothesis

M1 observed 2 of 6 canonical derivations declaring `base_model` and hypothesised that **claim absence**
was the dominant claimed-vs-derived divergence mode. **At scale that is wrong.** Among models whose
names imply derivation, 80–94% declare a parent depending on cohort.

The M1 sample (gpt2, distilgpt2, bert, distilbert, bloom, bloomz) was hand-picked for the *baseline*
and is disproportionately **old and canonical** — all predating widespread adoption of the field. The
two that did declare (SmolLM2-135M-Instruct, Qwen2.5-0.5B-Instruct) are recent. The adoption curve
above is the direct measurement of that bias.

## The reframing this forces

If 86% of new models declare a parent, the attestation stack's problem is **not** that claims are
missing. It is that the claims which exist are **unverifiable**:

- A declared `base_model` is an **unauthenticated self-assertion** by the publisher.
- The verification primitive built to check it **confuses training-recipe convergence with weight
  derivation** in 4 of 5 architecture families (M4), and returns *Confirmed Match* on **4/4**
  independently-trained same-architecture pairs via the metadata gate (M2).

So: a high and rising declaration rate, and a verification layer that cannot reliably adjudicate those
declarations **in either direction**. That connects M3 to the rest of the project instead of leaving it
a standalone descriptive exercise, and it needs no hub-scale fingerprinting to state.

## Incidental: multi-parent merges are real, and MPK has no verdict for them

**37–53 models per 3000 (1.2–1.8%)** declare **more than one** `base_model`, reaching **8 parents**.
Constitution §10.2 item 5 names multi-parent merging as an open problem for per-parent attribution and
license composability. This puts a measured floor under it. MPK's `compare` is pairwise and emits no
multi-parent verdict.

## Limitations

1. **The derivative proxy is a regex** over model names (`instruct|chat|sft|dpo|lora|merge|distil|
   quantiz|gguf|awq|gptq|abliterated|…`). It misses neutrally-named derivatives and over-flags some
   base models. The "of implied, declared" figures inherit that error — indicative, not precise.
2. **None of these samples is uniform random.** Four different sort orders bracket the population and
   the trend is consistent across them, which is stronger than any single ranked sample, but it is not
   a probability sample.
3. **Declaration is not correctness.** Nothing here checks whether a declared `base_model` is *true* —
   that is the derived side, which needs fingerprinting at scale and is out of reach on this hardware.
   This is the half of M3's original question that remains genuinely unanswered.

## Artifacts
```
M3/claim_survey.py · claim_survey2.py
M3/claim_{downloads,lastModified,likes,createdAt}.jsonl · claim_survey2.json
```
