# Coordinated disclosure draft — Cisco Foundation AI / AI Defense

**Status: DRAFT, NOT SENT.** Prepared under decision D2 (`dec_01M0E1WFW4N2QGP2H18KVPY3J2`).
Sending is a PI decision. Nothing here has been transmitted to Cisco or published anywhere.
Rewritten 2026-08-22 (`dec_01M0M4K8H0ZDH80YWWFQ8ZSFBM`); the previous version predated four of the
eight availability defects and all of the score-structure findings, and understated the case.

**Subject:** Model Provenance Kit v1.1.0 — six findings and a proposed mitigation
**Artifact under test:** `cisco-ai-defense/model-provenance-kit` @ `a75007d`, Apache-2.0, with the
published deep-signal fingerprint database (`sha256 f37f689b…`, 184 parquet, 39 families)
**Environment:** Apple M4, CPU only. Models 135M–1.4B. All findings reproducible from the shipped
repo with public models; no private data, no vendor access, no credentials.

---

## Framing, stated up front

We set out to break MPK's weight-level signals at zero adversary compute and **did not**. Across
four zero-training-compute attack families — exact weight-space symmetries, architecture-metadata
edits, isotropic embedding perturbation, and a precisely anchor-targeted attack — no transform
achieved evasion at any usable capability. Constitution §9's claim C1, *"fully evading all
weight-level signals is as expensive as training from scratch,"* **holds at κ = 0, with margin.**
We think that is worth stating first and plainly.

Two qualifications follow from measurements we made afterwards, and they are the reason for this
letter. First, at **non-zero** adversary compute the picture changes quantitatively (Finding 5).
Second, and more consequential in practice, most of the errors we observed are **false positives and
outright verification failures**, not evasions — MPK's problem is not that attackers get past it, but
that it says *Confirmed Match* too readily and refuses to run on a large class of legitimate models.

Findings are ordered by our assessment of operational severity.

---

## Finding 1 — denial of verification on models missing `architectures` (high, availability)

**8 of 10** models we sampled fail to verify at all. MPK exits 1, emits no JSON, and returns no
verdict. Every one of the eight produces the identical error:

```
Error: Failed to extract base features for '<model>':
1 validation error for MFIFingerprint
architectures
  Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]
```

The trigger is a `config.json` with no `architectures` key. MPK reads `None` and hands it to a
Pydantic model that requires `list`. Affected models span **five organisations**, including two
major vendors and one first-party Google release:

| model | publisher |
|---|---|
| `microsoft/MiniLM-L12-H384-uncased` | Microsoft |
| `google/bert_uncased_L-4_H-256_A-4` | Google |
| `huawei-noah/TinyBERT_General_4L_312D` | Huawei |
| `allenai/cs_roberta_base` | AllenAI |
| `allenai/biomed_roberta_base` | AllenAI |
| `allenai/news_roberta_base` | AllenAI |
| `allenai/reviews_roberta_base` | AllenAI |
| `emilyalsentzer/Bio_ClinicalBERT` | (community, widely deployed clinically) |

**Positive control.** `nreimers/TinyBERT_L-4_H-312_v2` carries `architectures: ["BertModel"]` and
verifies cleanly, exit 0. The defect is specifically the absent key, not the model family.

**Honest exclusion.** `prajjwal1/bert-tiny` also lacks `architectures` but fails earlier and
differently (`Unrecognized model … should have a model_type key`). We exclude it from the eight
because its cause is distinct; counting it would overstate the finding.

**Why this matters more than a crash.** A provenance tool that cannot produce a verdict is not
failing safe or failing closed — it is failing *silent*. In a supply-chain setting the practical
consequence of "no verdict" is that the model ships unverified. The fix is a one-line default
(`architectures: list[str] = []`) plus a `model_type` fallback, and the tool can then reach tier 2
or tier 3 rather than aborting.

---

## Finding 2 — recipe convergence is scored as derivation (high, correctness)

MPK returns **Confirmed Match**, its highest verdict, on independently-trained model pairs.
Constitution §8 lists this exact pair type under **Independent**.

| pair (independent training runs) | tier | pipeline | identity | EAS | WVC |
|---|---|---|---|---|---|
| pythia-70m vs -deduped | 1 | 1.0000 | 0.7856 | 0.9350 | 0.1556 |
| pythia-160m vs -deduped | 1 | 1.0000 | 0.6745 | 0.9649 | 0.1392 |
| pythia-410m vs -deduped | 1 | 1.0000 | 0.7810 | 0.9734 | 0.1266 |
| pythia-1b vs -deduped | 1 | 1.0000 | 0.7262 | 0.9767 | 0.1037 |
| pythia-1.4b vs -deduped | 1 | 1.0000 | **0.7976** | 0.9838 | 0.1023 |

Three points, in order of importance:

- **The gate is not the whole story.** For the 70m, 410m and 1.4b pairs the *identity score alone*
  is above the 0.75 High-Confidence line, so removing the MFI tier-1 gate would not fix them.
- **The 1.4b pair is from your own published benchmark.** On the example the tool ships as a
  reference, the weight signals fail independently of the gate.
- **It strengthens with scale.** EAS rises monotonically 0.9350 → 0.9838 while WVC falls
  0.1556 → 0.1023 across a 20× parameter range. The recipe-determined signal and the
  weight-inherited signal diverge as models grow, so this does not self-correct at frontier scale.

**Mechanism.** EAS (weight 0.36) is a Pearson correlation over cosine self-similarities of 64 anchor
embedding rows; END (0.19) is a histogram of embedding row norms. Both are stable consequences of the
*training recipe* rather than of inherited weights. WVC (0.21) is the only signal that compares
weights positionally — and it is the one that correctly reads ≈ 0.10 on these pairs.

**Independent replication on a second family.** On a 7-model BERT-family sweep with lineage taken
from publisher documentation, sensitivity is **4/4** and specificity is **1/3**. The tool recognises
genuine children reliably and is unreliable on models that were never derived. We note that one of
the two false positives sits at a tier determined by a `config.json` value *we* supplied to work
around Finding 1; excluding it gives specificity 1/2 on two models. Either reading supports the same
conclusion.

---

## Finding 3 — the tier-1 gate overrides all weight evidence (high, design)

This is the sharpest form of Finding 2, and we think it is the single most important item here.

We constructed a model whose every weight signal has been driven to its floor: randomised embedding
rows, randomised projection rows, flattened embedding-norm distribution, per-layer rescaled energy,
and randomised norm-layer vectors. Its identity score is **0.3255** — *half* the 0.65 weak-match
threshold, and below the 0.5298 null-distribution 75th percentile. On every signal MPK measures, it
is barely related to the reference.

**MPK returns `Confirmed Match`.**

The reason is that none of the transforms touches `config.json`, so `arch_hash` still matches, tier 1
fires, `pipeline_score` is pinned to 1.0, and the identity score is never consulted. The same
mechanism produces a real-world false positive: `microsoft/BiomedNLP-BiomedBERT`, pretrained **from
scratch** on PubMed abstracts, carries the *largest* embedding displacement in our sweep (1.4527) and
the *lowest* identity score (0.5248, below even the weak-match line) — and is reported as
`Confirmed Match` against `bert-base-uncased`.

A verdict that cannot be moved by any weight evidence is not reporting weight evidence. We suggest
tier 1 should cap the verdict at a metadata-only confidence, not pin it to the maximum.

---

## Finding 4 — the scoring function leaks its own weights, and one signal has half the range it appears to (medium, design)

**4a. Weight recovery from returned scores alone (observability level O0).** An adversary who can do
nothing but submit models and read scores can recover the private signal weights. Perturb a model so
that exactly one signal moves and the other four stay pinned at 1.0000 — norm-preserving
randomisation of a chosen row set achieves this — then the weight is the ratio of deltas,
`w = Δσ_id / Δs`. Measured against a public 135M model:

| target | recovered | published | error |
|---|---|---|---|
| `w_EAS` | **0.3600 ± 0.0001** | 0.36 | 0.006% |
| `w_WVC` | **0.2100 ± 0.0001** | 0.21 | 0.002% |

That is 0.57 of the unit identity weight, with no access to source, documentation or the fingerprint
database. The estimator requires no assumption about how the signal responds to the perturbation —
EAS is markedly non-linear in the perturbed fraction, and the ratio is still exact at every point —
so this is a property of the scoring design rather than of the particular transform we chose.

**4b. EAS has half the dynamic range it appears to have.** Driving the embedding to fully random
directions floors EAS at **≈ 0.50** (measured 0.4987 and 0.5013 at two seeds), not 0 — the value an
affinely rescaled cosine `(1+cos)/2` returns as `cos → 0` in high dimension. Consequences, using
your published weights and thresholds:

- EAS contributes an unconditional **0.1805** to `σ_id` for *any* pair of same-dimension models,
  related or not.
- That is **27.8%** of the 0.65 weak-match threshold and **24.1%** of the 0.75 high-confidence
  threshold, granted before any evidence of derivation is considered.
- Across every measurement we collected — 1,179 EAS observations — none falls below that floor.

**4c. The reachable set, measured.** Three of the five signals cannot be driven near zero:
EAS 0.4987, LEP 0.4836, NLF 0.6861 (WVC 0.0034 and END 0.0684 do reach ≈ 0). Those three carry 0.60
of the weight and leave an irreducible **0.3118**. The minimum attainable identity score is therefore
**0.3255**, measured rather than assumed. We flag this because it cuts *against* an attacker: it means
destroying the two most heavily weighted signals is **not** sufficient to clear the null-distribution
bar — that pair bottoms out at 0.6102. Evasion by signal destruction needs at least three signals.

---

## Finding 5 — C1 holds at κ = 0 but is quantitatively weaker at κ > 0 (medium, claim scope)

Constitution C1 states that evading the weight signals costs about as much as training from scratch.
At zero adversary compute we confirm it. At non-zero compute we measure the price directly, using
continued-pretraining and fine-tuning ladders across two parent families (11 true derivatives,
lineage verified from publisher documentation):

| threshold | embedding displacement | as fraction of a from-scratch run | basis |
|---|---|---|---|
| tool's rejection line (0.65) | 0.659 | **55%** | interpolated, inside the data |
| null-distribution p75 (0.5298) | 1.020 | 85% | extrapolated beyond our last observation |

On the strict reading — driving a model into the null distribution — C1 is approximately right at
85%. But **operational** evasion, meaning simply getting the deployed tool to answer "not matched",
arrives at about **55%** of a from-scratch run, and that crossing is interpolated inside our data
rather than extrapolated. We report the weaker of our two fits (R² = 0.693 on 11 points); an earlier
9-point fit gave 0.780 and a 56% crossing, and we prefer the fuller sample.

We are not claiming this breaks C1. We are reporting that the margin on the operational threshold is
roughly half what the strict reading suggests.

---

## Finding 6 — two low-severity implementation issues

**6a. `tokenizer.json` `truncation` block changes `family_hash`.** A semantically inert field
participates in the hash, so unrelated tooling that rewrites the tokenizer can move a model between
MFI tiers. We found this by byte-comparing artifacts after our own harness leaked truncation state;
MPK itself is deterministic (8/8 identical runs).

**6b. Benchmark labelling.** One pair in the shipped benchmark is labelled in a way inconsistent with
Constitution §8's own categories. Details on request; it is a documentation-level correction.

---

## Proposed mitigation — require aligned positional evidence

The common root of Findings 2 and 3 is that a verdict can be reached without any signal that compares
weights *positionally*. We prototyped a defence and evaluated it against our own corpus.

**Construction.** Hungarian assignment on the MLP input-projection rows after column L2
normalisation, then a correspondence score. The design rule that makes it sound: align only on
matrices whose symmetry group is permutation-only, so the alignment cannot absorb a rotation.

**Results.** On 18 pairs spanning all three derivation types (fine-tune, continued pretraining,
distillation):

- separation margin **+0.2469**, AUC **1.000**, two-sided exact permutation p = **4.6 × 10⁻⁵**;
- pre-registered held-out split by parameter scale — threshold fit on 13 development pairs
  (≤160M), applied unchanged to 5 held-out pairs (≥360M): **5/5 correct**, held-out margin
  **+0.7588**, wider out of sample than in development.

**Distillation caveat, stated because it caught us.** Distilled children must be matched to parent
layer `2i`, not `i` — the initialisation these models actually use. Under naive index-to-index
matching two of three distillation pairs fall into the negative range, which looks like a failure of
the method and is in fact a failure of the layer correspondence. We initially retracted these values
for that reason and were wrong to; the retraction is itself retracted.

**Non-guarantees.** This does not detect derivation that re-trains the MLP input projections
substantially; it assumes same-width parents and children on the aligned matrices; and 18 pairs is a
small corpus. It is offered as a direction, not a finished control.

---

## Handling

We are drafting an academic paper on these findings and intend to follow standard coordinated
disclosure. Our preference:

- **90 days** from acknowledgement before any public description, extendable by agreement.
- We will share the full harness, the exact model list, config hashes, and the reproduction
  scripts on request, at any point, including before you acknowledge.
- We will incorporate any correction you send. Several numbers above have already been revised
  downward by our own re-analysis, and we would rather be corrected than be first.
- If you would prefer we withhold any specific finding until a release ships, say which and we will.

We have no commercial interest in MPK, no relationship with any competitor, and are not seeking
access to any Cisco resource. Everything above was produced from the public repository and public
models on a single laptop.
