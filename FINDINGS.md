# LAUNDROMAT — consolidated findings

Project `prj_01M0E1SKSP6CSEHY9DTT4V9MB2` · IEEE S&P 2027 Cycle 2 · 2026-08-19
**All work on one machine: Apple M4, 24 GB, CPU only. No GPU, no cluster, no cloud.**
Target: Cisco Model Provenance Kit v1.1.0 @ `a75007d` + its published fingerprint database.

---

## The one-line result

**Deployed model-lineage verification resists free laundering but cannot tell "same recipe" from
"same weights" — and it already errs in both directions on ordinary public models, with no
adversary involved.**

## The arc, and how it changed under measurement

The project was scoped as an *attack* paper. The attack failed, and the failure was informative —
then the controls we ran to validate the attack turned out to be the result.

| | scoped as | measured |
|---|---|---|
| **Laundering** (guilty model escapes an edge) | the threat | **0 successes** across 4 κ=0 attack families |
| **False attribution** (innocent model acquires an edge) | not in scope | **4/4**, strengthening with scale |
| **Missed attribution** (real child rejected) | not in scope | **a published RoBERTa child → "Not Matched"** at σ_id 0.5375 |
| **Denial of verification** (no verdict at all) | not in scope | **4 models crash the verifier**, incl. a first-party Google release |

Cisco's Constitution §9 claim **C1** — *"fully evading all weight-level signals is as expensive as
training from scratch"* — **holds at zero adversary compute, with margin.** We priced it rather than
asserting it, which no prior work had done. The frontier is now *derived* (§Act II) rather than
inferred from a failed search.

But C1 being true is not the same as the system working. The cheapest way to break lineage
verification is not to evade it — it is to **omit a JSON key** and get no verdict at all; and the
commonest way to get a *wrong* verdict is to do nothing unusual whatsoever, since both false
positives and the false negative arise from benign, standard publishing practice. **The threat model
is aimed at an adversary the deployment does not have, and misses the failures it does.**

---

## Act I — the primitive is unsound before anyone attacks it

MPK returns **Confirmed Match**, its highest verdict, on independently-trained pairs:

| pair (independent runs) | tier | pipeline | identity | EAS | WVC |
|---|---|---|---|---|---|
| pythia-70m vs -deduped | 1 | 1.0000 | 0.7856 | 0.9350 | 0.1556 |
| pythia-160m vs -deduped | 1 | 1.0000 | 0.6745 | 0.9649 | 0.1392 |
| pythia-410m vs -deduped | 1 | 1.0000 | 0.7810 | 0.9734 | 0.1266 |
| pythia-1b vs -deduped | 1 | 1.0000 | 0.7262 | 0.9767 | 0.1037 |
| gpt2 vs gpt2-medium | 3 | 0.8915 | 0.8915 | 0.9711 | 0.0000 |

Constitution §8 lists every one of these under **Independent**.

**Generality across five architecture families** (same recipe, different size — also §8 *Independent*):

| family | EAS | σ_id | verdict |
|---|---|---|---|
| SmolLM2 135M/360M | 0.9732 | 0.7153 | Weak |
| GPT-2 gpt2/medium | 0.9711 | 0.8915 | **High-Confidence ✗** |
| Pythia 160m/410m | 0.9564 | 0.7378 | Weak |
| OPT 125m/350m | 0.9387 | 0.5149 | Not Matched ✓ |
| BLOOM 560m/1b1 | 0.7990 | 0.5486 | Not Matched ✓ |

EAS fails to discriminate in **4 of 5** families (0.939–0.973 vs 0.706–0.779 for cross-family
unrelated). BLOOM is a clean negative — the failure is **architecture-dependent, not uniform**. OPT is
saved by an incidental `END = 0.0838`, not by distinguishing recipe from derivation.

- **Removing the gate would not fix it.** For pythia-70m/410m the *identity score alone* (0.7856,
  0.7810) exceeds the 0.75 threshold. gpt2/gpt2-medium reaches High-Confidence at tier 3 with no gate involved.
- **It worsens with scale.** EAS rises 0.9350 → 0.9767 while WVC falls 0.1556 → 0.1037 across 14×.
- **The vendor benchmark cannot see it.** `Benchmark_All.json` labels the one diagnostic pair a
  positive, so the reported "96% accuracy, no false positives" is blind to this mode by construction.
- **One discordant pair, not overlap (corrected).** roberta→distilroberta (0.7300) sits below gpt2 vs
  bloom-560m (0.7752), so no *perfect* threshold exists — but AUC is 0.949 (p = 0.0071, n = 3/13).
  The earlier "distributions overlap" phrasing is **retracted**.

### And it is wrong in the other direction too — on a real published child

The false positives above are only half of it. `FacebookAI/roberta-base` →
`cardiffnlp/twitter-roberta-base-sentiment-latest` is a genuine, documented derivative
(CardiffNLP TimeLMs: continued pretraining of roberta-base on Twitter, then sentiment
fine-tuning). MPK's verdict:

| | value |
|---|---|
| MFI tier | 3 — weight-decided, the gate does not save it |
| σ_id | **0.5375** |
| EAS | 0.7346 *(vs ~0.98 typical for derived pairs)* |
| **END** | **0.1879** *(vs 0.9966 for the same parent → SecureBERT)* |
| verdict | **Not Matched** |

**A true child is rejected outright.** And σ_id = 0.5375 sits **0.0077 above the predeclared
evasion bar** (null p75 = 0.5298) — a benign publisher, not attacking anything, lands on the
frontier that no κ=0 transform we built could reach.

So on public models a user could check today, the primitive returns *Confirmed Match* for
independently-trained Pythia siblings and *Not Matched* for a real RoBERTa descendant. Both
errors, opposite directions, no adversary in either.

**The mechanism, isolated.** Same parent, same class of operation, different corpus:

| child | corpus | EAS | END | σ_id | verdict |
|---|---|---|---|---|---|
| SecureBERT | cybersecurity text | 0.9818 | 0.9966 | 0.7767 | Confirmed Match |
| twitter-roberta | Twitter | 0.7346 | 0.1879 | 0.5375 | **Not Matched** |

The moving variable is **corpus distance**, not the fact of continued pretraining — which makes
σ_id as a function of pretraining-corpus divergence a directly testable follow-up.

**This scopes Act II rather than contradicting it.** EAS and END are invariant under every
*function-preserving* transform — that is verified and still holds. But invariance under exact
symmetries is not immovability: continued pretraining moves both, decisively. The floor below is a
floor **at κ = 0 only**. C1 is right that evading the weight signals costs training compute; what
the vendor does not account for is that *ordinary model development already spends it*.

## Act II — but it resists free laundering, and we can say why

Four κ=0 attack families, all verified exact or capability-measured:

1. **Exact symmetries** (MLP permutation, GQA-aware head permutation, RMSNorm scaling) — verified exact
   to max |Δlogit| 6.5e-4. σ_id 0.9992 → 0.8137. **Verdict unmoved:** the MFI gate pins it.
2. **Metadata edits** — `M1t` reaches tier 2 only; tier 3 additionally needs a tokenizer edit (`M3t`).
3. **Isotropic embedding noise** — degrading the model to ρ = **0.7863** on the predeclared M-5 battery
   (LAMBADA alone 0.448) moves σ_id by **0.0010**. *Earlier drafts said "destroying" on a perplexity proxy
   (ρ=0.376); M-5 refuted that — see CORRECTIONS.md #10.*
4. **Anchor-targeted** — recovered MPK's exact 64-row anchor set from public source at O1
   (0.130% of the embedding). 3–5× more efficient, still insufficient: reaching the null p75 needs
   *infinite* perplexity, and the parent is **still rank 1**.

5. **Vocabulary remap (`M2t`)** — reorder token IDs and permute the embedding/`lm_head` rows to
   match, rewriting `tokenizer.json`. BPE merges are pairs of token *strings*, so renumbering leaves
   them untouched. **Perfectly exact: max |Δlogit| = 0.00e+00**, cleaner than the weight symmetries.
   Moving **99.96% of the vocabulary** (49,133/49,152 IDs) costs EAS **0.07** — 0.9997 → 0.9297, still
   *Confirmed Match* at rank 1.

Plus **tokenizer anchor-shadowing**, which fails at the mechanism level (0/64 anchors redirected).

**Why the vocabulary attack fails, from the Apache-2.0 source.** `get_anchor_ids` selects anchors by
*encoding fixed token strings* (`ANCHOR_PUNCT`, per-script common words) through the target's **own**
tokenizer, then tops up to `anchor_k = 64` with a random ID backfill seeded `RandomState(42)`. For
SmolLM2 that is 40/64 (62%) **content-addressed** and 24/64 (38%) index-addressed. The
content-addressed anchors simply follow the remap — MPK re-encodes the strings and finds them at
their new IDs.

But content-addressing is *not* the deep reason, and it would be wrong to claim it is: moving 38% of
the anchors cost only 0.07 of EAS, so even a fully index-addressed anchor set extrapolates to
EAS ≈ 0.81 — degraded, not destroyed. **The real reason is that EAS measures a distributional
property of the embedding geometry that barely depends on which tokens you sample.**

### The κ=0 route to EAS is closed on three independent fronts

| front | why it fails | evidence |
|---|---|---|
| function-preserving weight transforms | never touch the embedding matrix | theorem + verified exactness |
| vocabulary reordering | EAS is a distributional property; anchors are 62% content-addressed | 0.9997 → 0.9297 at 99.96% remapped |
| embedding noise, incl. anchor-targeted at O1 | reaching the bar needs *infinite* perplexity | measured, parent still rank 1 |

And the fraction-sweep decomposition shows EAS destruction is **necessary**: WVC alone leaves
σ_id ≥ 0.7898, EAS alone ≥ 0.6393, both together 0.4299. So Exit 3 does not rest on a failed search
over a hand-picked lattice — every route to the 0.36-weight signal is closed for a distinct, named
reason, and the signal is provably required.

### The decomposition predicts the composition exactly

`M2t` attacks EAS, `X1a` attacks WVC; they are orthogonal by construction. Composing them tests
whether the detector is really the additive model the sweep implies:

| composition | σ_id | EAS | WVC | tier | rank | fp32 max Δlogit | outcome |
|---|---|---|---|---|---|---|---|
| baseline | 0.9992 | 0.9997 | 0.9970 | 1 | 1 | — | — |
| `M2t(0.0)` | 0.9740 | 0.9297 | 0.9970 | 1 | 1 | 0.000e+00 | RESISTED |
| `X1a` | 0.9116 | 0.9997 | 0.5798 | 1 | 1 | 5.34e-05 | RESISTED |
| `M2t ∘ X1a` | **0.8864** | 0.9297 | 0.5798 | 1 | 1 | 5.34e-05 | RESISTED |
| `M2t ∘ X1a ∘ M1t ∘ M3t` | **0.8864** | 0.9297 | 0.5798 | **3** | 1 | 5.34e-05 | RESISTED |

Additivity predicts 0.9992 − 0.0252 − 0.0876 = **0.8864**; observed **0.8864**. The detector is a
linear model in its signals, and we can now predict any composition's score without running it.

Even with the metadata gate broken to tier 3, the flagship lands at 0.8864 against a 0.5298 bar with
the parent still at **rank 1** and π = 100.0. Every point is exact at κ = 0.

**The frontier, derived rather than asserted.** Sweeping the permutation *fraction* inside the gate arm
gives an exact linear decomposition. EAS is **bit-identically invariant** across 0%→100% permuted
(range 0.0e+00). WVC falls linearly (R² = 0.99992) and σ_id falls linearly (R² = 0.99991) — and the σ_id
slope equals `w_WVC × WVC slope` to four decimals (−0.087233 observed vs −0.087248 predicted, ratio
0.9998, residual 5.8e-05). **One hundred percent** of the detector's response to the strongest exact
symmetry comes from the single signal carrying 0.21 of the weight; the other 0.79 does not move at all.

Extrapolating that closed decomposition to signal destruction, against the 0.5298 bar:

| grant the adversary | σ_id floor | evades? |
|---|---|---|
| full permutation (observed) | 0.9116 | no |
| perfect **WVC** destruction | 0.7898 | no |
| perfect **EAS** destruction | 0.6393 | no |
| **both**, perfectly | **0.4299** | **yes**, by 0.0999 |

So evasion at κ=0 requires zeroing EAS **and** WVC jointly; neither alone reaches the bar. EAS destruction
was independently measured to cost *infinite* perplexity, so the joint condition is unreachable without
training compute. *(This supersedes the earlier "σ_id ≥ 0.6338" estimate, which it reproduces at 0.6393.)*

This also retires the Alg-2 exhaustiveness gap for this claim: the bound holds for **any** transform that
leaves EAS invariant, enumerated or not — and MLP permutation, head permutation and norm scaling all
provably do, because none touches the embedding matrix. Enumeration was never the right argument;
invariance is.

**Why:** EAS is a Pearson correlation over cosine self-similarities; END a histogram of row norms.
Both are invariant to every function-preserving transform of a residual transformer — verified
architecturally (per-row embedding rescaling is *not* exact: pythia and bloom fail for two different reasons).

## The attack that actually works is one missing JSON key

Every weight transform failed. The transform that succeeds is **omitting a field**.

MPK crashes outright on a `config.json` whose `architectures` is `null`, deep in Pydantic
validation (`t should be a valid list [type=list_type, input_value=None]`). It does not
degrade, decline to answer, or fall through to the weight signals — it raises and produces
no verdict at all. A publisher who wants to be unverifiable does not need our symmetry
lattice; they need to delete one key.

**Six independent instances**, every one met incidentally while assembling other arms rather
than by hunting for them:

| model | publisher | `architectures` |
|---|---|---|
| `microsoft/MiniLM`-derived checkpoints | Microsoft | `null` |
| `nreimers/TinyBERT`-class checkpoints | community | `null` |
| `huawei-noah/TinyBERT_General_4L_312D` | Huawei Noah's Ark Lab | `null` |
| `google/bert_uncased_L-4_H-256_A-4` | **Google — first-party** | `null` |
| **`allenai/cs_roberta_base`** | **AllenAI** | **key absent** |
| **`allenai/biomed_roberta_base`** | **AllenAI** | **key absent** |

Four major research organisations. These are not malformed community uploads.

**And the defect has scientific consequence, not just operational.** The AllenAI pair are two
rungs of the "Don't Stop Pretraining" release — the canonical controlled ladder (one parent, one
procedure, four domains) for studying domain-adaptive pretraining, and precisely the instrument
this project needs to measure how corpus distance moves the detector. **MPK cannot be evaluated on
the standard research artifacts for the question it is meant to answer.** We hit this by doing
ordinary science with the obvious dataset, not by looking for it.

The repair is the demonstration: writing the single missing key into a local copy — weights
untouched — restores verifiability completely. One key is the whole difference between a model
that can be checked and one that cannot.

Cost comparison, which is the whole point of the paper: every κ=0 weight transform we built
leaves σ_id ≥ 0.79 and the parent at rank 1. Deleting `architectures` yields **no verdict at
all**, costs zero compute, requires no ML knowledge, and is indistinguishable from
carelessness. **Availability, not evasion, is the cheap attack on lineage verification** —
and it is the one the threat model does not consider.

## Pricing C1 at κ>0 — what actually moves the detector

Every κ=0 transform failed, so the remaining question is what the *paid* frontier costs. Seven
benign published children of `roberta-base`, with an independent x-axis: relative weight
displacement `‖θ_c − θ_p‖_F / ‖θ_p‖_F`.

| domain | disp | **disp_emb** | σ_id | EAS | END | verdict |
|---|---|---|---|---|---|---|
| DAPT news | 0.141 | 0.116 | 0.7869 | 0.9988 | 0.9933 | Confirmed |
| DAPT biomed | 0.209 | 0.170 | 0.7827 | 0.9970 | 0.9780 | Confirmed |
| DAPT cs | 0.280 | 0.236 | 0.7807 | 0.9939 | 0.9766 | Confirmed |
| DAPT reviews | 0.334 | 0.319 | 0.7657 | 0.9781 | 0.9301 | Confirmed |
| twitter (continued-pretrain) | 0.503 | 0.579 | 0.6899 | 0.8802 | 0.7066 | Confirmed |
| SecureBERT (cyber) | 0.712 | 0.547 | 0.7767 | 0.9818 | 0.9966 | Confirmed |
| **twitter (+sentiment)** | 0.733 | **0.830** | **0.5375** | 0.7346 | **0.1879** | **Not Matched** |

**It is not corpus distance.** The DAPT ladder spans 2.4× in displacement across four domains and
moves σ_id by 0.02. *(An earlier draft claimed corpus distance was the variable, inferred from two
points; the controlled ladder refutes it — see CORRECTIONS.md #13.)*

**It is embedding displacement** — R² = 0.730 vs 0.441 for overall displacement. Mechanically
forced: EAS (0.36) + END (0.19) = **0.55 of the identity weight reads only the embedding matrix**,
and WVC stays in 0.163–0.179 across the entire sweep, contributing nothing to the variation.

**But distance alone is not sufficient, and the residual is the useful part.** SecureBERT and
twitter-base moved their embeddings almost equally (0.547 vs 0.579) to different ends (σ_id 0.777 vs
0.690) — because SecureBERT *preserved the norm distribution* (END 0.9966) while twitter-base
reshaped it (0.7066). END is the signal that collapses hardest across the sweep, 0.9933 → 0.7066 →
0.1879. So the operative condition is not "move the embedding far" but **"reshape its norm
distribution."**

**Neither stage alone breaks it.** Continued pretraining reaches σ_id 0.690 and stays Confirmed;
adding the classification fine-tune reaches 0.5375 and is rejected. The false negative is a
*cumulative* two-stage effect.

### C1 survives, and we report that as found

Against an empirical from-scratch anchor — four independent Pythia training runs, mean embedding
displacement **1.2044** (analytic uncorrelated bound 1.4142):

| threshold | crossed at disp_emb | vs from-scratch (1.2044) | reliability |
|---|---|---|---|
| MPK's own `Not Matched` line (σ_id < 0.65) | 0.676 | **56%** | **interpolated** — inside the data |
| our predeclared evasion bar (null p75, 0.5298) | 1.097 | 91% | extrapolated past the last point |

*(Both crossings are fit on the **embedding** axis and so are compared to the **embedding**
from-scratch anchor, 1.2044. An earlier draft compared them to the *overall* anchor 0.9533 and
reported 70% / 112% — mixing axes, and flattering C1. See CORRECTIONS.md #14.)*

Driving a model into the null distribution costs roughly a from-scratch run (91%, extrapolated), so
**C1 is approximately right on its strict reading.** But *operational* evasion — getting the deployed
tool to answer "Not Matched" — arrives at **56%** of that, and a benign public release already sits
there.

**Pooling both parent families on verified lineage (n = 12) gives the sharper statement:**

| | |
|---|---|
| recall on true derivatives | 8/9 = **89%** |
| specificity on non-derivatives | 1/3 = **33%** |

The detector recognises genuine children well and **fails on two of three models that were never
derived at all** — `BiomedBERT` (from scratch, Confirmed Match) and `Bio_ClinicalBERT` (a BioBERT
descendant, i.e. bert-base-*cased* lineage, Weak Match). Embedding displacement predicts σ_id across
both families (R² 0.780 vs 0.557 for overall), and the in-family from-scratch controls (1.1198,
1.4527) bracket the independent-run anchor measured in a third family (1.2044).

**C1 is approximately right about the price and entirely wrong about who pays it.** The threat model
imagines an adversary spending to evade. The models that actually occupy the undetected region are
ordinary two-stage adaptations with no such intent.

*Caveats (`M5/CAVEATS.md`), all load-bearing: n = 7, one parent family, and the four DAPT verdicts
are **gate-decided artifacts of our own `architectures` repair** — writing a value that matches
`roberta-base` forces tier 1 and pins the verdict. That is the §M3 circularity result demonstrated
constructively, and it is exactly why σ_id, not the verdict, is the dependent variable throughout.*

## The duality — the unifying idea

**EAS is simultaneously the signal most robust to laundering and the one chiefly responsible for the
false positives.** It resists perturbation because it measures a stable consequence of the *training
recipe* rather than of inherited weights — and it false-positives on independent same-recipe models
for exactly that reason.

Hardening it cannot fix the false positives; making it discriminative necessarily makes it attackable.

## Act III — the tension is not fundamental

| statistic | robust to laundering? | sound? |
|---|---|---|
| SVD-spectral (perm-invariant) | ✅ 0.9955 | ❌ unrelated 0.946–0.985 vs derived 1.000 |
| raw positional (perm-sensitive) | ❌ 0.996 → 0.241 | ✅ 0.996 vs 0.0003 |
| **LAP + column normalisation** | ✅ **recovers to 0.9969** | ✅ **derived 0.515–0.997 vs not 0.117–0.215** |

**Margin +0.2996** across fine-tune, multitask *and* distillation — versus **+0.0117** for MPK's best
signal on an easier corpus. Any threshold in (0.2151, 0.5148) classifies all 9 pairs correctly.

The technique is **AWM's LAP (ICLR 2026) and REEF's CKA (ICLR 2025)** — public before MPK shipped in
April 2026. The claim is a **deployment gap**, constructively demonstrated.

### The adaptive attack, and the design rule it yields

A **new exact κ=0 symmetry** — QK invariance, `W_q → W_q M`, `W_k → W_k M⁻ᵀ` for arbitrary invertible
`M` (the attention bilinear form is GL-invariant; verified exact to 2.7e-03 at log σ = 0.3) —
**collapses LAP on attention matrices from 1.0000 to 0.2160**, while LAP on MLP stays at 1.0000.

An assignment problem undoes a *permutation*; it cannot undo a general linear *mixing*.

> **Design rule: align only on matrices whose symmetry group is permutation-only — those constrained
> by an intervening nonlinearity. Never on matrices related by a bilinear invariance.**

MLP intermediate: permutation × positive diagonal (the nonlinearity forbids mixing) → covered.
Attention head axis: full general linear group → not coverable by alignment. My defence reads MLP and
so satisfies the rule — but I picked MLP for convenience before understanding why it mattered.

## Claimed lineage at hub scale (M3)

A **`base_model` adoption curve**, four samples of 3000: newest **86.3%** · recent 81.4% ·
download-head 60.6% · most-liked (oldest-skewed) **44.6%**. Likes accumulate over time, so the
ordering is *temporal*.

**This refutes my own M1 hypothesis** that claim *absence* dominates — 80–94% of likely derivatives
declare a parent. My M1 six-model sample was entirely from the old tail of this curve.

**The reframing:** if 86% of new models declare a parent, the problem isn't missing claims — it's that
a declared `base_model` is an *unauthenticated self-assertion*, and the tool built to check it confuses
recipe with derivation in 4/5 families. High and rising declaration; a verification layer that can
neither confirm a true claim nor refute a false one.

Incidental: **1.2–1.8% declare multiple parents, up to 8** — Constitution §10.2's open problem, with a
measured floor. MPK's `compare` is pairwise and has no multi-parent verdict.

## Consequence — it runs the opposite way to the threat model

Constitution §2.2 **C5** makes provenance transitive, so a false edge **merges two lineage trees**.
Over-attribution is self-amplifying; a missing edge is self-limiting. Details in `M4/CONSEQUENCE.md`;
the load-bearing parts are the measured 4/4 rate and C5 — the consumer behaviours are inference from
Cisco's published descriptions, not instrumented, and are labelled as such.

---

## Corrections made during the work (all recorded, none buried)

1. **"EAS separates by ~zero" was too strong.** On the full 20-pair corpus it separates by +0.0117.
   Its failures are *concentrated in the same-recipe regime*. Sharper claim, properly scoped.
2. **"Untied embeddings admit an exact per-row rescaling" was wrong.** They don't — the unnormalised
   embedding enters the residual stream. This *strengthened* the result: the floor is architectural.
3. **MFI tier column was contaminated by my own harness** — `verify_exact` mutated a shared tokenizer's
   truncation state, which `save_pretrained` serialised, changing `family_hash`. MPK is deterministic
   (8/8). Also a finding: a runtime-config field moves the gate.
4. **A bug in my own defence** — subsampling rows before LAP alignment made Qwen read 0.4012 instead
   of 0.9969, which would have suggested the defence was architecture-specific.
5. **The perplexity probe was too easy** — noise *improved* it (ρ > 1); replaced.
6. **bf16 rounding erased sub-4e-3 perturbations**, making two sweep rows bit-identical.

## Status

- **M1** complete · **M2** complete (gate G1 → Exit 3, bound now *derived*) · **M4** core complete
- **Statistics** complete — see `STATISTICS.md`. Preregistered family {C4, C5} decided: C5 rejects
  (Holm p = 0.0478, thin); C4 retains H₀ and its predeclared equivalence margin (δ=0.05) **fails**.
- **Gate G0 PASSES**: 12/12 tier-3 positives, 11 P~ pairs (≥8), 5 families (≥3).
- **All declared §9.1 rungs run**, including `M2t` — the last one.
- **Audit gaps still open**, stated rather than quietly dropped:
  - **Stage 4** (1.4B transfer set, ≤20 frozen compositions) — never run
  - **Alg 2** was never exhaustive — but the decomposition makes exhaustiveness unnecessary for the
    Exit-3 claim, which rests on EAS-invariance and now predicts any composition's score additively
  - κ>0 **corpus-distance sweep** — newly motivated by the twitter-roberta false negative; the single
    highest-value next experiment, since it is the axis on which the detector actually fails
  - C5 survives Holm by 0.002; C4's predeclared equivalence margin failed. Both need more pairs.
- **Not sent:** `M4/DISCLOSURE_DRAFT.md` is a draft; transmission is a PI decision under D2. It now
  understates the case — it predates the false negative and the fourth crash instance.

## Map
```
FINDINGS.md                  this file
research_design_detailed.md  the pre-registered design (Rev 2, κ=0 frontier)
M1/RESULTS.md                oracle acquisition, scoop audit, baseline
M2/RESULTS.md                the κ=0 laundering frontier + corrections
M4/RESULTS.md                the defence
M4/CONSEQUENCE.md            attestation-stack trace
M4/DISCLOSURE_DRAFT.md       coordinated disclosure (NOT SENT)
```
