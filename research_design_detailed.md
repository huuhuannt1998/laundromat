# LAUNDROMAT — Detailed Technical Research Design

**Project** `prj_01M0E1SKSP6CSEHY9DTT4V9MB2` · **Venue** IEEE S&P 2027 Cycle 2 · **Manuscript** `man_01M0E2J8TA2TGE0BW299B0TQMW`
**Drafted** 2026-08-19 after the M1 gate · **Gate resolution** `chk_01M0E5H3NYK5BTRAKPN529QTBX` (Option B ratified)
**Revision 2** — rewritten under the binding hardware constraint below. Revision 1 assumed training-based
transform arms were affordable; they are not, and every section that depended on them has been replaced.

---

## 0. The binding constraint, stated first because it determines the design

**There is no hardware for this project.** No GPU, no cluster, no cloud credits, no paid compute. The
entire project runs on one machine: **Apple M4, 10 cores, 24 GB RAM, CPU only, ~60 GB free disk.**

This is not a limitation to be apologised for in §11. It is the **design's organising principle**, and it
converts the project into a sharper paper than the one Revision 1 described:

> Every transform we can actually execute costs **exactly zero training compute**. So the paper's measured
> claim is about **the κ = 0 frontier** — what a publisher who runs no training at all can achieve.

That is the strongest possible position from which to test Cisco's Constitution C1, *"fully evading all
weight-level signals is as expensive as training from scratch."* We do not need to price the whole curve
to falsify that. We need one point at κ = 0.

**What this rules out, permanently, and is declared out of scope rather than deferred:** continued
pretraining, self-distillation, pruning-with-healing, any gradient-based transform, any transform requiring
a backward pass, any from-scratch training baseline, any model above ~1.4B, and any hub-scale sweep.
These appear in the paper only as an **analytic upper bound with citation** (§9.2), never as a measured arm.

**What survives, and is fully executable on the machine above:** exact function-preserving weight-space
symmetries (permutation, scaling, rotation), quantize/dequantize round-trips, calibrated noise injection,
architecture-metadata edits, tokenizer remaps, and every composition of these. All are pure tensor
manipulations plus file writes. Capability verification for exact symmetries requires **forward passes
only**, and for the exact transforms is a *numerical check of a proven property*, not an empirical estimate.

---

## Status and discipline

| Thrust | Type | Role |
|---|---|---|
| **T1 — Detector soundness, no adversary present** | **Measurement study** | **Act I. This is the paper's floor.** |
| **T2 — The κ = 0 laundering frontier** | **Hypothesis under test** | Act II. The ceiling. |
| **T3 — Attestation consequence + scoped defence** | Analysis + constructive design | Act III. |

**Gate G1** (§13) sits between T1 and T2. **Build nothing in T2 beyond the gate instance until G1 resolves.**

> **Every number here is a target to be measured, never a result.** Figures carried from M1 are marked
> **[M1, directional]** — M1 ran on the same CPU machine with small *n*, and each such number is
> directional until re-measured under this design's Stage-0 substrate. Nothing in this document is a finding.

**Predeclaration.** The transform family and its rungs (§9.1), the property/outcome catalog (§9.2), the
severity classes (§9.3), the statistical contrasts (§11.4) and the success thresholds (§9.4) are fixed
**before** any T2 run — predeclared so that "evasion" is never defined after the fact to fit whatever the
transforms happened to achieve. M1 already showed why this matters: the obvious threshold definition of
evasion is *invalid on this detector*, because benign unrelated models already cross the threshold.

---

## Spine (one line)

> Deployed model-lineage verification cannot distinguish *shared training recipe* from *shared weights*,
> and a publisher can exploit that with transforms costing **zero training compute** — falsifying the
> vendor's stated limit that evading weight-level signals costs as much as training from scratch.

## Spine (thesis)

**(A) The primitive is unsound before anyone attacks it.** 0.79 of the deployed detector's identity weight
sits on *structural* signals — embedding-anchor geometry, embedding-norm distribution, layer-energy
profile, norm-layer vectors — that are determined by the **training recipe**, not by weight inheritance.
The single signal comparing parameter values positionally carries 0.21 and is outvoted. Consequently the
weight-decided score distributions for derived and independent pairs are only imperfectly separated
(one discordant pair; AUC 0.949, p=0.0071 — the earlier "overlap" reading is retracted), and the vendor's own
benchmark cannot surface this because it labels the one diagnostic pair as a positive.

**(B) That unsoundness is exploitable at literally zero training cost.** The adversary need not destroy the
linkage signal — C1 is right that destruction costs a retrain. It suffices to move the pair into the region
*already occupied by benign independent-same-recipe pairs*. That region exists, is populated, and (this is
the claim under test) is reachable by a composition of exact symmetries and a metadata edit, all at κ = 0
with capability preserved **by construction**, not by measurement.

**(C) The consequence lands on controls that already ship.** ML-BOM inheritance, license-obligation
tracking, inherited-vulnerability assessment and region-of-origin blocking all attach obligations to a
derivation edge. An edge that cannot separate "same recipe" from "same weights" fails in *both* directions:
under-attributing a laundered child, over-attributing an independent lab that followed a public recipe.

**Structural analogy, shape only.** The composed metadata+weight attack has the shape of a **TOCTOU** bug:
a cheap attacker-controlled proxy (`arch_hash` from `config.json`) is checked, and on a hit the expensive
authoritative check is *skipped and overridden*. We claim **no** race condition, concurrency, or
time-of-use window — only the check-substitution shape. The analogy also **fails** in an important way,
stated in §5.6: removing the gate does not fix the vulnerability, because M1 found the authoritative check
*also* fails on the diagnostic pair.

---

## 1. Problem statement and formal model

### 1.0 Informal problem

A third party is handed a model artifact and must decide whether it descends from a particular parent —
with no cooperation from anyone, no mark planted in advance, and nothing but the released files. Deployed
tools answer by measuring artifact similarity against a calibrated threshold. We ask two questions. First,
*with no adversary*: does that similarity measure descent, or something correlated with descent that a
non-descendant can also have? Second, *with an adversary who runs no training at all*: can a genuine
derivative be made undetectable for free?

### 1.1 State spaces and objects

A **model artifact** is m = (θ, c, τ) ∈ M = Θ × C × T — released weights, architecture config, tokenizer.
Artifacts are what is *published*; a third party sees nothing else.

A **derivation step** is a partial map F : M × Ξ → M; write m ⇝_F m′. **Ground-truth provenance** P ⊆ M × M
is the reflexive-transitive closure of ⇝ over actually-executed steps. P is a fact about history and is
**not observable from artifacts** — the gap the whole field lives in, conceded by Constitution §10.1.2
("provenance is binary; modification is continuous").

A **detector** is D_cmp : M × M → [0,1] × V and D_scan : M × 2^M → (M × [0,1])*, with
V = {Confirmed, HighConfidence, Weak, NotMatched}. For MPK v1.1.0:

    s_MFI(c_A,c_B) → (tier, σ_MFI)
    s_j(θ_A,θ_B) ∈ [0,1],  j ∈ J = {EAS, NLF, LEP, END, WVC}
    σ_id   = Σ_j w_j s_j / Σ_j w_j          over non-NaN j
    σ_pipe = 1.0 if tier=1 ; 0.9 if tier=2 ; σ_id if tier=3

with **[M1, directional]** w = (EAS .36, WVC .21, END .19, LEP .16, NLF .08), bands σ>0.75 →
HighConfidence, 0.65<σ≤0.75 → Weak, else NotMatched.

**The partition the paper turns on:**

    J_struct = {EAS, NLF, LEP, END},  W_struct = 0.79     (distributional / geometric summaries)
    J_raw    = {WVC},                 W_raw    = 0.21     (the only positional comparison)

### 1.2 The adversary as a constrained function

L : M → M drawn from the predeclared family 𝓛_T (§9.1), subject to four constraints, none relaxable silently:

- **(F1) Capability floor.** Q(L(m)) ≥ (1−δ)·Q(m). *Without F1 the problem is trivial: retrain from
  scratch.* **For the exact-symmetry subfamily this constraint is satisfied by construction** — the
  transforms are function-preserving — and F1 reduces to a *numerical* verification of float error (§1.5, M-5).
- **(F2) Compute budget.** κ(L) = FLOPs(L) / FLOPs_pretrain(parent) ≤ κ_max. **Under §0, every executable
  transform has κ = 0: no training, no backward pass.** κ_max is therefore not a tuning knob in this
  project; it is pinned at 0 for every measured point, and κ > 0 exists only as the analytic bound of §9.2.
- **(F3) Publishability.** L(m) must load and run under stock `transformers` / `safetensors` with no custom
  code. A laundering artifact that cannot be published is not a laundering artifact.
- **(F4) Observability.** L consumes only what its declared level authorizes (§2.3).

**Reachable set** R(m; 0, δ) = { L(m) : L ∈ 𝓛_T, κ(L)=0, Q(L(m)) ≥ (1−δ)Q(m), F3, F4 }.

### 1.3 The distinguishing formal device — the composition-necessity indicator

M1 found the MFI gate forces σ_pipe = 1.0 whenever `arch_hash` matches, regardless of weights
**[M1, directional]**. So a weight-only transform cannot move the *verdict* on a same-architecture child,
and a metadata-only edit does not move σ_id at all. Any real attack must compose — and we need a device
certifying the composition is *necessary*, or "we composed five things and it worked" is unfalsifiable.

For L = L_k ∘ ⋯ ∘ L_1, define

    χ_comp(L) = 1  ⟺  E(L) = 1  ∧  ∀i ∈ [k] : E(L^{(−i)}) = 0

where L^{(−i)} deletes component i. **χ_comp = 1 asserts the composition achieves what no proper
sub-composition achieves.** A composed result reported without χ_comp is not reportable under this design.

*Note on matched budget:* the usual worry — "the strong version merely spent more resource" — is
**structurally impossible here**, because every component has κ = 0. Deleting a component cannot reduce a
budget that is already zero. This is a genuine methodological advantage of the no-hardware constraint and
is stated as such in the paper: the necessity test is exact rather than budget-matched-approximately.

### 1.4 The novel mechanism, formalized — *recipe-convergence masking*

Write s_struct(A,B) = (s_j(A,B))_{j∈J_struct} ∈ [0,1]^4 and s_raw(A,B) = s_WVC(A,B).

> **Definition ((ε,Δ)-masking).** A detector D exhibits **(ε,Δ)-recipe-convergence masking** if there exist
> pairs (A,B) with ¬P(A,B) and (C,D′) with P(C,D′) such that
>
>     ‖s_struct(A,B) − s_struct(C,D′)‖_∞ < ε      and      |s_raw(A,B) − s_raw(C,D′)| > Δ,   ε ≪ Δ
>
> i.e. on the signals carrying W_struct a *non-derived* pair is indistinguishable from a *derived* pair,
> while the positional signal carrying W_raw separates them cleanly — and is outvoted.

**T1's job is to establish this ∃-statement on a controlled set and estimate (ε, Δ) with intervals.** M1's
single observation — pythia-1.4b vs pythia-1.4b-deduped at EAS .9838 / NLF .9735 / LEP .9811 / END .9849
against WVC .1023, σ_id = .7976 **[M1, directional, n = 1 pair at 2 scales]** — is consistent with
ε ≈ 0.02, Δ ≈ 0.89, but it is one point and an extreme one (identical init seed). It generates the
hypothesis; it is not the measurement.

> **Attack corollary — the bridge from T1 to T2.** If D exhibits (ε,Δ)-masking then the adversary needs
> only
>
>     ∃ L ∈ R(m; 0, δ) :  s_struct(parent, L(child)) ∈ conv{ s_struct(A,B) : ¬P(A,B) }
>
> Destruction is expensive and C1 is right about that. **Displacement into an already-populated region is
> the free operation, and it is the one nobody has priced.**

### 1.5 The metric instrument

Six named metrics, all defined before any scenario or stage uses them.

**M-1 — Parent rank (primary).** R(L) = |{ m′ ∈ DB : σ_pipe(m′, L(child)) ≥ σ_pipe(parent, L(child)) }|,
the 1-based rank of the true parent in the scan ranking. Rank is primary because M1 showed the threshold
formulation is invalid on this detector: a benign unrelated pair scored 0.7752 while a genuine distillation
scored 0.7300 **[M1, directional]**, and distilgpt2's true parent was *already* out-ranked by an unrelated
model with no adversary present.

**M-2 — Rank displacement.** ΔR(L) = R(L) − R(∅).

**M-3 — Null percentile.** With N the frozen benign-independent null (Alg 6), π(L) = percentile of
σ_pipe(parent, L(child)) within N. π → 50 means *typical of unrelated pairs* — the honest operational
statement of evasion.

**M-4 — Evasion indicator (predeclared).** E(L) = 1 ⟺ R(L) > k* ∧ π(L) ≤ π*, with **k* = 5, π* = 75**,
fixed now. k* = 5 because MPK's default `--top-k` is 3 and the Explorer surfaces a small ranked set: a
parent outside the top 5 is not shown to a human auditor. π* = 75 requires the laundered pair to look no
more related than the 75th-percentile unrelated pair.

**M-5 — Capability retention, with two regimes.** ρ(L) = Q(L(m))/Q(m).
- *Exact-symmetry regime (X1, X2, X3):* function preservation is a **theorem**, so what we measure is
  numerical fidelity, not capability. Report **max logit deviation** and **perplexity delta** over a fixed
  1 000-sequence probe set, forward passes only. Predeclared tolerance: max |Δlogit| ≤ 1e-2 and
  |Δppl|/ppl ≤ 1e-3 in fp32. Any transform exceeding this is reported as **NOT exact at this precision** —
  which is itself a finding worth reporting, since fp16 permutation is known to perturb outputs.
- *Inexact regime (Q1, N1, M2t):* ρ is empirical. Q = mean normalized score over the Stage-0 battery
  (§11.1), which is **CPU-feasible**: perplexity on a held-out corpus plus small fixed-size subsets of
  LAMBADA, ARC-Easy and HellaSwag. δ = **0.02** for the headline; ρ reported continuously regardless.

**M-6 — Normalized adversary cost.** κ(L) = FLOPs(L)/FLOPs_pretrain(parent). **Every measured point in
this project has κ = 0** (no training). Wall-clock and peak RSS are reported separately so no reader
mistakes "free" for "instant" — and, per §0, they are the numbers that actually bound what we can run.

**Conditionals that localize a failure.** A miss is uninformative without knowing where it happened:

    C1: gate tier change?              tier(∅) → tier(L)
    C2: which signals moved?           Δs_j per j, attributed to W_struct vs W_raw
    C3: σ_id crossed, or only σ_pipe?  separates gate-evasion from weight-evasion
    C4: rank/percentile disagreement?  R>k* but π>π* means a SPARSE DATABASE, not a robust detector

**Miss taxonomy — reported separately, always.** Collapsing these into "failed" would be the easiest way to
write a dishonest paper.

| kind | definition | reading |
|---|---|---|
| **RESISTED** | ρ within tolerance, κ = 0, but R ≤ k* | the detector genuinely held at zero cost |
| **DESTRUCTIVE** | R > k* but ρ below tolerance | the trivial reading; not evasion |
| **INFEASIBLE** | F3 violated — artifact will not load/publish | not an attack |
| **SUCCESS** | R > k* ∧ π ≤ π* ∧ ρ within tolerance ∧ κ = 0 | C1 falsified at this point |

*(Revision 1's UNAFFORDABLE class is deleted: with κ ≡ 0 there is no budget to exceed. Its role — pricing
the expensive end — is now carried by the analytic bound in §9.2, not by a measured arm.)*

### 1.6 Tractability — naming the hard problem, then bounding it

**The hard problem.** min κ(L) s.t. E(L)=1, ρ(L)≥1−δ is a bilevel optimization whose inner objective is
black-box, non-differentiable and discontinuous (tiered score with hard gates). Two sub-problems are hard
in isolation:

- **Weight-space alignment under permutation is NP-hard** — optimal neuron matching reduces to
  graph/quadratic assignment; the maximization direction we need inherits the hardness. AWM's Linear
  Assignment Problem step is a relaxation of exactly this.
- **The gate makes the objective non-monotone.** σ_pipe is not monotone in σ_id: a transform lowering σ_id
  while preserving `arch_hash` leaves σ_pipe pinned at 1.0. Local search on σ_pipe has a **plateau of
  measure one over all weight-only transforms**.

**Proposition (bounded well-posedness).** Restricting 𝓛 to the finite predeclared family 𝓛_T (§9.1) with
declared rungs and composition depth k ≤ 4 makes |𝓛_T| finite and the objective computable, hence the
optimization is a search over a finite lattice — well defined and **exhaustively solvable**.

**Consequence, and this is where §0 bites hardest.** Revision 1 planned exact enumeration at 135M and beam
search at 1.4B. **Beam search is not affordable**: a single cold `compare` at 1.4B took **303 s** on this
machine **[M1, measured]**, and cache must be disabled for laundered artifacts (§7.4), so a beam of width 8
at depth 4 would be thousands of 300-second evaluations. Therefore:

- **≤ 500M: exact exhaustive enumeration** over the declared lattice. Every point measured; no sampling,
  hence no inferential statistics needed for this arm.
- **1.4B: a predeclared confirmation set only** — the ~20 compositions that the ≤500M enumeration
  identifies as most informative, run once each. This arm is a *transfer check*, explicitly underpowered,
  and is reported as such. It is **not** a search.

This is a genuine reduction in claim strength versus Revision 1 and it is stated in the paper's limitations.

### 1.7 Notation summary

| Symbol | Meaning |
|---|---|
| m = (θ,c,τ) | published artifact: weights, config, tokenizer |
| P(A,B) | ground-truth provenance (historical fact, unobservable) |
| D_cmp, D_scan | detector, pairwise and database-scan |
| tier, σ_MFI | MFI gate tier ∈ {1,2,3}; metadata score |
| s_j, w_j | signal j ∈ {EAS,NLF,LEP,END,WVC}; identity weight |
| J_struct / J_raw | structural signals (W_struct = .79) / positional signal (W_raw = .21) |
| σ_id, σ_pipe | identity score; gated pipeline score |
| L, 𝓛_T, k | transform; predeclared family; composition depth ≤ 4 |
| κ(L) | adversary FLOPs / parent pretraining FLOPs — **≡ 0 for every measured point** |
| ρ(L) | capability retention (exact regime: numerical fidelity) |
| R, ΔR, π | parent rank; displacement; null percentile |
| E(L) | evasion indicator (R > 5 ∧ π ≤ 75) |
| χ_comp | composition-necessity indicator |
| (ε,Δ) | recipe-convergence masking parameters |
| δ | capability tolerance = 0.02 |

---

## 2. System model in the notation

### 2.1 Components

Parent registry (MPK's bundled ~150 base models, 39 families, 184 parquet deep-signal files
**[M1, measured on the installed copy]**) · Detector (MPK v1.1.0 primary; arXiv 2502.00706 black-box tester
secondary) · Publisher-adversary · Auditor · Attestation consumers (ML-BOM builder, license checker,
CVE-inheritance mapper, region-of-origin engine).

### 2.2 Actor position

The adversary sits **at publication time**, upstream of the auditor, with full white-box control of its own
artifact and zero control of the detector. It is **not** a model-serving adversary — the sharp line against
arXiv 2509.26598, whose ten attacks are inference-time output filtering and are structurally unavailable
here: you cannot token-suppress a Frobenius-norm profile of published safetensors.

### 2.3 Observability lattice

| Level | May read | Realistic? |
|---|---|---|
| **O0** | own artifact + published parent artifact | yes — baseline |
| **O1** | O0 + detector source, signal definitions, weights w_j, thresholds | **yes — MPK is Apache-2.0** |
| **O2** | O1 + unlimited black-box `compare`/`scan` queries | yes — local install |
| **O3** | O2 + full reference fingerprint database contents | yes — public on HF |
| **O4** | O3 + gradients of σ_pipe w.r.t. θ | **NO — upper bound only.** σ_pipe is non-differentiable (hard tiers, histogram binning, rank ops). Any O4 result is labelled an upper bound on adversary power, never a realistic attack. |

**Headline curve is reported at O1** — source-informed, query-free — because that is the weakest level a
real laundering publisher plainly has, so results there are least contestable. O2/O3 arms answer separately:
"what does query access buy?" Under §0 the O2/O3 arms are cheap (queries are the same `compare` calls we
already run) and stay in scope.

### 2.4 Capability record (shipped with every result)

Anything not in this record was not assumed. One per row of every results table.

```json
{
  "record_version": "2.0",
  "actor": "publisher-adversary",
  "observability_level": "O1",
  "powers": {
    "white_box_own_artifact": true, "white_box_parent_artifact": true,
    "detector_source_read": true, "detector_query": false,
    "reference_db_read": false, "detector_gradients": false,
    "training_compute": false, "backward_pass": false
  },
  "knowledge": {
    "signal_definitions": true, "identity_weights": true, "verdict_thresholds": true,
    "reference_db_membership": false
  },
  "budget": {
    "kappa": 0.0,
    "kappa_definition": "FLOPs(L)/FLOPs_pretrain(parent); 0 by construction, no training performed",
    "wallclock_seconds": 0.0, "peak_rss_bytes": 0,
    "hardware": "Apple M4, 10 core, 24 GB, CPU only, no GPU/cluster/cloud"
  },
  "constraints": {
    "capability_regime": "exact|inexact",
    "exactness_tolerance": {"max_abs_logit_delta": 1e-2, "rel_ppl_delta": 1e-3, "dtype": "fp32"},
    "capability_floor_delta": 0.02,
    "publishable_standard_toolchain": true, "custom_loader_code": false,
    "composition_depth_max": 4
  },
  "transform": {"family": [], "params": {}, "composition_order": []},
  "provenance": {"parent": "", "child": "", "ground_truth_source": "", "declared_base_model": null}
}
```

### 2.5 Trust assumptions and what is *not* silently included

We trust artifact bytes to be what the adversary published. We do **not** assume the signing stack is
absent — Sigstore, in-toto, SLSA and HF signed commits all exist and T3 addresses them head-on (§6.3). We
do **not** assume the auditor holds training data, the child's data, or intermediate checkpoints; any
scheme needing those is outside the third-party model by definition (§7 scope list). We do **not** assume
the adversary can influence the reference database — poisoning is a distinct attack, explicitly out of scope.

---

## 3. Motivation, state of the art, and the gap

**Line 1 — Owner-planted marks.** Instructional Fingerprints, Chain-and-Hash, Perinucleus, FPEdit, EditMF,
KGW-distilled watermarks; Fernandez et al. (ICASSP 2024) watermark *by* choosing a functional invariant.
*Does:* lets an owner who planted something prove theft. *Does not:* say anything about an arbitrary model
whose owner planted nothing. Fernandez additionally supplies our transform recipe and independently
establishes that permutation/scaling invariants are essentially free — which under §0 is exactly why they
are the only transforms we can afford.

**Line 2 — Adversarial evaluation of behavioural fingerprints.** Nasery et al. (arXiv 2509.26598) bypass
ten schemes. *Does:* legitimises the genre; measures utility on IFEval/GSM8K/GPQA-D/TriviaQA. *Does not:*
touch the weight artifact — every attack is inference-time output filtering by a model *server*, with no
weight transform and no publisher adversary.

**Line 3 — Third-party detectors proposed as defences.** REEF (ICLR'25), AWM (ICLR'26), GhostSpec, SELF,
Spectral Signatures (KDD'26), Stemma, MPS, FNF, Shang et al. *Do:* propose detectors and self-evaluate
robustness; REEF and Spectral Signatures explicitly test permutation and scaling; GhostSpec runs a
spectral-divergence evasion fine-tune. *Do not:* evaluate a *deployed* tool, price the attack against
pretraining, report evasion jointly with capability, or test a detector containing a metadata gate — no
academic fingerprint has one. Stemma states outright that it does not consider adaptive attackers.

**Line 4 — The deployed stack.** Cisco MPK v1.1.0 + Constitution + Provenance Explorer; CycloneDX ML-BOM,
Sigstore, in-toto, SLSA, LF Model Signing. *Does:* ship, and stake a threat model — §9 names Concealment,
Fabrication and Laundering and asserts **C1** "fully evading all weight-level signals is as expensive as
training from scratch; cheaper techniques leave residue", **C2** "residual traces of the true parent persist
and are recoverable by weight-level forensics", **C3** "robust provenance detection must rely on
weight-level signals; metadata-only systems are trivially defeated". *Does not:* evaluate any of them. The
111-pair benchmark is benign-hard, not adversarial, and exact weight-space symmetries appear nowhere in §9.

**The 5-dimension intersection held jointly, that no prior work holds:**

| # | Dimension | Prior? |
|---|---|---|
| D1 | Target is a **deployed, shipping** tool, not an academic baseline | no |
| D2 | Evasion reported **jointly with capability**, not as a binary | no |
| D3 | Adversary cost **normalized to parent pretraining** — and pinned at **κ = 0** | no |
| D4 | Consequence traced into the **attestation stack** | no |
| D5 | Detector contains a **metadata gate**, forcing a composite attack | no (structurally absent elsewhere) |

Plus the floor, orthogonal to all five: **(ε,Δ)-recipe-convergence masking as a soundness property,
measured with no adversary present.**

---

## 4. Insight

1. **The decision weight sits on the wrong quantities.** 0.79 of the identity mass rests on summaries
   determined by the *training recipe* (architecture, init, optimizer, data distribution); 0.21 on the only
   positional comparison. Two labs running the same public recipe converge on the former, not the latter.
   *Mechanism:* the detector measures a quantity whose equivalence class is "same recipe" and infers
   membership of a strictly smaller class, "same weights".

2. **The gate inverts the trust ordering the Constitution declares.** §9 says metadata-only systems are
   trivially defeated and detection must rest on weight signals. The pipeline does the reverse: when
   `arch_hash` matches, σ_pipe is *set* to 1.0 and weight signals are discarded — including when they
   disagree. `arch_hash` hashes fields the publisher writes.

3. **Evasion is displacement, not destruction — and the destination is already populated.** The negative
   distribution reaches 0.7752 while a genuine distillation sits at 0.7300 **[M1, directional]**. The
   adversary moves *into a crowd*, not across empty space. That is why the metric is rank-in-a-null, and it
   is why the attack can be free.

4. **Exact symmetries are free but nearly useless alone against *this* detector — and that is a result.**
   EAS (row-space cosines), END (row L2 norms) and LEP (Frobenius energy) are invariant to hidden-dimension
   permutation and orthogonal rotation *by construction*; only NLF and WVC (W = 0.29) can move. And nothing
   moves the verdict while tier 1 fires. So the object of interest is the **composition**, and χ_comp
   certifies it. **[Analytical prediction from M1's signal semantics; explicitly untested — S1 falsifies it.]**

5. **A benchmark that mislabels its one diagnostic pair cannot see the failure mode.** MPK's shipped
   `Benchmark_All.json` places pythia-1.4b vs pythia-1.4b-deduped in the `similar` (positive) list although
   its own `provenance` field reads `"none"`; `run_benchmark.ipynb` labels by list membership and carries
   the provenance value into a column literally named `constitution` that it never reads. The tier-1 false
   positive scores as a success — which is why the published result reports no false positives at all.
   **[M1, directional — verified by reading the shipped repo, not by re-running the 111 pairs.]**

6. **The no-hardware constraint is a methodological asset, not only a limit.** Because every executable
   transform has κ = 0, (i) F1 holds *by theorem* for the exact subfamily rather than by measurement,
   (ii) the necessity test of §1.3 is exact rather than approximately budget-matched, and (iii) the hardest
   baseline (§11.2) is won by construction. A result at κ = 0 is the strongest possible falsifier of C1.

---

## 5. Proposed system and algorithms

`SUBSTRATE` → `ORACLE` (both detectors, capability-record emitting) → `TRANSFORMS` (𝓛_T, all κ = 0) →
`SEARCH` (exact lattice) → `INSTRUMENT` (metrics, frozen null, necessity) → `DEFENCE` (RESCORE) →
`CONSEQUENCE` (attestation propagation).

### 5.1 Algorithm 1 — `EVAL`: the deterministic primitive everything calls

```
Alg 1  EVAL(parent, child, L, level, DB) -> Result
 1  assert level_authorizes(L, level)                       # F4
 2  m' <- L(child)                                          # pure tensor ops; NO training
 3  if not loads_standard_toolchain(m'): return INFEASIBLE   # F3
 4  if L in EXACT_FAMILY:                                   # F1 by theorem, verify numerically
 5      fid <- max_abs_logit_delta(child, m', probe_1000)
 6      ppl <- rel_ppl_delta(child, m', probe_1000)
 7      rho <- 1.0 if (fid <= 1e-2 and ppl <= 1e-3) else EMPIRICAL
 8  if rho is EMPIRICAL: rho <- Q(m') / Q(child)            # forward passes only
 9  cmp  <- D_cmp(parent, m', no_cache=True)                # tier, sigma_MFI, s_j, sigma_id, sigma_pipe
10  scan <- D_scan(m', DB, no_cache=True, top_k=K), K >> k*
11  R    <- rank_of(parent, scan)                           # +inf if absent
12  pi   <- percentile(cmp.sigma_pipe, NULL)                # NULL frozen by Alg 6
13  E    <- 1 if (R > k_star and pi <= pi_star) else 0
14  kind <- SUCCESS if E and rho ok ; DESTRUCTIVE if E and not rho ok ; else RESISTED
15  return Result(cmp, scan, R, pi, rho, fid, ppl, kappa=0, E, kind, C1..C4, capability_record)
```

**Determinism requirements, enforced, and each traceable to a real failure already seen:** fixed seeds for
Q; MPK cache **disabled** for every laundered artifact (a stale cache keyed on model id would silently
return the parent's features); identical dtype and device for parent and child feature extraction — this
is exactly the confound that voided M1's secondary-oracle spot check and it must not recur.

### 5.2 Algorithm 2 — `LATTICE-EXACT`: exhaustive enumeration (≤ 500M)

```
Alg 2  LATTICE-EXACT(parent, child, F, k_max=4, level, DB) -> curve
 1  curve <- {}
 2  for k in 1..k_max:
 3    for each ordered composition (f_1..f_k) from F, f_i distinct:     # metadata component LAST
 4      for each param tuple (p_1..p_k) in declared_ladder(f_1..f_k):
 5        L <- compose(f_k[p_k], ..., f_1[p_1])
 6        curve[L] <- EVAL(parent, child, L, level, DB)
 7  return curve
```

Every point measured; no sampling, hence no inferential statistics for this arm (§11.4).

### 5.3 Algorithm 3 — `CONFIRM-AT-SCALE`: the 1.4B transfer check (**not** a search)

```
Alg 3  CONFIRM-AT-SCALE(parent, child, S, level, DB) -> results
 1  # S is a PREDECLARED set of <= 20 compositions chosen from the <=500M enumeration:
 2  #   the SUCCESS set, the RESISTED boundary cases, and the chi_comp ablations of the flagship.
 3  # S is frozen BEFORE any 1.4B run. No search, no tuning, no additions after seeing results.
 4  for L in S:
 5      results[L] <- EVAL(parent, child, L, level, DB)     # ~300 s each cold on the project machine
 6  return results
```

Budget: |S| ≤ 20 × ~300 s ≈ 100 min per (parent, child) pair, plus artifact write and disk churn. With
2 pairs at 1.4B this is a few overnight runs — the honest ceiling of what §0 permits.

### 5.4 Algorithm 4 — `NECESSITY`: the anti-triviality test

```
Alg 4  NECESSITY(L = L_k o ... o L_1, parent, child, level, DB) -> (chi_comp, table)
 1  base <- EVAL(parent, child, L, level, DB)
 2  if base.E == 0: return (0, {})
 3  for i in 1..k:
 4      # Budget matching is TRIVIAL here: every component has kappa = 0, so deleting one
 5      # cannot reduce a budget. No refitting needed. The test is EXACT, not approximate.
 6      table[i] <- EVAL(parent, child, compose(L without i), level, DB)
 7  chi_comp <- 1 if all(table[i].E == 0) else 0
 8  return (chi_comp, table)
```

**The headline budget is κ**, because κ is the quantity C1 is stated in and therefore the only one that can
falsify it. Wall-clock is reported but is a property of our machine, not of the attack.

### 5.5 Algorithm 5 — `MASK-DETECT`: the masking estimator (T1's core)

```
Alg 5  MASK-DETECT(P+, P-, P~) -> (eps_hat, Delta_hat, CI, margin, per-signal AUC)
 1  for S in {P+, P-, P~}: for (A,B) in S: v[A,B] <- (s_struct(A,B), s_raw(A,B))    # cmp only
 2  eps_hat   <- min_{(A,B) in P~} min_{(C,D) in P+} ||s_struct(A,B) - s_struct(C,D)||_inf
 3  Delta_hat <- |s_raw| gap realised at the argmin of line 2
 4  CI        <- BCa bootstrap, 10000 resamples, stratified by pair set
 5  margin    <- min_{P+} sigma_id - max_{P- u P~} sigma_id      # < 0 => NO separating threshold exists
 6  return (eps_hat, Delta_hat, CI, margin, AUC_j for each j in J)
```

Line 5 replaces M1's −0.0452 margin **[RETRACTED — degenerate min−max statistic; see CORRECTIONS.md]**
with AUC + exact permutation test. **`P~` — independent-but-same-recipe
— is the arm M1 lacked (n = 1) and is the single most important addition in the whole design.** It is also
free: it needs only `compare` calls on already-published models.

### 5.6 Algorithm 6 — `NULL`: the frozen reference distribution

```
Alg 6  NULL(DB, strata) -> N          # strata = publisher x size-bucket x tokenizer-family
 1  N <- [ sigma_pipe(X,Y) for each stratum, each unordered NOT-DERIVED pair (X,Y) ]
 2  assert |N| >= 200 and every stratum contributes >= 10       # else percentiles are unreliable
 3  return N          # built ONCE, before any transform is run, then FROZEN
```

### 5.7 Algorithm 7 — `RESCORE`: the scoped defence

```
Alg 7  RESCORE(A, B) -> (sigma_def, verdict, evidence)
 1  tier, sigma_MFI <- MFI(c_A, c_B)                  # D1: the gate ADVISES, never overrides
 2  s_raw <- WVC(theta_A, theta_B)                    # D2: positional evidence is NECESSARY
 3  s_str <- {EAS, NLF, LEP, END}
 4  expl  <- P_hat( s_str | same_recipe(c_A, c_B, tokenizer_family) )   # from the frozen P~ model
 5  s_str_adj <- max(0, (mean(s_str) - expl) / (1 - expl))              # D3: recipe discount
 6  sigma_def <- w_raw' * s_raw + w_str' * s_str_adj
 7  verdict <- CONFIRMED if (s_raw >= tau_raw and sigma_def > tau_hi)   # D4: gated conjunction
 8             HIGH if sigma_def > tau_hi ; WEAK if sigma_def > tau_lo ; else NOT
 9  evidence <- {tier, s_raw, s_str, expl, s_str_adj, gate_advisory_only: True}
```

**Informal guarantee.** Under the assumption that P~ is representative of the deployment population,
RESCORE cannot return CONFIRMED for a pair whose positional agreement s_raw is below τ_raw. Since M1
observed s_raw ≈ 0.10 for an independently-trained same-recipe pair against ≈ 0.99 for a fine-tune
**[directional]**, this removes the specific false-positive modes created by the metadata gate and by
recipe convergence.

**Non-guarantees — stated as prominently as the guarantee.**

- It does **not** defeat exact weight-space symmetries. A permutation drives WVC toward the unrelated
  regime while preserving function, so RESCORE — by *increasing* reliance on positional evidence — is
  **strictly more** vulnerable to permutation than MPK is. Composing it with an alignment step (LAP, as in
  AWM) is necessary and is **not** evaluated here: under §0 we cannot afford the alignment search. Named as
  required future work, not claimed.
- It does **not** cover distillation, where positional agreement is legitimately near zero for a genuine
  derivation. RESCORE will under-detect distillation; MPK's structural signals were the only thing catching
  it (gpt2→distilgpt2 at 0.891 with WVC = NaN **[M1, directional]**). A real regression, reported as one.
- It offers **no** guarantee against an adversary who knows RESCORE and adapts. No adaptive-attack
  evaluation is in scope — claiming otherwise would repeat exactly the mistake this paper documents in others.
- It does **not** address multi-parent merges, MoE component-level provenance, or federated settings.
- It provides **no cryptographic guarantee**. It is a scoring correction, not attestation.

*Where the TOCTOU analogy fails, as promised:* fixing the check-substitution (D1, gate advisory only) does
**not** fix the vulnerability, because M1 found σ_id = 0.7976 — above threshold — on the diagnostic pair.
The authoritative check is itself wrong. That is why D3, the recipe discount, is the load-bearing component.

---

## 6. Methods by thrust

**6.1 T1 (Act I, the floor).** Build the Stage-0 substrate with three arms: P+ derived, P− independent
different-recipe, **P~ independent same-recipe**. Run Alg 1 cmp-only over all pairs; Alg 5 for (ε̂, Δ̂) with
BCa intervals, the separability margin, per-signal AUC and the W_struct/W_raw decomposition; Alg 6 to
freeze N. Re-audit the shipped benchmark labelling (Insight 5) and report the corrected confusion matrix
*without* re-running all 111 pairs, stating that limitation explicitly. Replicate on the secondary oracle
after the same-device fix, so the claim is cross-substrate or is honestly reported as weight-space-only.
**Entirely `compare`-based on already-published models: fully affordable under §0.**

**6.2 T2 (Act II, the ceiling).** Gate instance G1 first, nothing else until it resolves. Then Alg 2 at
≤500M for the exact κ = 0 frontier; Alg 3 for the frozen 1.4B transfer set; Alg 4 on every SUCCESS. Report
(R, π, ρ, fidelity) at κ = 0, O1 headline with O2/O3 deltas.

**6.3 T3 (Act III).** Trace one derivation edge through four consumers — ML-BOM inheritance, license
obligation, inherited-CVE mapping, region-of-origin policy — stating for each what a laundered edge and a
recipe-convergence false edge cost, in both directions. Position explicitly against the signing stack:
**Sigstore/in-toto/SLSA attest who published what, not what it descends from; a laundered artifact can be
perfectly signed.** That sentence is the paper's answer to "doesn't signing solve this?" and must be
defended, not asserted. Then implement Alg 7 and report its regressions as loudly as its gains.

---

## 7. Challenges and mitigations

1. **No hardware (§0).** *Threat:* the design cannot execute any training-based transform, any from-scratch
   baseline, or any search at 1.4B. *Mitigation:* restrict the measured claim to the κ = 0 frontier and say
   so in the title if necessary; move all training-based transforms to an analytic bound with citation
   (§9.2); replace 1.4B search with a frozen ≤20-composition transfer set (Alg 3); state the reduction in
   claim strength in the limitations section rather than hiding it in a footnote. *Why this is survivable:*
   a κ = 0 result is the **strongest** falsifier of C1, not the weakest, so the constraint costs
   generality, not force.

2. **Small-n positives.** *Threat:* the tier-3 separation claim rests on 3 weight-decided positives, and
   gate G0's own criterion of ≥12 was **not met when G0 was passed** — a process failure, logged in
   CORRECTIONS.md. *Mitigation:*
   Stage 0 requires ≥ 12 tier-3 positives — vocabulary-extended, depth-pruned, distilled and
   continued-pretraining children **that already exist as published models** (we consume other people's
   training, we do not perform it). Gate G0 blocks T1 reporting until met.

3. **Ground-truth construction.** *Threat:* only 2 of 6 canonical derivations declare `base_model`
   **[M1, directional]**, so labels come from prose. *Mitigation:* three-source rule — a pair enters P+ only
   with (i) an explicit statement in the model card or originating paper **and** (ii) either a declared
   `base_model` field or independent corroboration (e.g. MPK's own README using the pair as a canonical
   example). Every label ships its source string. Ambiguous pairs are excluded, not guessed.

4. **Cache and dtype contamination.** *Threat:* MPK's cache keys on model id, so a laundered artifact
   reusing an id silently returns parent features; the same class of bug (CPU child vs author-cached GPU
   parent) voided M1's secondary-oracle spot check. *Mitigation:* `--no-cache` mandatory for laundered
   artifacts; always written to a fresh local path; Stage-0 assertion that `EVAL(parent, child, identity)`
   reproduces published-artifact numbers exactly; secondary-oracle parent cache regenerated on this machine.

5. **Non-monotonicity stalls any σ-guided search.** *Threat:* σ_pipe is pinned at 1.0 across all weight-only
   transforms — a plateau of measure one. *Mitigation:* we do not run σ-guided search at all (Alg 2 is
   exhaustive, Alg 3 is a frozen set); composition order constrains the metadata component to last, so the
   weight component's effect is exposed rather than masked.

6. **Disk exhaustion.** *Threat:* a laundered 1.4B artifact is ~2.8 GB and the machine has ~60 GB free;
   Alg 2's lattice generates hundreds of artifacts. *Mitigation:* artifacts are generated, evaluated and
   **deleted** within one `EVAL` call; only the Result and its capability record persist. At most 3 large
   artifacts on disk at once, asserted at runtime.

7. **The forthcoming-Cisco-paper risk.** *Threat:* the Constitution blog states it "builds on forthcoming
   work … including empirical evidence", which may contain the adversarial evaluation we claim is absent.
   *Mitigation:* the floor (T1) does not depend on the attack's novelty — it is a soundness measurement.
   Monitor monthly. Coordinated disclosure under D2 is scheduled **early**, precisely so the timestamp is
   unambiguous, with the benchmark labelling defect sent first as a good-faith opener.

8. **Defence overclaiming.** *Threat:* proposing RESCORE and evaluating it only on the attacks it was
   designed for is the exact failure this paper documents in others. *Mitigation:* the non-guarantee list
   (§5.7) is a required subsection; RESCORE's permutation behaviour is reported as a **strict worsening**;
   no adaptive-attack claim is made.

**Explicitly out of scope, and must not be silently assumed:** database poisoning; fabrication (forging a
derivation edge that does not exist); multi-parent merges, MoE component provenance, federated and NAS
lineage; models above ~1.4B; every training-based transform (§0); hub-scale sweeps; any scheme requiring
the auditor to hold training data or checkpoints; behavioural fingerprints as a *primary* target.

---

## 8. Contributions

| | Contribution | Survives the gate? |
|---|---|---|
| **C-A** | **(ε,Δ)-recipe-convergence masking** — definition, estimator (Alg 5), and measurement that deployed lineage verification cannot separate shared recipe from shared weights; plus the non-separability margin. | **Yes — unconditional. The floor.** |
| **C-B** | **Benchmark-validity audit** of the shipped 111-pair evaluation, showing the one diagnostic pair is labelled a positive so the tier-1 false-positive mode is invisible to the reported accuracy. | **Yes — unconditional.** |
| **C-C** | **A rank-based evasion instrument** (M-1…M-6, Alg 6) with the miss taxonomy, replacing the threshold formulation M1 showed is invalid on this detector. | **Yes — methodological.** |
| **C-D** | **The κ = 0 laundering frontier** — which zero-training-compute compositions evade, with necessity certified exactly by χ_comp. | Conditional on G1 exit. |
| **C-E** | **An empirical verdict on Constitution C1/C2/C3** — the first adversarial evaluation of a deployed provenance stack. | Direction conditional; *a* verdict is delivered on every exit. |
| **C-F** | **Attestation-stack consequence analysis** + **RESCORE** with a stated soundness bound and explicit non-guarantees. | **Analysis half unconditional.** |

Four of six stand regardless of the gate. That is the design's insurance, and under §0 insurance matters more than usual.

---

## 9. Predeclared catalog

### 9.1 Transform family 𝓛_T — **executable arm, every entry κ = 0**

| id | family | declared rungs | exact? | observability |
|---|---|---|---|---|
| **X1** | hidden-dim permutation, propagated | fraction ∈ {0.25, 0.5, 1.0} | **yes (theorem)** | O0 |
| **X2** | scaling / unscaling invariance | subset ∈ {attn, mlp, all} | **yes (theorem)** | O0 |
| **X3** | rotation in invariant subspaces | rank ∈ {low, full}, where architecture admits | **yes (theorem)** | O0 |
| **M1t** | config metadata edit (break `arch_hash`) | fields ∈ {dims, model_type, arch-name, rope} | n/a (weights untouched) | O1 |
| **M2t** | tokenizer remap / vocab reorder | overlap target ∈ {0.9, 0.5} | no — ρ measured | O0 |
| **Q1** | quantize → dequantize round-trip | {int8, int4} | no — ρ measured | O0 |
| **N1** | calibrated noise injection | σ/‖θ‖ ∈ {1e-4, 1e-3, 1e-2} | no — ρ measured | O0 |

Composition depth k ≤ 4; metadata component applied **last**. Rungs are fixed; **no rung is added after
seeing results.**

### 9.2 Out-of-budget transforms — analytic bound only, never a measured arm

Continued pretraining, self-distillation, pruning-with-healing and any gradient-based transform are
**unaffordable under §0**. They enter the paper only as a cited upper bound: the literature already
establishes that sufficient continued pretraining or distillation destroys weight-space linkage (REEF's
700B-token and 90%-pruning arms; GhostSpec's spectral-divergence fine-tune; Cisco's own 4 benchmark
misclassifications, all "extreme architectural transformations"). The paper states plainly: *the expensive
end of the curve is known to succeed and is not our contribution; our contribution is that the cheap end —
κ = 0 — also succeeds (or does not).* No number from that literature is presented as ours.

### 9.3 Pair-set templates (ground truth)

- **P+ (derived)** — declared or documented parent→child under the three-source rule; sub-templates:
  instruction-FT, multitask-FT, distillation, vocab-extension, depth-prune, continued-pretraining.
  **≥ 12 must be tier-3** (`arch_hash`-breaking). All are *already-published* models.
- **P− (independent, different recipe)** — different publisher, architecture, tokenizer.
- **P~ (independent, SAME recipe)** — same architecture and hyperparameters, independent runs:
  pythia-*/pythia-*-deduped, sibling OLMo runs, multi-seed runs of one open recipe, independent
  reproductions (OpenLLaMA vs Llama; Constitution §8 lists these as Independent). **≥ 8 pairs, ≥ 3 families.**
- **P= (identity)** — control, must return 1.0.
- **P⊥ (same-tokenizer trap)** — shared tokenizer, independent training (gpt2 vs gpt-neo-125m).

### 9.4 Severity classes for a linkage error (T3)

| class | definition | attestation consequence |
|---|---|---|
| **S0 cosmetic** | verdict band changes, parent still rank 1 | none |
| **S1 advisory** | parent leaves rank 1 but stays ≤ k* = 5 | human auditor still sees it |
| **S2 operational** | parent leaves top-k*, ML-BOM edge omitted | license obligation and CVE inheritance silently dropped |
| **S3 policy** | S2 **and** the artifact clears a region-of-origin or license-class filter it should not | control bypassed end to end |
| **S4 inverted** | a *false* edge asserted to a wrong parent | out of scope; recorded if observed |

### 9.5 Frozen thresholds

k* = 5 · π* = 75 · δ = 0.02 · exactness tolerance max|Δlogit| ≤ 1e-2, |Δppl|/ppl ≤ 1e-3 (fp32) ·
composition depth ≤ 4 · |N| ≥ 200, ≥ 10 per stratum · bootstrap 10 000 BCa · **κ ≡ 0 for all measured points**.

---

## 10. Scenarios

### S1 — Free symmetry alone (the null-result scenario we expect to fail, and must)

*Path:* X1 (full propagated permutation). *k* = 1. *Setup:* SmolLM2-135M → -Instruct, O0, κ = 0.
*Purpose:* test Insight 4.

```
CLEAN   tier=1  sigma_MFI=1.00  EAS=.999 NLF=.99 LEP=.99 END=.99 WVC=.99
        sigma_id=.999  sigma_pipe=1.00  Confirmed   R=1  pi=100  rho=1.00 (exact)  kappa=0
ATTACK  tier=1  sigma_MFI=1.00  EAS=[invariant] NLF=[drop] LEP=[invariant] END=[invariant] WVC=[drop]
        sigma_id=[falls by at most the W=0.29 carried by NLF+WVC]
        sigma_pipe=1.00   <-- GATE PINS IT      Confirmed   R=1
        E=0  kind=RESISTED    C1: tier 1->1 unchanged    C3: sigma_id moved, sigma_pipe did not
```

*Predicted:* **RESISTED.** If S1 succeeds, Insight 4 is wrong and so is the design's premise about the
gate — report it immediately as the headline and re-center §5.

### S2 — Composed metadata + symmetry (**the flagship, and the gate instance G1**)

*Path:* X1 ∘ X2 ∘ M1t, metadata last. *k* = 3. *Property:* S2 operational — ML-BOM edge omission.
*Setup:* Qwen2.5-0.5B → -Instruct and SmolLM2-135M → -Instruct, O1, κ = 0, ρ = 1.00 by theorem.

```
CLEAN   tier=1  sigma_pipe=1.00  Confirmed  R=1  pi=100  rho=1.00  kappa=0
ATTACK  tier=3   <-- M1t breaks arch_hash; the gate no longer fires
        EAS=[invariant, high]  END=[invariant, high]  LEP=[invariant, high]   <-- W_struct=.79 SURVIVES
        NLF=[low]  WVC=[low]                                                   <-- W=.29 moved
        sigma_id = .79-weighted high + .29-weighted low ;  sigma_pipe = sigma_id
        R=?  pi=?   rho=1.00 (verified numerically, not estimated)  kappa=0
        E=?   <-- THE OPEN QUESTION. This is G1.
```

**Why no simpler version reaches it.** X1/X2 alone: the gate pins σ_pipe = 1.0 (S1). M1t alone: tier drops
to 3 but σ_id is untouched at ~0.999, so R = 1. Only the composition both *exposes* σ_id and *depresses* it.
**Necessity:** Alg 4, and here the budget match is exact — all three components cost zero, so "it merely
spent more" is not available as a reading.

### S3 — Displacement into the masked region

*Path:* X1 ∘ N1 ∘ Q1 ∘ M1t. *k* = 4. *Setup:* pythia-410m child, O1, κ = 0 (N1 and Q1 are tensor ops, not
training). *Target:* not σ → 0, but s_struct(parent, L(child)) entering conv(P~) — making a derived pair
*look like two independent runs of the same recipe*. This is the scenario that operationalises the attack
corollary of §1.4, and it is the one where ρ must be measured rather than proven.

```
CLEAN   tier=1  sigma_pipe=1.00  R=1  pi=100
ATTACK  tier=3  s_struct -> [target: within eps of P~ centroid]   s_raw -> [target: P~ regime]
        R=[target > 5]  pi=[target <= 75]  rho=[MEASURED, must be >= 0.98]  kappa=0
        E=?  C2: attribution of the sigma_id drop across W_struct vs W_raw
```

### S4 — Recipe-convergence false positive (T1, no adversary at all)

*Path:* none. *Setup:* every P~ pair. *Property:* S2 in the **over**-attribution direction — an independent
model wrongly inheriting a parent's license obligations and CVEs.

```
CLEAN   (there is no attack; both artifacts are exactly as published)
        pythia-1.4b vs pythia-1.4b-deduped
        tier=1  sigma_MFI=1.00
        EAS=.9838  NLF=.9735  LEP=.9811  END=.9849    <-- W_struct=.79, all near-identical
        WVC=.1023                                      <-- W_raw=.21, correctly near-zero, OUTVOTED
        sigma_id=.7976 (ABOVE the .75 threshold)  sigma_pipe=1.00  Confirmed
        GROUND TRUTH: NOT DERIVED. Constitution S8 lists this pair under "Independent".
        [M1, directional -- n=1; T1 must establish this across the full P~ arm]
```

*(Revision 1's distillation-ceiling scenario is deleted — it required training compute. Its role, pricing
the expensive end, is now §9.2's analytic bound.)*

### 10.5 Scenario → instrument map

| Scenario | M-1 R | M-2 ΔR | M-3 π | M-4 E | M-5 ρ | M-6 κ | χ_comp | ε̂,Δ̂ | severity |
|---|---|---|---|---|---|---|---|---|---|
| S1 symmetry alone | ✓ | ✓ | ✓ | ✓ | ✓ (exact) | ✓ (=0) | — | — | S0 |
| **S2 composed (flagship = G1)** | ✓ | ✓ | ✓ | ✓ | ✓ (exact) | ✓ (=0) | **✓** | — | S2 |
| S3 masked-region displacement | ✓ | ✓ | ✓ | ✓ | ✓ (measured) | ✓ (=0) | ✓ | ✓ | S2/S3 |
| S4 recipe convergence (T1) | ✓ | — | ✓ | — | — | — | — | **✓** | S2 (inverted) |

---

## 11. Evaluation design

### 11.1 Staged pathway, with the real budget attached

| Stage | Content | Cost on the project machine |
|---|---|---|
| **0 — substrate (gate G0, blocks all reporting)** | Build P+, P−, P~, P=, P⊥ under the three-source rule. Require ≥ 12 tier-3 positives, ≥ 8 P~ pairs over ≥ 3 families, \|N\| ≥ 200 with ≥ 10 per stratum. Verify determinism: identity transform reproduces published numbers exactly; `--no-cache` honoured; secondary-oracle parent cache regenerated here. | `compare` on published models only; ~5–300 s per pair; days of background runs |
| **1 — T1 soundness** | Alg 5 over the substrate → (ε̂, Δ̂) with BCa CIs, separability margin, per-signal AUC, W_struct/W_raw decomposition, benchmark-label audit. **This stage alone is a paper.** | no new artifacts; pure `compare` |
| **2 — gate G1** | S2 only, at 135M and 500M, O1, exact enumeration over X1×X2×M1t rungs. Resolve. | ~10² artifacts at ≤500M, each generated→evaluated→deleted |
| **3 — T2 frontier** | Conditional on G1. Alg 2 at ≤500M (exact); Alg 4 on every SUCCESS. | ~10³ EVAL calls at ≤500M; the bulk of the project's wall-clock |
| **4 — transfer check** | Alg 3: frozen ≤20 compositions at 1.4B. Explicitly underpowered; reported as a transfer check, not a search. | ~20 × 300 s per pair; overnight |
| **5 — cross-substrate** | Replicate the SUCCESS set against the secondary oracle (same device, same dtype). A transform evading weight-space but not behavioural testing is a *weaker* attack and the paper says so. | CPU generation; the binding cost — cap at ≤ 10 candidates |
| **6 — T3 + RESCORE** | Propagation analysis; Alg 7 on the Stage-0 substrate, including its distillation and permutation regressions. | re-scoring cached signal vectors; cheap |

### 11.2 Baselines, including the mandatory hardest one

| Baseline | Why |
|---|---|
| **B-0 identity** | no transform; anchors ΔR |
| **B-1 quantization round-trip (Q1)** | the standard benign transform; anything weaker is noise |
| **B-2 published benign derivative** | a real fine-tune we did not make; shows what "normal" looks like |
| **B-3 — THE HARDEST: train from scratch at matched κ** | the strongest simple alternative. **At κ = 0 this baseline is a randomly-initialised model with ρ ≈ 0.** So any κ = 0 transform with ρ ≈ 1 and E = 1 *strictly dominates* it. This is not us dodging the hard baseline — it is the hard baseline being **won by construction**, and it is precisely why a κ = 0 result falsifies C1: C1 asserts B-3 dominates, and at κ = 0 it demonstrably cannot. **This argument must be made explicitly in the paper**, because a reader who skims will otherwise assume we skipped it. |
| **B-4 benign P~ member** | the *destination* of displacement — the adversary's aspiration, not a competitor |

### 11.3 Metrics

Primary M-1 / M-4. Secondary M-2, M-3, M-5 (with regime), M-6, χ_comp, (ε̂, Δ̂), margin. Conditionals C1–C4
on every row. **Never a bare point estimate.**

### 11.4 Statistics

- **Exact arms need no testing.** The ≤500M lattice enumerates the declared family exhaustively; every
  point is measured. Report values and enumeration completeness, **not p-values** — a p-value on an
  exhaustive enumeration is a category error.
- **T1 is observational and needs proper inference.** BCa bootstrap (10 000, stratified by pair set) for
  ε̂, Δ̂, margin and all AUCs. Pair-level clustering respected.
- **Mixed-effects model for the T1 separability claim:**
  `σ_id ~ pair_class + (1 | parent_family) + (1 | recipe_family)` — **random intercepts for parent family
  and recipe family**, because pairs within a family are not independent and treating them as such is the
  single most likely way to overstate significance here.
- **Held-out split.** Rungs are declared in §9.1. **Development set = 135M; held-out = 500M.** The 1.4B
  transfer set (Alg 3) is frozen before it is run. No rung, threshold or ordering is tuned on held-out data.
- **Preregistered contrast set** (fixed now, 5 contrasts, Holm-corrected):
  1. S2 vs S1 — does composition beat free symmetry alone?
  2. S2 vs M1t-alone — does composition beat metadata alone? *(1 and 2 together are χ_comp)*
  3. κ = 0 frontier vs **B-3** — the C1 test (won by construction; reported for completeness)
  4. P~ vs P+ on s_struct — the masking existence claim
  5. P~ vs P+ on s_raw — the separability claim motivating RESCORE

### 11.5 Key ablations

| Ablation | Question |
|---|---|
| drop each composition component (Alg 4) | χ_comp — is the composition necessary? |
| gate forced to tier 3 *in the detector* | how much of MPK's accuracy is the gate carrying? |
| identity weights flattened (w_j = 1/5) | is the failure the weighting, or the signals? |
| WVC-only scoring | how good is the positional signal alone — RESCORE's ceiling |
| structural-only scoring (drop WVC) | direct measurement of recipe-convergence exposure |
| O1 → O2 → O3 | what do query access and DB access buy? |
| permutation fraction 0.25 / 0.5 / 1.0 | is the symmetry effect graded or all-or-nothing? |
| fp32 vs fp16 permutation | does exactness survive the precision publishers actually ship? |
| RESCORE vs MPK on the P+ distillation subset | the defence's **regression**, reported as loudly as its gain |

---

## 12. Detailed paper outline

| § | Thrust | Driving question | Primary figure/table |
|---|---|---|---|
| 1 Introduction | — | Can a shipping provenance tool tell "same recipe" from "same weights", and can a publisher exploit that for free? | Fig 1: same-recipe false positives — the pythia ladder, EAS vs WVC diverging with scale |
| 2 Background: lineage verification and what rests on it | — | What decisions depend on a derivation edge? | Fig 2: edge → ML-BOM → license / CVE / region |
| 3 Threat model | T2 | Who is the publisher-adversary, and what does κ = 0 mean? | Tab 1: observability lattice + capability-record schema |
| 4 **Recipe-convergence masking** | **T1** | **Does the detector measure descent, or recipe?** | **Fig 3: s_struct vs s_raw scatter, P+ / P− / P~ — the money figure** |
| 5 Benchmark validity audit | T1 | Can the vendor's own evaluation see this? | Tab 2: label vs `provenance` field; corrected confusion matrix |
| 6 Zero-cost laundering transforms | T2 | What can be done with no training at all? | Tab 3: 𝓛_T with exactness and observability |
| 7 **The κ = 0 frontier** | **T2** | **Does free laundering evade a deployed detector?** | **Fig 4: R and π per composition at κ = 0, with ρ verified exact** |
| 8 Composition necessity | T2 | Is the composed attack real or decorative? | Tab 4: χ_comp ablation (exact budget match) |
| 9 Consequence | T3 | What breaks downstream, and why doesn't signing fix it? | Tab 5: scenario → severity → control bypassed |
| 10 RESCORE and its limits | T3 | Can it be fixed, and what does the fix *not* cover? | Tab 6: guarantees / non-guarantees, incl. the permutation regression |
| 11 Discussion, disclosure, limitations | — | Scale, hardware, and what we did not test | — |

Written for the strongest ceiling. **The gate exit re-centers §7**; on Exit 3 it shrinks to a subsection
and §4 absorbs the space.

---

## 13. The arc and the gate

**Act I (T1)** measurement, no adversary — *the floor*. **Act II (T2)** hypothesis under test — *the
ceiling*. **Act III (T3)** consequence and scoped defence.

### Gate G1

**Instance:** Scenario S2 — X1 ∘ X2 ∘ M1t on Qwen2.5-0.5B→Instruct and SmolLM2-135M→Instruct, at O1,
exact enumeration over the declared rungs, **κ = 0, ρ = 1.00 by theorem and verified numerically.**

**Question:** does a **free, exactly capability-preserving** composed transform achieve
E = 1 (R > 5 ∧ π ≤ 75) with χ_comp = 1?

**Build nothing in T2 beyond this instance until G1 resolves.**

| Exit | Condition | Which paper gets written | Which section re-centers |
|---|---|---|---|
| **Exit 1 — C1 falsified** | E = 1, χ_comp = 1, at κ = 0, ρ exact | **"Free laundering defeats deployed provenance verification."** C1 falsified outright: evasion costs nothing and destroys no capability. Highest claim ceiling. | §7 leads; §4 becomes the mechanism explaining §7 |
| **Exit 2 — partial** | E = 1 only with the inexact rungs (Q1/N1/M2t) at κ = 0 but ρ < 1 | **"Laundering is free but not lossless — here is the exchange rate."** C1 falsified on *compute* while a real capability cost appears. The ρ axis becomes the contribution and S3 is the flagship. Still a strong paper, arguably the more honest one. | §7 re-centers on the ρ axis; S3 replaces S2 as flagship |
| **Exit 3 — held** | E = 0 for every κ = 0 composition with ρ ≥ 0.98 | **"Deployed provenance verification resists free laundering but confuses recipe with descent."** C1 survives *at zero compute* and we say so plainly. The paper becomes T1 + T3: a soundness and benchmark-validity study with a defence and an honest negative attack result, with §9.2's cited bound marking where evasion is known to begin. Moderate ceiling — but C-A, C-B, C-C and half of C-F all stand. | §4 leads and expands; §7 shrinks to "what free laundering cannot do" |

**Which way the current data leans.** Toward **Exit 3 for symmetries alone** and **Exit 1 or 2 for the
composition.** The signal semantics predict exact permutation moves only NLF + WVC (W = 0.29) while EAS,
END and LEP (W = 0.79) are invariant by construction — so free symmetry alone is unlikely to clear
π* = 75. But the single discordant tier-3 pair **[margin claim retracted; AUC 0.949]** suggests the distance to
travel may be small, so a
free composition plausibly reaches it. **This is a prediction recorded before the run, not a result.**

### Close

- **Floor.** Deployed lineage verification confuses training-recipe convergence with weight derivation; the
  weight-decided distributions do not separate; the vendor's own benchmark cannot see it. Holds with no
  adversary, on every exit, and is fully affordable under §0.
- **Ceiling.** A **zero-training-compute** composition displaces the true parent out of the auditor's view,
  falsifying Constitution C1 and breaking the ML-BOM edge that license, CVE and region-of-origin controls
  rest on.
- **Which paper gets written.** Exit 1 → the free-laundering attack paper. Exit 2 → the exchange-rate
  paper. Exit 3 → the soundness-and-defence paper with an honest negative attack result. In all three
  cases four of six contributions survive, and in all three cases there is a paper that can be written on
  one laptop.
