# Mock-review response experiments — 2026-08-24

Source: `manuscripts/feedback/SP2027_Recipe_or_Descent_Master_Review_and_Revision_Plan.md`
(a mock IEEE S&P review + 22-experiment revision plan). This file records what was run
this session, what it found, and what remains — triaged honestly against the actual scope,
which is a multi-week program, not a single session.

## P0 items — resolved

- **P0-B (compsoc template).** VERIFIED against the live CFP (`sp2027.ieee-security.org/cfpapers.html`,
  fetched 2026-08-24): `compsoc` is mandatory ("papers that fail to use the compsoc template ...
  are subject to rejection without review"). Applied: `\documentclass[conference,compsoc]{IEEEtran}`.
  **Also corrected a wrong assumption of my own**: the real limit is **13pp main text + up to 5pp
  references/appendices, 18pp total** — not 13pp total, which is what I had been enforcing across
  four prior rounds. The local `layout_audit.py` venue config (`IEEE-SP.yaml`) is not wrong
  (`page_limit_main: 13`), but the script checks total PDF pages rather than pages-before-References;
  that's a tooling gap, not a paper defect. Current PDF: 14 pages total, body ends within page 13,
  references run 1 page over — compliant with the verified real limit.
- **P0-C (deployed → publicly released).** Applied throughout (8 sites + title), since the paper
  demonstrates a publicly released tool, not evidenced operational reliance.
- **§32.2 (duplicated paragraph).** Fixed — a genuine duplicate "load-bearing" sentence in §X.
- **§33 (three contributions).** Applied — intro contribution list reduced from 5 items to
  C1 soundness / C2 mechanism / C3 design lesson, per the review's own proposed structure.
- **§34 (abstract simplification).** Applied — cut from 345 to 293 words, removed the 0.14%/0.006%
  precision pair per §29's demotion recommendation (kept one qualitative sentence instead).

## P0-A — CONFLICT WITH STANDING PI DIRECTION, NOT ACTED ON

The review's P0-A calls vendor disclosure "mandatory" before submission. This directly conflicts
with the PI's explicit, standing instruction (2026-08-23, recorded verbatim in RKA
`jrn_01M0RN2M43V9B05KJQKKPAR7JJ`): *"I do not want to send anything to anybody. This is for
research purposes."* Not actioned. Flagged, not silently complied with or silently dropped.

## Experiments run this session (pure recomputation, zero new downloads)

Built a consolidated 31-row lineage-verified corpus (`/tmp/corpus.json`, not yet promoted to a
tracked artifact) from the pooled M5 corpus (14 rows), Table I pythia pairs (5 rows), and the
Table IV tier-3 derivatives (12 rows) already in the paper's existing artifacts.

### E6 — gate counterfactual, 4 policies (was: "demonstrated on selected cases")

| policy | TP | FP | TN | FN | recall | specificity |
|---|---|---|---|---|---|---|
| vendor (as shipped) | 21 | 6 | 1 | 3 | 0.88 | 0.14 |
| weight-only (ignore gate) | 21 | 5 | 2 | 3 | 0.88 | 0.29 |
| conservative AND | 21 | 5 | 2 | 3 | 0.88 | 0.29 |
| conflict-aware (abstain on disagreement) | 21 | 5 | 1 | 3 | — | 1 abstention / 31 (3%) |

**Answer to the review's stated question ("does the gate amplify an unsound score, or is it the
primary source of error?"): mostly NOT the gate.** Removing it fixes exactly one of six false
positives (BiomedBERT, the tier-1/no-weight-evidence case already in §V-E). The five pythia
same-recipe pairs are false positives under **every** policy including weight-only, because the
identity score itself — not just the gate — is unsound on them. This sharpens rather than
contradicts the paper's existing claim in §V-A that "the weight evidence fails in the same
direction, which is why removing the gate would not repair the error"; it now has 31-row,
four-policy support instead of "selected cases."

### E8 — signal ablation (leave-one-out + concept groups)

| config | recall | specificity |
|---|---|---|
| vendor weights (all 5) | 0.84 | 1.00 |
| −EAS | 0.47 | 1.00 |
| −WVC | 0.84 | **0.00** |
| −END | 0.74 | 1.00 |
| −LEP | 0.68 | 1.00 |
| −NLF | 0.84 | 0.50 |
| WVC only | 0.05 | 1.00 |
| EAS only | 1.00 | **0.00** |
| structural only (EAS+END+LEP+NLF) | 0.84 | 0.00 |

n=21 rows carrying all five signals (Table I's pythia rows lack END/LEP/NLF in the recorded
artifact and are excluded from this ablation for fairness).

**This directly substantiates the paper's closing line** ("the signals hardest to launder are
the ones that were never measuring descent"): WVC is the only signal whose removal collapses
specificity to exactly 0, and WVC alone is the only single signal that achieves perfect
specificity — at the cost of 0.05 recall. EAS alone is the mirror image: perfect recall, zero
specificity.

### E10 — threshold sensitivity sweep (was: three fixed thresholds, no sweep)

Swept σ_id ∈ [0,1] in steps of 0.01 against 21 full-signal rows, tracking descent recall (TPR)
against same-recipe-independent false-positive rate simultaneously.

**No threshold in [0,1] simultaneously achieves TPR=1.0 and same-recipe FPR=0.0.** At the
threshold where same-recipe FPR first reaches 0 (t=0.80), descent recall has already fallen to
**0.42**. This is a stronger and more general statement than the vendor's fixed 0.65/0.75
thresholds happening to misfire on particular pairs — the score does not separate the classes at
*any* operating point, which is the review's "much stronger than reporting selected false
positives" bar, met.

### E11 — null-distribution robustness (partial)

Confirmed via `M2/results/null_frozen.json`: n=1,388, p75=0.5296 (matches the paper's 0.5298 to
rounding), 95% bootstrap CI **[0.5176, 0.5392]** (2,000 resamples) — the bar is not knife-edge.
Confirmed the strata composition tool 1 flagged: 1,238/1,388 (89.2%) in the 1–10B bucket against
a ≤1.4B evaluated population.

**Blocked, honestly:** the frozen artifact stores only aggregate per-stratum *counts*, not
per-score stratum *labels*, so a stratified p75 (≤1B only, or encoder-family only) cannot be
recomputed from what's on disk. Recomputing it requires re-running the DB scan with per-item
stratum tracking — a real task, not attempted this session. Recorded as genuinely open, not
papered over with a plausible-looking number.

## Not attempted this session — scale, and why

The remaining mandatory items (E1 latest-commit reproduction, E2 gold-corpus manifest, E3 AWM
baseline, E4 expanded second-verifier corpus, E5 same-recipe negative expansion, E9
recalibration, E12 training-progress ladder, E18–E22 defence expansion/adaptive
attacks/automatic layer correspondence/runtime scaling) each require either new model downloads,
implementing another paper's actual pipeline (AWM), or multi-hour compute campaigns. None were
attempted. This is triage, not completion — see the priority list below.

---

## Session 2 (2026-08-24, continued) — writing/structure items

Applied after the user's decision to work the full feedback list.

**Experiments added to the manuscript:**
- **E21 alignment ablation** — run from existing artifacts (`lap2.json`, `lap_defence.json`,
  `defence_signals.json`). New §X-B + Appendix A table. Findings:
  - Alignment buys *nothing* on clean pairs (raw positional has the widest clean margin, 0.822).
    What it buys is survival: raw positional collapses to ≈0.001 under laundering.
  - Column normalisation is what pins the score — assignment alone lands 0.359–0.881 depending
    on transform; with normalisation the laundered score returns to its clean value.
  - **The spectral (SVD) comparator is a negative control reproducing the paper's own thesis**:
    laundering-robust at 0.9955 while unable to separate classes at all (derived 1.0000,
    same-recipe 0.9924, unrelated 0.946–0.985 — margin 0.008). A comparator can be maximally
    robust and carry no descent information. This is §V's finding reached from the opposite
    direction, and it is the strongest corroboration in the paper.

**Writing items completed:**
- §16/E12-alternative — **compute claim reframed**. §VII retitled "How Far Must a Descendant
  Move?"; all four sites (abstract, intro, §VII, conclusion) now state crossings as fractions of
  *from-scratch embedding displacement*, with an explicit statement that the displacement→compute
  mapping is an assumption we cannot discharge. This removes the claim the review said "must be
  fixed one way or another."
- §32.1 — **all revision-history narration removed** (4 sites verified gone: earlier-floor
  assumption, "first reported", "correction to our own earlier report", retraction footnote).
- §32.2 — duplicate "load-bearing" paragraphs merged into one.
- §32.3 — **BiomedBERT and DistilBART separated into named Case 1 / Case 2.** This fixes a real
  perceived inconsistency: the 0.5248 figure (BiomedBERT) previously followed the DistilBART
  paragraph, reading as if it described DistilBART.
- §33 — contributions reduced 5 → 3 (C1 soundness / C2 mechanism / C3 design lesson).
- §34 — abstract simplified 345 → ~290 words, precision pair removed.
- §36 — **related-work taxonomy table added** (Table: system × access × signal × same-recipe
  negatives × adaptive attack), positioning against REEF, AWM, Nikolic et al., Stemma.
- §37 — **attacker-control dimension added** (M0–M3), with a table mapping every finding to
  observability × control × compute. Makes the cheapest attack legible: the two M1 rows
  (publisher-controlled metadata only) need no ML knowledge at all.
- P0-C — "deployed" → "publicly released" throughout, including the title.

**Structural:** four full listings moved to a proper Appendix (defence 18-pair, tier-3
derivatives, crash listing, benign-adaptation, BERT-family) — this uses S&P's 5-page
reference/appendix allowance rather than deleting evidence, which is what the earlier
(mistaken) 13-page-total assumption had forced.

**Known trivial defect:** body is 13 pages + ~3 lines. Needs a 3-line trim before submission.
I stopped chasing it after it consumed disproportionate effort; pagination reflow is
non-monotonic and it is a 5-minute fix at submission time, not a structural problem.

**Still not attempted** (unchanged): E1, E2 (full manifest), E3 AWM, E4, E5, E7, E9, E12
(checkpoint ladder), E13, E14, E15, E16, E17, E18, E19, E22, A1 clean-room. §35's four figures
remain unbuilt — the paper still has zero figures.
