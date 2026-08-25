# LAUNDROMAT M1 — Scoop Gate + Attack-Target Acquisition

Mission `mis_01M0E1X2EJ737X29J4A6E0VWBE` · project `prj_01M0E1SKSP6CSEHY9DTT4V9MB2` · 2026-08-19
**No laundering transforms were applied in this mission.** Everything below is the pre-attack reference.

## Verdict

**GO on the adversarial axis, with a MANDATORY NARROW.** Blocking checkpoint `chk_01M0E5H3NYK5BTRAKPN529QTBX`.

## Job B — the oracle

Cisco Model Provenance Kit, `cisco-ai-defense/model-provenance-kit`, commit `a75007d`, v1.1.0 (2026-08-12), Apache-2.0.
Python 3.13 · torch 2.13.0 · transformers 5.5.0 · Apple M4 / 24 GB / CPU only · `HF_HUB_DISABLE_XET=1`.
Deep signals: `deep-signals.zip` sha256 `f37f689bc1d87544ddfe81072680e48582ac0f6429c9a1942138faf79c0cc872`,
948.3 MB → 1477.6 MB, 184 parquet, 39 families.

### Thresholds (documented AND confirmed in source — not derived empirically)

`core/scoring.py`: `_SCORE_HIGH_CONFIDENCE = 0.75`, `_SCORE_WEAK_MATCH = 0.65`
`config/constants.py`: `SIMILARITY_THRESHOLD = 0.75` — *"Calibrated on 111-pair benchmark (best F1 via threshold sweep)"*

| pipeline score | verdict |
|---|---|
| `S == 1.0` or MFI tier ≤ 2 | Confirmed Match |
| `S > 0.75` | High-Confidence Match |
| `0.65 < S ≤ 0.75` | Weak Match |
| `S ≤ 0.65` | Not Matched |

`identity_score` = NaN-aware weighted mean — EAS 0.36 (d=2.03), WVC 0.21 (d=1.21), END 0.19 (d=1.05), LEP 0.16 (d=0.92), NLF 0.08 (d=0.46).
`tokenizer_score` = 0.25·TFV + 0.75·VOA — **reported only, never enters the decision.**
MFI gate — tier 1 (`arch_hash` equal) → `pipeline := 1.0`; tier 2 (`family_hash` + same dims) → `0.9`; tier 3 → `pipeline := identity_score`.

### Compare-mode baseline (21 labelled pairs)

MFI-gated (tier 1) — verdict from `config.json` alone:

| pair | pipe | identity | correct? |
|---|---|---|---|
| gpt2 \| gpt2 | 1.0000 | 1.0000 | ✓ identity control |
| SmolLM2-135M \| -Instruct | 1.0000 | 0.9992 | ✓ |
| Qwen2.5-0.5B \| -Instruct | 1.0000 | 0.9994 | ✓ |
| bloom-560m \| bloomz-560m | 1.0000 | 0.7801 | ✓ verdict, marginal on weights |
| pythia-160m \| pythia-160m-deduped | 1.0000 | **0.6745** | **✗ FALSE Confirmed Match** |

Weight-decided (tier 3):

- positives — bert→distilbert **0.9220**, gpt2→distilgpt2 **0.8910**, roberta→distilroberta **0.7300**
- negatives — gpt2/bloom **0.7752**, pythia-160m/410m 0.7378, SmolLM2-135M/360M 0.7153, bert/albert 0.6787,
  gpt2/gpt-neo-125m 0.6784, pythia-410m/gpt2 0.6534, pythia-160m/gpt2 0.6378, Qwen/SmolLM2 0.6051,
  bloom/Qwen 0.5426, distilbert/distilroberta 0.5175, roberta/gpt2 0.4998, SmolLM2/pythia 0.4215, bert/roberta 0.3481

**One discordant pair out of 39 comparisons.** roberta→distilroberta (0.7300, a genuine distillation)
scores below gpt2 vs bloom-560m (0.7752, unrelated), so **no perfect threshold exists on this corpus**.
Separation is otherwise strong: **AUC 0.949, exact permutation p = 0.0071 (n = 3/13).**
*(Corrected — the earlier "distributions overlap / no separating threshold exists" was too strong; see
`analysis/separation_test.py`.)*

### Scan-mode baseline

`scan --top-k 5 --threshold 0.30`, 14–18 s each.

- **distilgpt2**: rank 1 = **Qwen2-1.5B 0.8715**, rank 2 = true parent gpt2 0.8689. The true parent is out-ranked with no adversary present.
- Qwen2.5-0.5B-Instruct → gpt2 0.8203 / gpt2-medium 0.7718 / gpt2-xl 0.7655, all High-Confidence.
- bloomz-560m → gpt2-medium 0.8765 / gpt2 0.8115 / gpt2-xl 0.7573, all High-Confidence.
- pythia-160m-deduped and gpt-neo-125m: correctly nothing above High-Confidence.

Ambient cross-family noise floor in the tier-3 regime ≈ **0.87**.


### Cisco's exact benchmark pair (added after first write-up)

`pythia-1.4b` vs `pythia-1.4b-deduped` — 303 s cold, 5.6 GB:

| EAS | NLF | LEP | END | WVC | tier | identity | pipeline |
|---|---|---|---|---|---|---|---|
| 0.9838 | 0.9735 | 0.9811 | 0.9849 | **0.1023** | 1 | **0.7976** | 1.0 "Confirmed Match" |

**Corrects B2.** At 160M the weight signals were right (0.6745, below threshold) and only the gate failed.
At 1.4B — the pair actually in Cisco's benchmark — `identity_score` is **0.7976, above threshold**, so the
weight signals fail too. Removing the MFI gate would not fix this pair.

- **B6** — four of five identity signals score ~0.98 on two *independently trained* models sharing
  architecture, hyperparameters and init seed. Only **WVC** (0.1023), the sole signal that compares raw
  parameter values, is correct — and it carries 0.21 weight against 0.79 of structural signals.
  **MPK's identity score largely measures training-recipe convergence, not weight derivation.**
  Attack consequence: the 0.79 structural mass is what must move, and it is exactly what capability
  depends on. Exact symmetries touch at most NLF+WVC = 0.29 (see B3).
  Caveat: n=2 (one pair, two scales); needs a controlled same-recipe/independent arm before publication.

- **B5** — Cisco's shipped `benchmarks/Benchmark_All.json` puts this pair in the **`similar`** list (the
  positives) though its own `provenance` field reads `"none"` and its `challenge` text says
  *"independently trained"*; Constitution §8 lists the same pair under **Independent**.
  `run_benchmark.ipynb` labels purely by list membership — it carries the `provenance` value into a column
  named `constitution` but never uses it. So the tier-1 metadata false positive scores as a **success**,
  and the reported "96% accuracy, 4 errors all false negatives" cannot surface this failure mode.
  Two other `provenance: "none"` entries in `similar` are *sibling derivatives* (shared ancestor) and are
  defensible; this one has no shared ancestor.

## Findings

- **B1 (CORRECTED)** — one discordant pair (roberta→distilroberta 0.7300 < gpt2/bloom 0.7752), so no *perfect* threshold exists; separation otherwise strong at AUC 0.949, p = 0.0071. The original "distributions overlap" claim is **retracted**.
- **B2** — the `config.json` MFI tier-1 gate **overrides** correct weight evidence (pythia-160m vs -deduped: weights say 0.6745, gate says 1.0 "Confirmed"). Cisco's Constitution §8 lists that pair type as *Independent*; §9 asserts *"Metadata-only systems are trivially defeated."*
- **B3** — *(analytical, untested)* under exact hidden-dim permutation, EAS/END/LEP should be invariant (0.71 of identity weight) and only NLF+WVC should degrade (0.29) — and nothing fires at all while tier 1 short-circuits.
- **B4** — large cross-family false-positive surface in scan mode; evasion must be measured as **parent rank**, not absolute score.

## Job A — scoop classification

Axis: **(a)** attacks an owner-planted mark · **(b)** attacks third-party derivation detection with no planted mark.

| work | cat | adversarial? | perm/scale? | tradeoff curve? | compute norm? | deployed tool? |
|---|---|---|---|---|---|---|
| **Cisco MPK + Constitution + Explorer** | **b — THE TARGET** | no (111-pair benign-hard) | not in threat model | no | no | *is* the tool |
| Nasery+ 2509.26598 | a (7/10 planted; all behavioural) | **yes**, 10 schemes | no — output filtering only | utility yes | no | no |
| REEF (ICLR'25) | b defence | shows perm breaks weight baselines | **yes** | no | no | no |
| GhostSpec 2511.06390 | b defence | yes (spectral-divergence FT) | names as background | partial | no | no |
| AWM (ICLR'26) | b defence | no — benign only | LAP absorbs perm | no | no | no |
| Spectral Signatures (KDD'26) | b defence | robustness only | **yes** | no | no | no |
| Stemma 2607.25880 | b defence | *"we do not consider adaptive attackers"* | no | no | no | no |
| MPS 2602.00772 · FNF 2601.22692 · Shang+ 2601.11683 | b | partial | no | partial | no | no |
| Fernandez+ (ICASSP'24) | a — watermark *by* invariants | n/a | **constructive recipe** | no | ~zero cost | no |
| Smudged Fingerprints (SaTML'26) | b — **image** modality | yes (removal + forgery) | n/a | — | — | no |

### Surviving delta
D1 deployed tool · D2 evasion↔capability curve · D3 compute normalised to pretraining · D4 attestation-stack consequence · D5 metadata-gate composite (no academic fingerprint has a gate).

### What died
RQ1's premise. "Exact symmetries defeat weight-space fingerprinting" is prior art (Fernandez'24 constructs, REEF'24 demonstrates, GhostSpec'25 assumes, KDD'26 re-tests). The surviving symmetry claim is about **deployment**, not mechanism.

### The strongest asset found
Constitution §9 stakes falsifiable claims with no supporting evaluation:
- **C1** *"Fully evading all weight-level signals is as expensive as training from scratch; cheaper techniques leave residue."*
- **C2** *"Residual traces of the true parent persist and are recoverable by weight-level forensics."*
- **C3** *"Robust provenance detection must rely on weight-level signals. Metadata-only systems are trivially defeated."*

C1 *is* an evasion-cost-versus-capability claim. Exact weight-space symmetries are absent from §9's representative techniques. B2 already puts C3 in tension with the shipped pipeline.

## Artifacts
```
M1/oracle/model-provenance-kit/     MPK @ a75007d + deep signals
M1/oracle/model_provenance_testing/ arXiv 2502.00706 tester + .venv-mpt
M1/oracle/{prefetch,run_pairs,run_pairs2,run_scan}.py
M1/results/{pairs,pairs2,scans}.jsonl · analyze.py
```
