# Corrections log

Every correction made during LAUNDROMAT, with what triggered it. Recorded so none is discoverable
only by a reviewer.

| # | claim as first stated | status | trigger |
|---|---|---|---|
| 1 | "EAS separates derived from unrelated by ~zero" | **scoped** — true only of the same-recipe arm; AUC on the full M1 corpus is **1.000** | re-analysis of collected data |
| 2 | "Untied-embedding models admit an exact per-row embedding rescaling" | **wrong** — residual stream carries the unnormalised embedding; verified on pythia + bloom | testing my own claim |
| 3 | MFI tier column in the M2 gate runs | **contaminated** — my harness leaked tokenizer truncation state into `tokenizer.json`, changing `family_hash`. MPK is deterministic (8/8) | byte-comparing artifacts |
| 4 | "EAS uniformly high in ALL families" | **narrowed** — BLOOM is a clean negative at 0.7990; effect is architecture-dependent (4 of 5) | collecting the 5th family |
| 5 | LAP defence "architecture-specific" (Qwen 0.4012) | **my bug** — subsampled rows *before* alignment, destroying correspondence. True value 0.9969 | inspecting an anomalous number |
| 6 | M1 hypothesis: "claim *absence* dominates hub divergence" | **refuted** — 80–94% of likely derivatives declare; my 6-model sample was the old tail of an adoption curve | M3 survey at scale |
| 7 | Margin bootstrap CIs | **degenerate estimator** — min−max margin can only rise under resampling; lower bound pinned by construction | noticing the CI lower bound equalled the point estimate |
| 8 | **B1: "weight-decided distributions overlap; no separating threshold exists"** | **RETRACTED** — one discordant pair of 39; AUC 0.949, exact permutation p = 0.0071 | running the predeclared statistics |
| 9 | "M3 verification is a runaway; killing it" (told the PI mid-run) | **wrong call** — the job was healthy and completed: 266 scanned, 121 correctly rejected as oversize, 12 verified. Second time I mistook a quiet job for a dead one | the job finishing normally |
| 10 | "Isotropic noise / int4 **destroys** the model" (ρ = 0.376 / 0.232, perplexity) | **invalid instrument** — the predeclared M-5 battery gives ρ = 0.7863 and 0.8632. Perplexity overstated damage by up to 0.63. Exact symmetries confirmed at ρ = **1.0000** | running M-5 after deferring it four times |
| 11 | Preregistered contrasts C4 / C5 reported as testable as constructed | **two defects + a power floor** — P~ contained an identity pair (WVC 1.0); both arms confounded by shape (WVC ≈ 0 whenever dims differ, derived or not); and on the valid stratum n=3v3 puts the minimum attainable exact p at **0.10**, so C5 could never reject | running the preregistered contrast set |
| 12 | Design §11.4: family clustering is "the single most likely way to overstate significance" | **backwards here** — the mixed-effects fit made the contrast ~100× *stronger* (p 0.0028 → 0.00003), because ICC = 0.26 and pooling dumped between-family variance into the residual | fitting the model the design demanded |

| 13 | "The moving variable is **corpus distance**, not the fact of continued pretraining" | **refuted** — inferred from two points; the AllenAI DAPT ladder spans 2.4× displacement across four domains and moves σ_id by 0.02. The predictor is **embedding** displacement (R² 0.730 vs 0.441), and even that is insufficient — the norm *distribution* (END) is what decides | running the controlled ladder |

| 14 | "operational evasion arrives at **70%** of a from-scratch run" | **axis mismatch** — the crossing is fit on *embedding* displacement but was divided by the *overall* from-scratch anchor (0.9533) instead of the embedding one (1.2044). Same-axis it is **56%**, i.e. cheaper than reported; the error flattered C1 | pooling the second family and rechecking units |

| 15 | "completing the BERT arm **strengthens** the embedding-displacement predictor, R² 0.780 → 0.8366" | **population mismatch** — 0.8366 is the correlation over all 7 BERT rows *including non-derivatives*; the paper's quantity is the pooled regression over *true derivatives only* across both families, which **falls to 0.6927** (n=9→11). Completing the arm weakens the fit, it does not strengthen it. The direction claim survives (embedding 0.693 > overall 0.560) and the crossing moves only 56.1% → 54.7% | recomputing the paper's own statistic instead of a convenient nearby one |

| 16 | **"destroying the two most heavily weighted signals perfectly clears the evasion bar" (floor 0.4299)** | **wrong — the bound assumed every signal can reach 0.** Measured floors: anchor 0.4987, layer-energy 0.4836, norm-layer 0.6861 never approach zero. Substituting them, that pair bottoms out at **0.6102, well above the 0.5298 bar**. Evasion needs at least three signals; enumerating all 32 subsets, no pair suffices. The bar is still reachable (all-signal floor 0.3255, 38.6% below it) so the conclusion holds, but the attack is **harder** than we reported | measuring the floors instead of assuming them |
| 17 | "1,170 anchor-signal observations, none below 0.5013, corpus minimum equals the floor" | **superseded within the hour** — a second seed floors the signal at **0.4987**, below the figure I had just called a minimum. The mechanism predicts 0.5 and the observations straddle it; asserting either measured value as a hard bound was the error | running the same construction at a different seed |

| 18 | "`ratify_manuscript_claim` returns HTTP 500 for any `claim_version > 1`. Confirmed version-gated." | **wrong diagnosis** — CB5 ratified at **version 2** today without error. The real guard (`manuscript_native.py:402`) refuses when any ratification row for the claim joins an **active** decision, regardless of version; CB5 passed because its prior ratification was bound to a decision since superseded. The 500 is a separate defect: a `ValueError` precondition failure mapped to 500 instead of 4xx, which hid the message that would have explained it | attempting the operation instead of re-asserting the earlier conclusion |
| 19 | ratifying CB5 at v2 | **my error, same session** — I bound wording whose `allowed_wording` says "an O0 version is conjectured, not demonstrated" and whose `prohibited_wording` bars "recoverable from observed scores alone". Both were falsified by our own O0 recovery hours earlier. I checked currency for CB7 and CB8 and ratified those correctly or not at all; for CB5 I ratified first and checked after | checking the other two claims against the evidence, then belatedly checking the one already bound |

| 20 | **"Three of the five signals cannot be driven anywhere near zero; the irreducible mass is 0.32 and the minimum attainable identity score is 0.3255"** | **two of those three floors were artifacts of the perturbation I chose.** Cross-checking against real published models in our own corpus: the layer-energy signal reaches **0.0599** (I claimed 0.5386) and the norm-layer signal reaches **exactly 0.0000** (I claimed 0.6836) — the latter because I randomised norm vectors uniformly on [0,2], and an all-positive vector necessarily correlates positively with an all-positive reference, while real children reach a clamped negative cosine. Only the embedding anchor survives the cross-check (0.4987 vs a real-model minimum of 0.5360, with a mechanism that predicts 0.5). Irreducible mass 0.3204 → **0.1795**; least evidenced identity score → **0.1893**. The headline survives untouched: **no pair of signals suffices** and evasion still needs ≥3, because the anchor's 0.1795 is what forbids it either way | a reviewer flagged an unrelated anomaly (the norm-layer signal reading 0.0 across a whole family), and following it exposed the floor claim |

| 21 | the "real models" column added to *fix* #20 | **repeated #20's exact error.** Three of its five cells were wrong, and the anchor cell (0.5360) was not a real model at all — it was a rung of our own X5anchor noise ladder, i.e. a construction, printed under a heading that said "real published models". Corrected against 1,100+ genuine published-model comparisons: anchor 0.5570, embedding-norm 0.0002, layer-energy 0.0395. Least evidenced identity score 0.1893 → **0.1859**. No headline moves | a reviewer walked our own released artifacts and found it in ten minutes |

Correction #21 is the one I find hardest to explain. #20 says in terms: *a floor measured from one
family of transforms bounds that family, not the signal.* The table I wrote to record that lesson
put a transform output in the column reserved for real models. Writing a correction down does not
prevent committing it; only recomputing does.

Correction #20 is the most instructive in this table. A floor measured from one family of
transforms bounds *that family*, not the signal — and I presented it as the latter. The check
that catches it is free and I did not run it: compare the constructed extreme against the
extremes the corpus already contains. The real models had been sitting in the artifacts the
whole time, reading 0.0 on a signal I had just called irreducible at 0.68.

Corrections 18 and 19 are the same lesson from opposite directions. #18: a blocker recorded as
"confirmed" was never re-tested, and one attempt dissolved it. #19: an operation was performed
before its precondition was checked. Both were cheap to avoid and neither was caught by a gate —
`manuscript_readiness` reports *whether* a claim is ratified, never whether the ratified wording is
still true.

Corrections 16 and 17 arrived together and point the same way as 11: a quantity that had been
*assumed* (signals reach zero; a single seed fixes a floor) was finally *measured*, and the
measurement disagreed. #16 is the more consequential — it is the only correction in this table
that makes our own attack look worse rather than better, and it was found by testing a bound we
had already published rather than by a reviewer.

Correction #15 is the same error as #14 one level up: both times I computed a number on one
population or axis and compared it to a published number computed on another, and both times the
mistake flattered the result. The check that catches it is mechanical — recompute the *paper's*
statistic, on the *paper's* population, not a nearby quantity that is easier to obtain.

Corrections 7, 8, 10, 11 and 12 all surfaced only after a "did we run all experiments?" audit forced a
predeclared instrument to actually be run — the statistics, the M-5 capability battery, and the
preregistered contrast set. **Every single time a deferred predeclared instrument was finally executed,
it falsified something already written down.** Eleven of the fourteen came from continuing to collect or test
after a conclusion had been recorded. #13 is the cleanest example: a two-point hypothesis, recorded
confidently, refuted by the first controlled experiment that could test it.

The generalisable lesson for the paper's methods section: the failure mode was not bad measurement, it was
*deferring the measurement that had been declared in advance* and substituting a cheaper proxy (perplexity
for a task battery, a pooled t-test for the clustered model, an unstratified contrast for a stratified one).
Each substitution looked locally reasonable and each one was wrong.
