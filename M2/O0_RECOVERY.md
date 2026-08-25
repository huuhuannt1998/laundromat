# O0 weight recovery — recovering the verifier's private weights from verdicts alone

**Observability level: O0** (verdicts and scores only; no source, no fingerprint database).
Runner: `M2/o0_weight_recovery.py` (WVC arm), `M2/o0_eas_recovery.py` (EAS arm).
Artifacts: `M2/results/o0_weight_recovery.jsonl`, `M2/results/o0_eas_recovery.jsonl`.

## Construction

Perturb a model so that **exactly one** of the five identity signals moves and the other four
stay pinned at 1.0000. Then the weight of the moved signal is the ratio of deltas:

    w_s  =  (sigma_id(0) - sigma_id(p))  /  (s(0) - s(p))

Isolation is achieved by norm-preserving randomisation of a chosen row set:

| arm | rows randomised | why the others stay pinned |
|---|---|---|
| WVC | all 2-D projection matrices, **excluding** embedding / lm_head / norm vectors | preserving each row's L2 norm leaves END and LEP invariant; excluding the embedding leaves EAS and END invariant; excluding norm layers leaves NLF invariant |
| EAS | embedding rows **only**, norms preserved | everything outside the embedding is untouched, so WVC / LEP / NLF are invariant; row-norm preservation keeps END invariant |

Both isolations were verified empirically, not assumed: in each arm the four
non-target signals read exactly 1.0000 at every p.

## Result

| target | recovered | published | error | s.d. over 5 values of p |
|---|---|---|---|---|
| `w_EAS` | **0.3600** | 0.36 | **0.006%** | 0.0001 |
| `w_WVC` | **0.2100** | 0.21 | **0.002%** | 0.0001 |

Together these account for **0.57 of the unit identity weight**, recovered with no access to
source code, documentation, or the fingerprint database — only the score the verifier returns.

## The estimator needs no linearity

The WVC arm was first analysed by regressing sigma_id on p (OLS: 0.9999 - 0.2093p, R^2 = 0.999998).
That works only because WVC happens to be near-linear in p. **EAS is not**: at p = 0.125 it reads
0.9347 where linearity predicts 0.8750, and at p = 1.0 it reads 0.5013 where linearity predicts 0.0.

The ratio estimator is nevertheless exact at every single p (s.d. 0.0001), because it never
references p at all — only the two observed signal values. The attacker therefore does not need to
know, or control, how the signal responds to the perturbation. They need only isolate it.
This makes the recovery far more general than a regression would suggest, and it is why the
WVC estimate improves from 0.14% error (OLS) to 0.002% (ratio) on the same data.

## Second finding: EAS has half the range it appears to have

Driving the embedding to fully random directions floors EAS at **0.5013**, not 0. The value is
what an affinely-rescaled cosine, (1 + cos)/2, returns when cos -> 0 for random high-dimensional
rows. So the usable range of the *highest-weighted* signal is [0.501, 1.000], width 0.499.

Consequences, using the published weights and thresholds:

- EAS contributes an unconditional floor of 0.36 x 0.5013 = **0.1805** to sigma_id for **any**
  pair of same-dimension models, related or not.
- That is **27.8%** of the 0.65 weak-match threshold and **24.1%** of the 0.75 high-confidence
  threshold, granted before any evidence of derivation is considered.
- Predicted sigma_id at p = 1 with the other four pinned: 0.36(0.5013) + 0.21 + 0.19 + 0.16 + 0.08
  = 0.8205. Observed: 0.8205. Additivity residual 3e-5 — the same linear model confirmed in M2.

### Falsification test

If the floor is real, no EAS measurement anywhere in the project should fall below it.
Scanning every result artifact: **1,179 EAS observations, none below 0.4987.** A later run at a different seed floors it at 0.4987, so the mechanism-predicted 0.5 is straddled rather than exactly attained — the value produced by fully randomising the embedding. The next
smallest is 0.5360 (an unrelated-model anchor attack). The floor survives the test.

## Scope

Single reference model (SmolLM2-135M) for the isolation construction; the additivity that
licenses the ratio estimator was independently confirmed on three architectures in M2
(residuals <= 1e-4). The EAS floor claim is supported corpus-wide (1,179 observations) but the
0.5013 value itself is measured on one tokenizer/vocabulary; the *mechanism* (rescaled cosine)
predicts ~0.5 for any high-dimensional embedding, and no counterexample appears in the corpus.
