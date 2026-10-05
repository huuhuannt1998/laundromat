# E23 — the same-recipe false positive at 6.9B

Answers the standing review objection that every empirical arm stopped at 1.4B.

## Why this was previously recorded N/A, and why that was wrong

The checklist said E23 "requires hardware beyond the one 24 GB laptop." That was an
inherited assumption. Two measurements retire it:

* MPK's `scanner.compare` extracts model A to a small feature record, extracts B, then
  scores the two records; weights are never co-resident, and `model_loader.py` pins
  `torch_dtype=float16`. Peak is ONE model.
* What actually blocked it was our own `M4/p_tilde_arm.py`, whose `defence()` holds both
  models at once in fp32 — 51.5 GB at 6.9b. A harness limit, not a hardware limit.

Rewritten to reduce one model at a time and spill to disk, the 6.9B pair ran in
**6.6 minutes at 10.9 GB peak**. The rewrite reproduces the frozen 70m value exactly
(0.197389 vs 0.197389; max per-layer delta 1.5e-8), so E23 is directly comparable to the
existing ladder.

## Result: the false positive survives a 4.9x scale increase

| pair (independent runs of one recipe) | identity | pipe | tier | verdict | LAP |
|---|---|---|---|---|---|
| pythia-70m / -deduped   | 0.7856 | 1.000 | 1 | Confirmed Match | 0.1974 |
| pythia-160m / -deduped  | 0.6745 | 1.000 | 1 | Confirmed Match | 0.1961 |
| pythia-410m / -deduped  | 0.7810 | 1.000 | 1 | Confirmed Match | 0.2151 |
| pythia-1b / -deduped    | 0.7262 | 1.000 | 1 | Confirmed Match | 0.1714 |
| **pythia-6.9b / -deduped** | **0.8211** | **1.000** | **1** | **Confirmed Match** | **0.3250** |

Control — different scales, so no derivation is possible:
`pythia-2.8b` vs `pythia-6.9b` -> tier 3, pipeline 0.6910, **Weak Match**, LAP n/a
(shape mismatch, the defence abstains). The tool is not indiscriminate: it reserves
Confirmed Match for the same-shape same-recipe class.

At 6.9B the identity score is the HIGHEST anywhere on the ladder (0.8211). The
underlying weights agree more too: cosine between the independent runs rises from
0.13-0.20 at 70m-1.4b to **0.3215** at 6.9b. Scale makes recipe convergence worse, not
better, so the confusion this paper documents does not thin out in the regime that
matters commercially. The LAP defence still separates (0.325 against ~1.0 for a true
derivative) but its margin narrows over the same span.

## Three of four published labels failed weight verification

Nothing here trusts a model card. `verify_label.py` compares the claimed parent and
child tensor by tensor; it is calibrated at 0.9999 on a true fine-tune
(`pythia-410m-deduped` -> `mnoukhov/pythia410m-sft-tldr`), 0.1995 on that same child
against the WRONG parent, and 1.0000 on identity.

| claimed | measured | status |
|---|---|---|
| `pythia-2.8b-deduped` = independent run of `pythia-2.8b` | **cos 0.99596** over all 388 tensors (attn 0.996, mlp 0.994, norm 1.000, embed 0.990); no tensor byte-identical | EXCLUDED — cannot be two independent runs |
| `lambda/pythia-2.8b-deduped-synthetic-instruct` fine-tuned from `pythia-2.8b-deduped` | cos 0.191 | EXCLUDED — not weight-derived |
| `pszemraj/pythia-6.9b-HC3` `base_model: pythia-6.9b-deduped` | cos 0.339 to `-deduped`, 0.325 to `pythia-6.9b` | EXCLUDED — derived from neither |
| `pythia-6.9b` / `-deduped` independent | cos 0.3215 | **VALID — used** |

The 2.8b case matters beyond this experiment: at every other scale the pair sits at
0.13-0.20, so 2.8b is the lone outlier, and any study that takes the Pythia deduped
suite as a source of independent-run negatives without checking will silently include a
near-duplicate. This is the same failure class as B5 (the labelling defect in the
vendor's shipped benchmark) and the BERT-Tiny defect in our own corpus.

Had the 2.8b pair been used unchecked it would have read as a spectacular result —
Confirmed Match at identity 0.9967 and LAP 0.9887 — which is in fact the CORRECT verdict
on two near-copies, and would have been reported as a false positive.

## Reproduce

    python3 M9/fetch.py all           # 39 GB, one weight format per repo
    python3 M9/verify_label.py <parent> <child>
    python3 M9/e23_scale.py           # verified pairs only
