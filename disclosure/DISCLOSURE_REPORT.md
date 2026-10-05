# Coordinated disclosure report — Model Provenance Kit

**Product.** Model Provenance Kit, `github.com/cisco-ai-defense/model-provenance-kit`, Apache-2.0.
**Version tested.** 1.1.0, commit `a75007d5ca4aaf8df2e3f055f318557a965b93aa`
("Bump version to 1.1.0 (#17)", 2026-08-12). Upstream has since added exactly one commit,
`87fe4b7c7ce0e0e94fc47159ed889d345a4c6fbf` ("Bump the uv group across 1 directory with 2
updates (#18)", 2026-09-18), whose parent is the pinned commit and which modifies only `uv.lock`.
No source file has changed since the version tested, so both issues below reproduce on the
current tip. Verified 2026-09-17.
**Reporter.** Authors of an academic evaluation of publicly released model-lineage verification,
under submission to IEEE S&P 2027 (anonymous during review).
**Status.** Prepared, not yet transmitted. Pending author authorisation.
**Intended channel.** GitHub private vulnerability reporting on the repository, per the project's
own `SECURITY.md`; fallback `oss-security@cisco.com`.
**Report date (to be filled on send).** ______
**Requested embargo.** None required. Both issues are observable in published source and in the
behaviour of the released CLI on public checkpoints. We ask only for an acknowledgement we can
cite, and we will record the response, or its absence, in the paper.

---

## Summary

Two defects in the released tool. Neither requires an adversary, credentials, or a private model.
Both were found by running the shipped CLI against ordinary public Hugging Face checkpoints.

| # | Class | Effect | Severity we assign |
|---|---|---|---|
| 1 | Input validation / missing schema default | Availability: the CLI aborts and returns **no verdict** when a model's `config.json` omits `architectures`. Reachable by ordinary correct publishing. | Moderate |
| 2 | Gate semantics | Integrity: a **tier-1 `Confirmed Match`** is returned with `identity_score` and all five weight signals `null`. The top verdict is issued with no weight evidence computed. | Moderate–High |

We do not claim either is exploited in the wild, and we surveyed no deployment.

---

## Issue 1 — Absent `architectures` key aborts verification (denial of verification)

### Behaviour

`provenancekit compare <parent> <child> --json --no-cache` exits non-zero with
`Error: Failed to extract base features for '<child>'` and emits no JSON when the child's
`config.json` has no `architectures` key. The comparison never runs, so the consumer receives
neither a positive nor a negative verdict.

### Root cause

`src/provenancekit/core/signals/metadata.py:126` builds the MFI fingerprint with

```python
"architectures": getattr(config, "architectures", ["unknown"]),
```

The fallback `["unknown"]` is unreachable. `transformers.PretrainedConfig` declares
`architectures: list[str] | None = None` (`configuration_utils.py:224`), so the attribute always
exists on a loaded config and is `None` when the JSON key is absent. `getattr` therefore returns
`None`, never the default. The MFI model at `src/provenancekit/models/signals.py:38` declares
`architectures: list[str]`, which rejects `None`, and the failure surfaces as a feature-extraction
abort rather than as a validated field error.

The degradation path already exists and is simply not reached: an **empty** `architectures` list
degrades cleanly to tier 2. An absent key and an explicit `null` both abort. Null handling is a
second, separable defect from the missing default.

### Affected public models we observed

Eight public models return no verdict against their stated parent as published. Seven come from
four established organisations and one is a widely used community release:

| Model | Parent compared against | `config.json` sha256 prefix |
|---|---|---|
| `microsoft/MiniLM-L12-H384-uncased` | `google-bert/bert-base-uncased` | `235fcbb1b55045e0` |
| `huawei-noah/TinyBERT_General_4L_312D` | `google-bert/bert-base-uncased` | `9435ceadf314b412` |
| `google/bert_uncased_L-4_H-256_A-4` | `google-bert/bert-base-uncased` | `61716a972c73e9cc` |
| `emilyalsentzer/Bio_ClinicalBERT` | `google-bert/bert-base-uncased` | `4a470c65801cab55` |
| `allenai/cs_roberta_base` | `FacebookAI/roberta-base` | `f91fab60e48f6f77` |
| `allenai/biomed_roberta_base` | `FacebookAI/roberta-base` | `f91fab60e48f6f77` |
| `allenai/news_roberta_base` | `FacebookAI/roberta-base` | `f91fab60e48f6f77` |
| `allenai/reviews_roberta_base` | `FacebookAI/roberta-base` | `f91fab60e48f6f77` |

The four AllenAI rungs ship a byte-identical `config.json`, so the eight models represent five
distinct publisher-side omissions. We count models because the verifier is invoked per model.

### Prevalence in the population that matters

The shipped fingerprint catalog has no missing gate-critical field across the 157 of 184 assets we
could retrieve. That measures curation: an asset is catalogued because it was fingerprintable.
Sampling instead what a consumer submits — the forty most-downloaded public models for each of
fourteen supported families — 47 of 545 retrievable configurations lack the key. Setting aside 23
GGUF repositories the tool does not read and seven test fixtures, 17 ordinary models remain, 3.1%.
Eleven of the seventeen are DeBERTa, nine of them first-party Microsoft releases (`deberta-base`
and `-large`, both v2 sizes, all four v3 sizes and `mdeberta-v3-base`), so an entire model line
from a major publisher cannot be verified as published: run end to end, a database scan of each of
the nine returns no verdict.

A second trigger is reachable without omitting anything: renaming `model_type` to a string the
installed `transformers` does not register also yields no verdict under the default setting, which
runs no model-hosted code, so a new architecture's custom value would do the same.

### Reproduction

```bash
git clone https://github.com/cisco-ai-defense/model-provenance-kit
cd model-provenance-kit
git checkout a75007d5ca4aaf8df2e3f055f318557a965b93aa
uv sync

uv run provenancekit compare \
  FacebookAI/roberta-base allenai/cs_roberta_base --json --no-cache
# exit 1, no JSON on stdout,
# stderr: Error: Failed to extract base features for 'allenai/cs_roberta_base'

# Control: the same model with one key added, weights untouched, verifies normally.
python - <<'PY'
import json, pathlib
p = pathlib.Path("cs_roberta_local/config.json")
c = json.loads(p.read_text()); c["architectures"] = ["RobertaForMaskedLM"]
p.write_text(json.dumps(c))
PY
uv run provenancekit compare FacebookAI/roberta-base ./cs_roberta_local --json --no-cache
# returns a verdict
```

### Suggested fix

Coerce at the boundary rather than relying on an unreachable `getattr` default:

```python
"architectures": getattr(config, "architectures", None) or ["unknown"],
```

and make the MFI field `list[str] | None` with tier assignment degrading to tier 2 when it is
absent, which is the path the empty-list case already takes. Report a validated field error rather
than a feature-extraction abort so the operator can see which field is missing.

---

## Issue 2 — Tier-1 `Confirmed Match` with all weight signals `null`

### Behaviour

Comparing `facebook/bart-large-cnn` against `sshleifer/distilbart-cnn-12-6` returns:

```json
{"scores": {"mfi_score": 1.0, "mfi_tier": 1, "mfi_match": "exact",
            "identity_score": null, "tokenizer_score": 1.0,
            "pipeline_score": 1.0, "provenance_decision": "Confirmed Match"},
 "signals": {"eas": null, "nlf": null, "lep": null,
             "end": null, "wvc": null, "tfv": 1.0, "voa": 1.0}}
```

The highest verdict in the ladder is issued with the identity score and every one of the five
weight signals returned as `null`. Nothing was computed for the gate to override.

### Why we report it as a defect and not a design choice

The tier-1 rule is documented: a matching architecture hash pins `pipeline_score` to 1.0. The
defect is that the pin is applied when the weight comparison produced **no value at all**, so the
output cannot be distinguished from one where weight evidence was computed and agreed. A consumer
reading `provenance_decision` alone, which is what a policy gate does, cannot tell the two apart.

The same mechanism also overrides weight evidence that *was* computed and disagreed. Comparing
`google-bert/bert-base-uncased` against
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` — a model whose own card states it was
"pretrained from scratch using abstracts from PubMed" — returns `Confirmed Match` at tier 1 on a
configuration used exactly as published. Its identity score, had the pipeline consulted it, is
0.5248, below the tool's own 0.65 weak-match line; its positional signal is 0.0015 and its
embedding displacement 1.4527, larger than the 1.2044 we measure between independently trained
pairs. The correct answer was available and was not used.

This bears directly on the project's own Model Provenance Constitution, which states that
fabricating a derivation link is "trivial at the metadata surface; defeated by weight-level
analysis, which cannot be faked without actual weight transfer." On these two published pairs
weight-level analysis did not defeat the metadata, because it was not consulted.

### Reproduction

```bash
uv run provenancekit compare \
  facebook/bart-large-cnn sshleifer/distilbart-cnn-12-6 --json --no-cache
# pipeline_score 1.0, provenance_decision "Confirmed Match",
# identity_score null, eas/nlf/lep/end/wvc all null

uv run provenancekit compare \
  google-bert/bert-base-uncased \
  microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract --json --no-cache
# tier 1, pipeline_score 1.0, "Confirmed Match", identity_score 0.5248
```

### Suggested fix

Two changes, either of which removes the ambiguity:

1. Do not emit a `Confirmed Match` when `identity_score is None`. Return an explicit
   `Insufficient data` outcome, which the result model already defines, so a policy gate can
   distinguish "agreed" from "not computed".
2. When weight signals are computed and contradict the tier — identity score below the 0.65
   weak-match line at tier 1 — surface the disagreement in the verdict rather than discarding it. The
   conflict-aware behaviour is what a defence-in-depth consumer needs, and the tier pin currently
   removes the ability to implement it downstream.

### Adversarial reachability, for completeness

`architectures` and `model_type` are publisher-written fields with no attestation. We measured
what control of those fields alone buys, holding weights byte-identical:

- **Forging descent.** `nlpaueb/legal-bert-base-uncased` against `google-bert/bert-base-uncased` is
  tier 3: `Not Matched` at 0.6061 as the hub serves its `pytorch_model.bin`, and `Weak Match` at
  0.7391 from the safetensors copy we edited (see the note on file format below). Rewriting one field, `architectures`, to the
  target's value makes every tier-1 hash field match and lifts the result to tier 1,
  `Confirmed Match`, pipeline 1.0. The identity score is unchanged at 0.7391 and no weight was
  touched.
- **Denying descent.** `cardiffnlp/twitter-roberta-base` against `FacebookAI/roberta-base` is
  tier 1, `Confirmed Match`, pipeline 1.0 as published. Renaming `model_type` from `roberta` to
  `bert`, a registered but different type, drops it to tier 3 and to `Weak Match`, pipeline 0.6997 (0.6899 read from the hub's `.bin`).
  The same edit applied to `textattack/bert-base-uncased-SST-2` moves it from tier 2
  `Confirmed Match` to tier 3 `High-Confidence Match`, which is still a match, so denial through
  this field is not reliable in that direction.

We report this as reachability, not as a separate vulnerability: it is the same gate semantics as
issue 2, exercised deliberately.

**Note on file format.** The identity score depends on how identical weights are serialised. The
norm-layer and positional signals concatenate tensors in the order the weight file lists them; a
`pytorch_model.bin` lists each weight before its bias and layers in module order, while a
safetensors file sorts tensors by name. Re-saving legal-bert's own `.bin` as safetensors, every
tensor kept, moves its identity score against `bert-base-uncased` from 0.6061 to 0.7391, and plain
fine-tunes of `bert-base-uncased` from 0.82 to 1.00. Matching tensors by name rather than by
position would remove the dependence.

---

## Mitigation we evaluated

Recovering row correspondence before comparison separates the classes on our corpus where the
shipped signals do not. We solve a linear assignment between MLP input-projection rows after
column L2 normalisation, then compare under that assignment. On thirty audited pairs the
derivative and same-recipe-independent classes order without error, and the score is unchanged by
the zero-compute laundering transforms we built. The technique is published prior work, not ours;
what we add is a design rule: align only on matrices constrained by an intervening elementwise
nonlinearity, never on matrices related by a continuous bilinear invariance, because the latter
gives an adversary a free parameter and we break it at zero training compute.

Two limits we state plainly. The signal is not evaluated against an adversary who knows it and has
training compute to spend. On width-mismatched children it requires a fitted projection between
the two hidden spaces, and under that projection the derivative and non-derivative scores in our
corpus are not separated.

## What we published

An academic paper describing the above, plus a reproduction artifact containing the frozen result
files and the scripts that produced them. No laundering tool is packaged or released. Everything
in the paper derives from the public Apache-2.0 repository at the pinned commit, the published
fingerprint database, and public checkpoints, on one laptop, with no access requested and no
hosted or third-party system touched.
