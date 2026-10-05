# M5 corpus-distance sweep — methodological caveats

## The config repair is not neutral, and that is itself a finding

Two AllenAI DAPT models ship without an `architectures` key, which crashes MPK. To run the
experiment at all we materialise a local copy and write the key in. **But the value we write
determines the MFI tier**, and therefore the verdict:

- write `["RobertaForMaskedLM"]` (matching `roberta-base`) → `arch_hash` matches → **tier 1** →
  `pipeline_score` pinned at 1.0 → **Confirmed Match**, regardless of what the weights say
- write anything else → hashes miss → **tier 3** → the weight signals decide

All three DAPT rows came back tier 1 for exactly this reason. So their *verdicts* are
gate-decided artifacts of our repair, not evidence about the weight signals.

This is the M3 circularity result demonstrated constructively rather than observationally: by
choosing one string in a file we did not train, we choose the verdict. It also means the repair
should be read as "restoring verifiability", not "reconstructing ground truth".

**Consequence for the analysis:** the dependent variable is `identity_score` (σ_id), *not*
`pipeline_score` or the verdict, precisely because σ_id is computed from the weights and is
independent of the tier the gate assigns. Any statement in this sweep about verdicts on repaired
models must carry this caveat.

## Lineage depth is uncontrolled for the twitter arm

`cardiffnlp/twitter-roberta-base-sentiment-latest` is a **grandchild** of `roberta-base`
(roberta-base → twitter-roberta-base-2021 → sentiment fine-tune), so a comparison against
`roberta-base` spans two training stages. `cardiffnlp/twitter-roberta-base` is included to
isolate the first stage. Do not attribute the difference to corpus distance without that
comparison.

## Displacement is a proxy for training expenditure, not a measure of it

`‖θ_c − θ_p‖_F / ‖θ_p‖_F` measures how far the weights moved, not how many FLOPs moved them.
It is used here because it is directly comparable across models and needs no vendor disclosure.
The empirical "trained from scratch" anchor (independent Pythia runs, mean **0.9533**) is the
right yardstick for C1 for the same reason — but a model could in principle move far with little
compute, or little with much.

## Hub lineage metadata is itself unreliable — which is the argument, twice over

Establishing ground truth for these sweeps required reading model cards one by one, and the
metadata was frequently missing:

| defect | instances |
|---|---|
| `architectures` absent or `null` (crashes MPK) | 6 — Microsoft, Google (first-party), Huawei, AllenAI ×2 |
| model card empty (21 bytes) | 3 — `allenai/{cs,news,reviews}_roberta_base` |
| `_name_or_path: null` (no parent recorded) | every model checked |
| no card at all | `textattack/bert-base-uncased-SST-2` |

Three of our own seven BERT-family labels were wrong on the first pass because we inferred lineage
from repo names. Each is now recorded in `lineage.json` with its evidence, and inferred labels are
marked as inferred.

This cuts both ways and both ways matter:

1. **It is the case *for* weight-based verification.** Declared lineage is often absent, and where
   present it is unverifiable self-report. Something that reads the weights is genuinely needed.
2. **It is the case *against* the metadata gate.** MPK's MFI tier is computed from exactly these
   fields. Tier 1 pins `pipeline_score` to 1.0 and the weight signals are never consulted — which is
   how `BiomedBERT`, pretrained from scratch and scoring 0.5248 on its own weight signals, is
   returned as **Confirmed Match**. The gate treats the least reliable input as decisive.

## Two BERT-family rows were not collected

`textattack/bert-base-uncased-imdb` and `csarron/bert-base-uncased-squad-v1` are absent from
`sweep_bert.jsonl`. Their downloads stalled repeatedly on half-open connections (a residue of a
mid-session connectivity drop): the transfer would reach ~290 MB of a ~440 MB file and then move zero
bytes, with the client blocked indefinitely rather than timing out. After a bounded direct fetch and
two relaunches, collection was abandoned rather than spend further wall-clock on it.

**What this costs:** both are pure fine-tunes of `bert-base-uncased`, the same class as
`textattack/bert-base-uncased-SST-2`, which *was* collected (displacement 0.0153, σ_id 0.8249,
Confirmed Match — correct). They would have added two more low-displacement confirmations, not a new
condition. No reported conclusion depends on them.

**What it does not cost:** every decisive BERT row was collected — the from-scratch false positive
(BiomedBERT), the from-scratch true negative (legal-bert), a correct continued-pretrain child
(finbert), a correct fine-tune (SST-2), and the cross-lineage false positive (Bio_ClinicalBERT).

The BERT arm is therefore n = 5, not n = 7, and is reported as such.
