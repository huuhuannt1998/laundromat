# Experiment run queue — LAUNDROMAT

Set up 2026-08-31 after the laptop crashed with five projects running concurrently.
**Nothing here runs on its own.** Start it with `./run_queue.sh --go`.

## Rule that caused the crash

Each 6.9B arm holds one fp16 model, about **14 GB**, on a **24 GB** machine. Two at once,
or one alongside other heavy projects, exhausts RAM. The queue is therefore strictly
sequential and gated by `preflight.py`, which refuses to start when free RAM is below
16 GB, free disk below 20 GB, or another python/uv process is holding more than 2 GB.

At the time of writing the preflight **BLOCKS**: only ~7 GB RAM is free. Close the other
projects before running.

## The queue

### 1. E27 — genuine ancestor at 6.9B (the only substantive gap)

**Why.** E23 established the same-recipe *false positive* at 6.9B (identity **0.8211**,
Confirmed Match). It has no *true positive* at that scale, because all three published
derivative labels we screened failed weight verification. A reviewer can therefore ask
whether the tool simply says Confirmed Match to everything large.

Pythia checkpoints answer it without trusting any published label: `step128000` is an
ancestor of the final model of the same run **by construction**. This is the instrument
§VII already uses at 160m (E12).

**What is measurable.** `config.json` is byte-identical across revisions, so the MFI gate
fires at tier 1 and pins the pipeline score to 1.0 whatever the weights say — E12 found
exactly that. The verdict is therefore uninformative here. The dependent variables are the
**identity score** and the **LAP defence**.

**The sharp question.** Does a genuine 6.9B ancestor score *above* the **0.8211** that the
same-recipe non-derivative earned? If it does not, the tool's weight evidence ranks a false
positive above a true positive at this scale — a strictly stronger result than E23 alone.
Either outcome is publishable; they say different things.

| | |
|---|---|
| state | **resumable — both checkpoints already materialised (26 GB on disk)** |
| new download | none |
| peak RAM | ~14 GB |
| wall time | ~35–45 min (LAP measured at 31 s/layer × 32 layers = 16.5 min, plus two extractions) |
| writes | `M9/e27_ancestor_scale.jsonl` (append-only, resumable per step) |
| frozen data | untouched |

Do **not** delete `M9/work_e27/` — that is what makes this resumable without a 14 GB
re-download.

## Optional, not gaps

- **E9 recalibration with 6.9b included.** The current arm (28 audited pairs) predates
  6.9B. Its conclusion — refitting the combination does not repair the failure — does not
  depend on that pair. Low value, moderate cost.
- **E24 / Stemma replication.** Blocked by *absent code*, not hardware: no public
  implementation of the method exists. Its stated requirement (cite and discuss) is met.

## Housekeeping done during setup

- `M9/reduced/` (25 GB E23 spill cache) deleted — regenerable, and E23's results are
  frozen and manifest-registered. Free disk went 79 GB → 110 GB.
- `check-paper` now resolves `pdftotext`/`pdfinfo` explicitly and **fails loudly** if they
  are missing. They dropped off `PATH` after the restart, which silently gutted the gate —
  the third time a gate degraded quietly in this project.
