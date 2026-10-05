# paper-tools

## check_assertions.py — assertion-level reconciliation

### The gap this fills

The project has two gates and they check two different things:

| gate | checks |
|---|---|
| `check-paper` | the manuscript against **LaTeX** — refs, cites, errors, overfull, page limits |
| `verify-data` | result files against **MD5** — that frozen artifacts have not moved |

Neither checks the manuscript against **itself**. That is where every defect this
project has actually shipped has lived:

| when | defect | caught by |
|---|---|---|
| `28d8196` | Figure 1 still rendered `11 of 37` after the text moved to `8 of 32` | a hand-written grep, after the fact |
| 2026-09-04 | intro still said `four` weak matches after body and tables said `three` | manual audit |
| 2026-09-17 | `All three rejections sit below the evasion bar` — the third is `0.5375`, **above** the `0.5298` bar | manual audit |

All three are propagation failures: a correction reached some sites and not
others. None is a LaTeX error and none is an artifact mismatch, so neither
existing gate could see them.

### Why provenance comments cannot close it

The manuscript carries 237 `% provenance:` tags binding prose to claim IDs. They
are a *record* of dependency, not an *enforcement* of it — nothing reads them.
And even a perfect dependency walk over claim IDs would not have caught these:
two sentences asserting the same fact routinely bind to **different** claim IDs,
so there is no edge between the places that must agree. The 2026-09-04 case is
exactly this — the intro's count bound to `clm_…BXP5A3`, the body's identical
count to `clm_…TSWFFY0`.

Dependency-level propagation is necessary but not sufficient. This tool does
assertion-level reconciliation instead: it checks the *claim*, not the *link*.

### What it checks

1. **Shared values.** Each registered quantity must appear in every section that
   is supposed to state it. Catches a number corrected in one place and not
   another.
2. **Named scores.** Each rejected derivative's identity score must still appear
   where claimed, so the registry cannot drift away from the paper.
3. **Figure text.** Figures bake their text into a PDF, so no `.tex` check can
   see them. Each registered phrase printed in a figure is matched to the LaTeX
   label it points at, and the section number beside it is compared against what
   LaTeX actually assigned (read from `main.aux`). Also sweeps every figure for
   terminology the term lock retired.

   Figure 1 has shipped a stale number **twice** — `28d8196`, and again
   2026-09-17 with three wrong cross-references (`§5.2`→`§5.5`, `§8.3`→`§9`,
   `§9`→`§10`). Section numbers are hardcoded in `M6/make_fig0.py` while LaTeX
   assigns them, so inserting or moving any section silently breaks them and
   nothing notices.

4. **The C17 guard.** Recomputes how many rejected derivatives fall at or below
   the evasion bar, then holds every "… below the evasion bar" sentence to a
   quantifier consistent with that count. This is the check that would have
   caught the 2026-09-17 defect at both of its sites.

### Usage

    python3 paper-tools/check_assertions.py                 # the [READY] bundle
    python3 paper-tools/check_assertions.py <bundle-dir>     # any snapshot

Exit 0 if all assertions hold, 1 otherwise. It is wired into the scratchpad
`gate.sh`, so it runs with every build check.

### Validated against a known-bad input

Not merely written — tested. Run against the pre-fix snapshot
`01-before-C17_20260917-2034`, it fails with exactly the two real sites:

    FAIL  05-soundness: "All three rejected derivatives fall below the evasion bar"
    FAIL  A1-ablation:  "All three rejections sit below the evasion bar"

and against the pre-fix figures (kept as `fig0_before.pdf`, `fig3_before.pdf`):

    FAIL  fig0_architecture.pdf: "the gate overrides" cites §5.2, sec:gate is §5.5
    FAIL  fig0_architecture.pdf: "in our pipeline" cites §8.3, sec:consequence is §9
    FAIL  fig0_architecture.pdf: "alignment separates" cites §9, sec:defence is §10
    FAIL  fig3_displacement.pdf: retired term rejection

Run against the current bundle, it passes. Keep that snapshot; it is the
regression test.

### Extending it

Add a row to `SHARED_VALUES` for any claim worth guarding — that is how a claim
in the Step-1 portfolio becomes enforceable. Keep the registry small: a value
belongs here when it is stated in more than one place, because that is the only
situation in which the sites can disagree.

### One implementation note

The bundle directory is literally named `[READY] (S&P 2027) LAUNDROMAT - P6`,
and `[READY]` is a valid glob character class matching one of `R,E,A,D,Y`. An
unescaped glob therefore finds zero files and reports every check as failing.
`glob.escape` is load-bearing here, not decoration.
