#!/usr/bin/env python3
# OBSOLETE (2026-10-04): checks the removed [READY] (S&P 2027) section files; the TMLR folder's gate.py does number traceability and term locks for '[READY] (TMLR) LAUNDROMAT - P6'.
"""Assertion-level reconciliation for the LAUNDROMAT manuscript.

Why this exists
---------------
This project has lost the same class of correction three times:

  * 28d8196  Figure 1 still rendered "11 of 37" after the text moved to "8 of 32".
  * 2026-09-04  The intro still said "four" weak matches after body and tables
                moved to "three".
  * 2026-09-17  "All three rejections sit below the evasion bar" survived a
                re-audit that added a third rejection at 0.5375, which is ABOVE
                the 0.5298 bar.

All three are propagation failures, not authoring failures, and none was caught
by check-paper (LaTeX hygiene + page limits) or verify-data (MD5 over result
files). Those two gates check the manuscript against LaTeX and against the
artifacts. Nothing checked the manuscript against ITSELF.

Provenance comments cannot close this: two sentences asserting the same fact
bind to different claim IDs, so a dependency walk over claim IDs never connects
the places that must agree. This checks the assertion, not the dependency.

Usage:  check_assertions.py [bundle-dir]     (default: the [READY] bundle)
Exit:   0 all assertions hold, 1 otherwise.
"""
import re, sys, glob, os

BUNDLE = sys.argv[1] if len(sys.argv) > 1 else (
    "/Users/huanbui/Research/LAUNDROMAT-P6/[READY] (S&P 2027) LAUNDROMAT - P6")

# ---------------------------------------------------------------- registry
# Each entry: a load-bearing quantity, its value, and every section that must
# state it. Derived from the Step-1 claim portfolio. Adding a claim here is how
# you put it under guard.
SHARED_VALUES = [
    ("structural signal weight", "0.79",
     ["01-introduction", "04-verifier", "05-soundness", "06-laundering"]),
    ("positional signal weight", "0.21",
     ["01-introduction", "04-verifier", "06-laundering"]),
    ("evasion bar", "0.5298",
     ["03-threat-model", "06-laundering", "07-pricing", "A1-ablation"]),
    ("BiomedBERT identity score", "0.5248",
     ["01-introduction", "05-soundness", "A1-ablation"]),
    ("all-signal floor construction", "0.3255",
     ["06-laundering", "08b-consequence", "A1-ablation"]),
    ("from-scratch embedding anchor", "1.2044",
     ["05-soundness", "07-pricing"]),
    ("alignment margin, recovered", "0.3405",
     ["09-defence"]),
]

# The rejected derivatives, with the identity score each was rejected at.
# Check 2 recomputes how many fall at or below the evasion bar and holds the
# prose quantifier to that number. This is the C17 guard.
REJECTED = [("dynamic\\_tinybert", 0.3914),
            ("xtremedistil", 0.4211),
            ("twitter-roberta-base-sentiment-latest", 0.5375)]
EVASION_BAR = 0.5298

# Figures bake their text into a PDF, so no .tex check can see them. Each entry
# pairs a phrase printed in the figure with the LaTeX label it points at; the
# section number beside that phrase must match what LaTeX actually assigned.
# Figure 1 has shipped a stale number twice (28d8196, and again 2026-09-17),
# because section numbers are hardcoded in M6/make_fig0.py while LaTeX assigns
# them. Inserting or moving any section silently breaks them.
FIGURE_REFS = {
    "fig0_architecture.pdf": [
        ("same recipe",              "sec:soundness"),
        ("the gate overrides",       "sec:gate"),
        ("no zero-compute transform","sec:laundering"),
        ("one absent key",           "sec:denial"),
        ("in our pipeline",          "sec:consequence"),
        ("alignment separates",      "sec:defence"),
    ],
}
# Figure 2 prints its class sizes as "n=N". They must match the corpus the text
# describes, and the three corpus classes must sum to the audited-corpus size.
# Population counts are exactly what drifted on 2026-09-04.
FIGURE_COUNTS = {
    "fig2_distributions.pdf": {
        "classes": [("derived", 23), ("same recipe, independent", 6),
                    ("same family, independent", 2)],
        "sums_to": 31,            # the audited corpus of section 5.1
        "standalone": [("unrelated (frozen null)", 1388)],
    },
}

# Terms retired by the Step-1 term lock. None may survive in a figure.
# Bare "rejection" is included deliberately. The old fig3 labelled the 0.65 line
# "0.65 rejection", which matches none of the longer retired phrases. No figure
# in this paper has a legitimate use for the word, so the bare term is safe to
# flag here even though it would be too broad for prose.
RETIRED_IN_FIGURES = ["rejection", "weak-match threshold", "first-tier gate"]

QUANTIFIER = {1: ["one of the", "the one"], 2: ["two of the three", "both"],
              3: ["all three", "three of the three"]}

def load(bundle):
    out = {}
    # glob.escape: the bundle directory is literally named "[READY] ...", and
    # an unescaped "[READY]" is a glob character class matching one of R,E,A,D,Y.
    # Without this the scan silently finds zero files and every check "fails".
    pat = os.path.join(glob.escape(bundle), "sections", "*.tex")
    for p in sorted(glob.glob(pat)):
        name = os.path.basename(p)[:-4]
        out[name] = re.sub(r"(?m)^%.*$", "", open(p).read())
    return out

def figure_text(pdf):
    import subprocess
    for exe in ("/Library/TeX/texbin/pdftotext", "pdftotext"):
        try:
            r = subprocess.run([exe, pdf, "-"], capture_output=True, text=True)
            if r.returncode == 0:
                return re.sub(r"\s+", " ", r.stdout)
        except FileNotFoundError:
            continue
    return None

def find_aux():
    for c in ("/private/tmp/claude-501/-Users-huanbui-Research-LAUNDROMAT-P6/"
              "6a04ab68-179e-4b0b-b41e-0717f424d4f5/scratchpad/gatebuild/main.aux",):
        if os.path.exists(c):
            return c
    return None

def check_figures(bundle):
    fails = []
    aux = find_aux()
    nums = {}
    if aux:
        for m in re.finditer(r"\\newlabel\{(sec:[^}]+)\}\{\{([0-9.]+)\}", open(aux).read()):
            nums[m.group(1)] = m.group(2)
    for fig, refs in FIGURE_REFS.items():
        path = os.path.join(bundle, "figures", fig)
        if not os.path.exists(path):
            print(f"   ..    {fig} not present"); continue
        txt = figure_text(path)
        if txt is None:
            print(f"   ..    {fig} (pdftotext unavailable)"); continue
        for term in RETIRED_IN_FIGURES:
            if term.lower() in txt.lower():
                fails.append(f"{fig} contains retired term \"{term}\"")
                print(f"   FAIL  {fig}: retired term \"{term}\"")
        if not nums:
            print(f"   ..    {fig}: no main.aux found, section numbers unchecked")
            continue
        for phrase, label in refs:
            m = re.search(r"§\s*([0-9.]+)\s*" + re.escape(phrase), txt)
            if not m:
                fails.append(f"{fig}: no section reference printed beside \"{phrase}\"")
                print(f"   FAIL  {fig}: no § beside \"{phrase}\"")
            elif label not in nums:
                print(f"   ..    {fig}: {label} not in main.aux")
            elif m.group(1) != nums[label]:
                fails.append(f"{fig}: \"{phrase}\" cites §{m.group(1)} but "
                             f"{label} is §{nums[label]}")
                print(f"   FAIL  {fig}: \"{phrase}\" cites §{m.group(1)}, "
                      f"{label} is §{nums[label]}")
            else:
                print(f"   ok    {fig}: §{m.group(1)} {phrase}")
    # figure population counts
    for fig, spec in FIGURE_COUNTS.items():
        path = os.path.join(bundle, "figures", fig)
        if not os.path.exists(path):
            print(f"   ..    {fig} not present"); continue
        txt = figure_text(path)
        if txt is None: continue
        total = 0
        for label, n in spec["classes"] + spec["standalone"]:
            if re.search(r"n\s*=\s*%d\b" % n, txt):
                print(f"   ok    {fig}: n={n} ({label})")
            else:
                fails.append(f"{fig}: expected n={n} for {label}, not found")
                print(f"   FAIL  {fig}: expected n={n} ({label}), not found")
        total = sum(n for _, n in spec["classes"])
        if total != spec["sums_to"]:
            fails.append(f"{fig}: class sizes sum to {total}, corpus is {spec['sums_to']}")
            print(f"   FAIL  {fig}: classes sum to {total}, corpus is {spec['sums_to']}")
        else:
            print(f"   ok    {fig}: classes sum to {total} = audited corpus")

    # other figures: retired-term sweep only
    for path in sorted(glob.glob(os.path.join(glob.escape(bundle), "figures", "*.pdf"))):
        fig = os.path.basename(path)
        if fig in FIGURE_REFS: continue
        txt = figure_text(path)
        if txt is None: continue
        hit = [t for t in RETIRED_IN_FIGURES if t.lower() in txt.lower()]
        for t in hit:
            fails.append(f"{fig} contains retired term \"{t}\"")
        print(f"   {'FAIL' if hit else 'ok  '}  {fig}: "
              f"{'retired term ' + hit[0] if hit else 'no retired terms'}")
    return fails

def main():
    if not os.path.isdir(os.path.join(BUNDLE, "sections")):
        print(f"FAIL  no sections/ under {BUNDLE}"); return 1
    S = load(BUNDLE)
    fails = []

    print("1. shared values stated consistently across sections")
    for label, value, required in SHARED_VALUES:
        missing = [f for f in required if f not in S or value not in S[f]]
        if missing:
            fails.append(f"{label} ({value}) missing from: {', '.join(missing)}")
            print(f"   FAIL  {label:<34} {value:<8} missing from {', '.join(missing)}")
        else:
            print(f"   ok    {label:<34} {value:<8} in {len(required)} sections")

    print("\n2. rejected-derivative scores appear where claimed")
    for name, score in REJECTED:
        hits = [f for f, t in S.items() if f"{score:.4f}" in t]
        if not hits:
            fails.append(f"rejected derivative {name} at {score} appears nowhere")
            print(f"   FAIL  {name:<42} {score} appears in no section")
        else:
            print(f"   ok    {name:<42} {score} in {', '.join(sorted(hits))}")

    print("\n3. evasion-bar quantifier matches the scores (the C17 guard)")
    below = [s for _, s in REJECTED if s <= EVASION_BAR]
    n = len(below)
    print(f"   {len(REJECTED)} rejected derivatives; {n} at or below the "
          f"{EVASION_BAR} bar ({', '.join(str(s) for s in below)})")
    allowed = QUANTIFIER.get(n, [])
    pat = re.compile(r"([A-Za-z][^.;]{0,120}?below the evasion bar)", re.I)
    checked = 0
    for f, t in S.items():
        flat = re.sub(r"\s+", " ", t)
        for m in pat.finditer(flat):
            frag = m.group(1).strip(); checked += 1
            low = frag.lower()
            bad = [q for q in QUANTIFIER.get(len(REJECTED), []) if q in low]
            good = [q for q in allowed if q in low]
            if bad and not good:
                fails.append(f"{f}: quantifier claims all {len(REJECTED)} are "
                             f"below the bar, but only {n} are -- \"{frag}\"")
                print(f"   FAIL  {f}: \"{frag[:88]}\"")
            else:
                print(f"   ok    {f}: \"{frag[:88]}\"")
    if not checked:
        print("   ..    no 'below the evasion bar' sentence found")

    print("\n4. figure text against the sections it cites")
    fails += check_figures(BUNDLE)

    print("\n" + ("PASS  all assertions hold" if not fails
                  else f"FAIL  {len(fails)} assertion(s) violated"))
    for f in fails: print(f"      - {f}")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
