"""E25 -- what the verifier's errors actually do to a consumer.

Section VIII-B says "we observed no downstream harm and surveyed no consumer's
configuration". This measures it instead, on a pipeline built from the tools a consumer
would really use: a CycloneDX 1.6 ML-BOM validated against the official pinned schema,
and an Open Policy Agent gate enforcing the two obligations the paper's own framing names
--- licence propagation and inherited advisory scope.

For every pair we hold a ground-truth relationship for, we run the gate twice: once on
the BOM the VERIFIER's verdict produces, and once on the BOM GROUND TRUTH would produce.
Where the two disagree, the verifier has caused a wrong admission or a wrong denial. That
disagreement is the measurement.

Both consumer wirings are evaluated, because a reviewer will say a careful pipeline fails
closed and the question is whether that rescues it.
"""
import json, subprocess, itertools, collections
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
import sys; sys.path.insert(0, str(_ROOT/"M8"))
from emit_bom import emit, validate

POLICY = _ROOT/"M8"/"policy"/"supplychain.rego"
LIC = json.loads((_ROOT/"M8"/"model_licences.json").read_text())
MATCH = {"Confirmed Match","High-Confidence Match","Weak Match"}

def gate(bom):
    validate(bom)
    p = subprocess.run(["opa","eval","-I","-d",str(POLICY),"data.supplychain"],
                       input=json.dumps(bom), capture_output=True, text=True)
    r = json.loads(p.stdout)["result"][0]["expressions"][0]["value"]
    return bool(r.get("admit", False)), r.get("deny", [])

def licence_of(m, default="apache-2.0"):
    return LIC.get(m) or default

# ---- corpus: ground truth + measured verdict --------------------------------------
# The obligation we instrument is INHERITED ADVISORY SCOPE, not licence propagation.
# Our corpus contains zero licence-violating derivative pairs -- every derivative carries
# a licence compatible with its parent -- so a licence policy cannot discriminate on it,
# and running one would have measured nothing. Advisory scope needs no fabricated licence:
# it needs one realistic consumer action, publishing an advisory against an ancestor, and
# then asks whether true descendants inherit the exposure. Both error directions become
# visible, and every relationship used is a real one.
PYTHIA = {"pythia-70m/-deduped": ("EleutherAI/pythia-70m","EleutherAI/pythia-70m-deduped"),
          "pythia-160m/-deduped": ("EleutherAI/pythia-160m","EleutherAI/pythia-160m-deduped"),
          "pythia-410m/-deduped": ("EleutherAI/pythia-410m","EleutherAI/pythia-410m-deduped"),
          "pythia-1b/-deduped": ("EleutherAI/pythia-1b","EleutherAI/pythia-1b-deduped"),
          "pythia-1.4b/-deduped": ("EleutherAI/pythia-1.4b","EleutherAI/pythia-1.4b-deduped")}
# Verdicts measured elsewhere but absent from the manifest. A manifest row with
# measured=False means WE did not record the verdict there, NOT that the verifier
# returned none -- conflating those two would fabricate the very failure this experiment
# measures, so unknown verdicts are skipped explicitly rather than defaulted.
MEASURED_ELSEWHERE = {}
for f in ("M4/p_tilde_arm.jsonl","M4/power_fix.jsonl","M1/results/pairs.jsonl"):
    q=_ROOT/f
    if not q.exists(): continue
    for line in q.read_text().splitlines():
        if not line.strip(): continue
        d=json.loads(line)
        a=d.get("a") or d.get("model_a"); b=d.get("b") or d.get("model_b")
        sc=(d.get("scores") or (d.get("result") or {}).get("scores") or {})
        if a and b and sc.get("provenance_decision"):
            MEASURED_ELSEWHERE[(a,b)] = sc["provenance_decision"]

rows=[]; skipped_unknown=0
man = json.loads((_ROOT/"M6"/"gold_lineage_manifest.json").read_text())
for r in man["rows"]:
    if r["analysis_set"]=="excluded": continue
    par, ch = r["parent"], r["child"]
    if par == "pythia":                      # suite-level row, resolve to real repo ids
        if ch not in PYTHIA: continue
        par, ch = PYTHIA[ch]
    v = r.get("verdict") or MEASURED_ELSEWHERE.get((par, ch))
    if v is None:
        skipped_unknown += 1        # verdict unknown to us; not the same as "no verdict"
        continue
    rows.append(dict(parent=par, child=ch,
                     truly_derived=r["relationship_class"].startswith("D-"),
                     verdict=v, no_verdict=False))
# models the verifier cannot answer on at all
for l in (_ROOT/"M1"/"results"/"mpk_crashes.jsonl").read_text().splitlines():
    if not l.strip(): continue
    d=json.loads(l)
    if d.get("defect"):
        rows.append(dict(parent=d.get("parent"), child=d.get("child"),
                         truly_derived=True, verdict=None, no_verdict=True))

print(f"pairs evaluated: {len(rows)}  ({sum(1 for r in rows if r['no_verdict'])} of them no-verdict); "
      f"{skipped_unknown} skipped for want of a recorded verdict\n")
print(f"{'wiring':<12}{'wrong admits':>14}{'wrong denials':>15}{'correct':>9}")
print("-"*52)
summary={}
for wiring in ("fail-open","fail-closed"):
    wrong_admit=[]; wrong_deny=[]; ok=0
    for r in rows:
        if not r["parent"] or not r["child"]: continue
        cl, pl = licence_of(r["child"]), licence_of(r["parent"])
        # One realistic consumer action: an advisory is published against the ancestor.
        # The question the pipeline must answer is whether this model inherits it.
        adv = ["ADV-2026-0001 poisoned-checkpoint disclosure against the ancestor"]
        got = emit(r["child"], r["parent"], r["verdict"] if not r["no_verdict"] else None,
                   child_licence=cl, parent_licence=pl, on_no_verdict=wiring,
                   parent_advisories=adv)
        truth = emit(r["child"], r["parent"],
                     "Confirmed Match" if r["truly_derived"] else "Not Matched",
                     child_licence=cl, parent_licence=pl, on_no_verdict=wiring,
                     parent_advisories=adv)
        a_got,_ = gate(got); a_true,d_true = gate(truth)
        if a_got == a_true: ok += 1
        elif a_got and not a_true: wrong_admit.append((r["child"], d_true))
        else: wrong_deny.append(r["child"])
    summary[wiring]=dict(wrong_admit=wrong_admit, wrong_deny=wrong_deny, ok=ok)
    print(f"{wiring:<12}{len(wrong_admit):>14}{len(wrong_deny):>15}{ok:>9}")

for w,s in summary.items():
    if s["wrong_admit"]:
        print(f"\n{w} -- admitted despite a real obligation:")
        for c,d in s["wrong_admit"][:6]:
            print(f"   {c.split('/')[-1]}")
            for m in d[:1]: print(f"      missed: {m[:95]}")
    if s["wrong_deny"]:
        print(f"\n{w} -- denied though ground truth admits ({len(s['wrong_deny'])}):")
        for c in s["wrong_deny"][:6]: print(f"   {c.split('/')[-1]}")
json.dump({k:{"wrong_admit":[c for c,_ in v["wrong_admit"]],
              "wrong_deny":v["wrong_deny"],"correct":v["ok"]} for k,v in summary.items()},
          open(_ROOT/"M8"/"e25_results.json","w"), indent=1)
print("\nwrote M8/e25_results.json")
