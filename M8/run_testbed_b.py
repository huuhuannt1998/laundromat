"""E25b -- the consumer testbed re-run with ground truth derived from audited labels only.

Why this exists. The frozen E25 (run_testbed.py -> e25_results.json) took `truly_derived`
from the manifest's relationship_class for measured rows and HARD-CODED it to True for every
row of the no-verdict list (M1/results/mpk_crashes.jsonl). The 2026-09-08 review found three
consequences: (i) MiniLM-L12-H384 is distilled from UniLMv2, not from bert-base-uncased, and
google/bert_uncased_L-4_H-256_A-4 is pretrained from scratch, yet both were counted as models
that inherit an advisory published against bert-base-uncased; (ii) Bio_ClinicalBERT is D-CP in
the manifest but its own evidence quote and model card put it in the bert-base-CASED lineage,
so section V-E's table and E25 disagreed about it; (iii) the five models whose configuration we
repaired appeared twice, once as published (no verdict) and once repaired (verdict), and the
repaired state is our instrument, not a state a consumer receives.

This script:
  * takes ground truth for measured rows from the manifest and records the label grade;
  * labels every no-verdict row from the child's own model card, quoted verbatim below, with
    a label_source of `primary` (card statement), `secondary` (release convention or
    associated publication) or `none`;
  * applies one card-evidence override to the manifest (Bio_ClinicalBERT), recorded as such;
  * reports two populations: the 37-row union the frozen run used, and the 32-row
    consumer-view population in which each model appears once, in the configuration state a
    consumer downloads. The manuscript reports the 32-row population.

Pipeline, policy, BOM emitter and licence table are unchanged from E25. Nothing in
run_testbed.py or e25_results.json is modified. Output: M8/e25b_results.json.
"""
import json, subprocess, collections
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
import sys; sys.path.insert(0, str(_ROOT/"M8"))
from emit_bom import emit, validate

POLICY = _ROOT/"M8"/"policy"/"supplychain.rego"
LIC = json.loads((_ROOT/"M8"/"model_licences.json").read_text())
OUT = _ROOT/"M8"/"e25b_results.json"

def gate(bom):
    validate(bom)
    p = subprocess.run(["opa","eval","-I","-d",str(POLICY),"data.supplychain"],
                       input=json.dumps(bom), capture_output=True, text=True)
    r = json.loads(p.stdout)["result"][0]["expressions"][0]["value"]
    return bool(r.get("admit", False)), r.get("deny", [])

def licence_of(m, default="apache-2.0"):
    return LIC.get(m) or default

# ---- card evidence for rows the manifest does not label, or labels against its own quote ----
# Each entry: derived?, label_source, verbatim quotation from the child's model card
# (fetched 2026-09-08), and the reason.
CARD = {
 "microsoft/MiniLM-L12-H384-uncased": dict(derived=False, source="primary",
   quote="12-layer model with 384 hidden size distilled from an in-house pre-trained UniLM v2 model in BERT-Base size",
   why="teacher is UniLMv2, not bert-base-uncased; no weight is inherited from the advisory's ancestor"),
 "google/bert_uncased_L-4_H-256_A-4": dict(derived=False, source="primary",
   quote="We have shown that the standard BERT recipe (including model architecture and training objective) is effective on a wide range of model sizes",
   why="pretrained from scratch with the BERT recipe; distillation is recommended only for downstream fine-tuning"),
 "huawei-noah/TinyBERT_General_4L_312D": dict(derived=True, source="primary",
   quote="In general distillation, we use the original BERT-base without fine-tuning as the teacher and a large-scale text corpus as the learning data",
   why="distilled from BERT-base; the 30,522-entry uncased vocabulary identifies the uncased release"),
 "emilyalsentzer/Bio_ClinicalBERT": dict(derived=False, source="primary",
   quote="Model parameters were initialized with BioBERT (BioBERT-Base v1.0 + PubMed 200K + PMC 270K)",
   why="BioBERT-Base v1.0 is initialised from bert-base-CASED (vocabulary 28,996 cased entries); "
       "as a child of bert-base-UNCASED this is cross-lineage, which is what the manifest's own evidence quote says"),
}
# The four AllenAI rungs are manifest rows (D-CP; biomed GOLD, the other three SILVER); their
# no-verdict rows take the manifest label and grade.

PYTHIA = {"pythia-70m/-deduped": ("EleutherAI/pythia-70m","EleutherAI/pythia-70m-deduped"),
          "pythia-160m/-deduped": ("EleutherAI/pythia-160m","EleutherAI/pythia-160m-deduped"),
          "pythia-410m/-deduped": ("EleutherAI/pythia-410m","EleutherAI/pythia-410m-deduped"),
          "pythia-1b/-deduped": ("EleutherAI/pythia-1b","EleutherAI/pythia-1b-deduped"),
          "pythia-1.4b/-deduped": ("EleutherAI/pythia-1.4b","EleutherAI/pythia-1.4b-deduped")}
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

GRADE = {"GOLD": "primary", "SILVER": "secondary"}
rows=[]; skipped_unknown=0
man = json.loads((_ROOT/"M6"/"gold_lineage_manifest.json").read_text())
for r in man["rows"]:
    if r["analysis_set"]=="excluded": continue
    par, ch = r["parent"], r["child"]
    if par == "pythia":
        if ch not in PYTHIA: continue
        par, ch = PYTHIA[ch]
    v = r.get("verdict") or MEASURED_ELSEWHERE.get((par, ch))
    if v is None:
        skipped_unknown += 1; continue
    derived = r["relationship_class"].startswith("D-")
    source = GRADE.get(r["provenance_confidence"], "none"); override=None
    if ch in CARD and CARD[ch]["derived"] != derived:
        override = dict(manifest_class=r["relationship_class"], card=CARD[ch])
        derived = CARD[ch]["derived"]; source = CARD[ch]["source"]
    rows.append(dict(parent=par, child=ch, truly_derived=derived, label_source=source,
                     manifest_class=r["relationship_class"], verdict=v, no_verdict=False,
                     config_state="repaired" if r.get("config_modified") else "as published",
                     override=override))
for l in (_ROOT/"M1"/"results"/"mpk_crashes.jsonl").read_text().splitlines():
    if not l.strip(): continue
    d=json.loads(l)
    if not d.get("defect"): continue
    ch=d["child"]
    if ch in CARD:
        derived, source, cls = CARD[ch]["derived"], CARD[ch]["source"], "card"
    else:                                   # the four AllenAI rungs: manifest label and grade
        m=[x for x in man["rows"] if x["child"]==ch]
        assert m, f"no manifest row and no card entry for {ch}"
        derived=m[0]["relationship_class"].startswith("D-"); source=GRADE.get(m[0]["provenance_confidence"],"none")
        cls=m[0]["relationship_class"]
    rows.append(dict(parent=d["parent"], child=ch, truly_derived=derived, label_source=source,
                     manifest_class=cls, verdict=None, no_verdict=True, config_state="as published",
                     override=None))

def evaluate(rowset, wiring):
    per=[]; missed=[]; quarantined=[]; ok=0
    for r in rowset:
        cl, pl = licence_of(r["child"]), licence_of(r["parent"])
        adv = ["ADV-2026-0001 poisoned-checkpoint disclosure against the ancestor"]
        got = emit(r["child"], r["parent"], None if r["no_verdict"] else r["verdict"],
                   child_licence=cl, parent_licence=pl, on_no_verdict=wiring, parent_advisories=adv)
        truth = emit(r["child"], r["parent"], "Confirmed Match" if r["truly_derived"] else "Not Matched",
                     child_licence=cl, parent_licence=pl, on_no_verdict=wiring, parent_advisories=adv)
        a_got,_ = gate(got); a_true,_ = gate(truth)
        if a_got == a_true: outcome="correct"; ok+=1
        elif a_got and not a_true: outcome="missed"; missed.append(r["child"])
        else: outcome="quarantined"; quarantined.append(r["child"])
        per.append(dict(child=r["child"], parent=r["parent"], truly_derived=r["truly_derived"],
                        label_source=r["label_source"], config_state=r["config_state"],
                        verdict="no verdict" if r["no_verdict"] else r["verdict"], outcome=outcome))
    return dict(missed=missed, quarantined=quarantined, correct=ok, n=len(rowset), rows=per)

union37 = rows
repaired = {r["child"] for r in rows if r["config_state"]=="repaired"}
consumer = [r for r in rows if not (r["config_state"]=="repaired" and r["child"] in repaired)]
result = dict(
  note="'union' reproduces the frozen E25 population (repaired rows plus their as-published "
       "no-verdict rows). 'consumer' keeps each model once, in the state a consumer downloads: "
       "the manuscript reports 'consumer'.",
  skipped_for_want_of_verdict=skipped_unknown,
  card_evidence=CARD,
  overrides=[dict(child=r["child"], **r["override"]) for r in rows if r.get("override")],
  populations={})
for name, rowset in (("union", union37), ("consumer", consumer)):
    result["populations"][name] = {w: evaluate(rowset, w) for w in ("fail-open","fail-closed")}
    print(f"\n=== population '{name}': {len(rowset)} pairs "
          f"({sum(1 for r in rowset if r['no_verdict'])} no-verdict; "
          f"{sum(1 for r in rowset if r['truly_derived'])} truly derived) ===")
    print(f"{'wiring':<12}{'missed':>8}{'quarantined':>13}{'correct':>9}")
    for w,s in result["populations"][name].items():
        print(f"{w:<12}{len(s['missed']):>8}{len(s['quarantined']):>13}{s['correct']:>9}")
        for c in s["missed"]: print(f"     missed:      {c}")
        for c in s["quarantined"]: print(f"     quarantined: {c}")
OUT.write_text(json.dumps(result, indent=1))
print(f"\nwrote {OUT.relative_to(_ROOT)}")
