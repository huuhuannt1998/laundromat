"""E43 -- the 32-model admission outcomes if legal-bert were read in the other file format.

Review round 7 (N3). Section V-D (E35/E36) shows legal-bert scoring 0.6061, Not Matched, from the
pytorch_model.bin the hub serves, and 0.7391, Weak Match, from safetensors with every tensor kept.
Table 12 uses the as-served reading. This re-runs the unchanged pipeline (emit_bom, the pinned
schema, the OPA policy) on the consumer population recorded in M8/e25b_results.json, first as
recorded (it must reproduce 8/5/19 and 3/8/21), then with legal-bert's verdict set to Weak Match.
No other row changes. Writes a NEW file, M8/e43_testbed_format.json.
"""
import json, subprocess, pathlib, sys
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
sys.path.insert(0, str(ROOT/"M8"))
from emit_bom import emit, validate
POLICY = ROOT/"M8"/"policy"/"supplychain.rego"
LIC = json.loads((ROOT/"M8"/"model_licences.json").read_text())
def gate(bom):
    validate(bom)
    p = subprocess.run(["opa","eval","-I","-d",str(POLICY),"data.supplychain"],
                       input=json.dumps(bom), capture_output=True, text=True)
    return bool(json.loads(p.stdout)["result"][0]["expressions"][0]["value"].get("admit", False))
rows = json.loads((ROOT/"M8"/"e25b_results.json").read_text())["populations"]["consumer"]["fail-open"]["rows"]
def evaluate(override):
    out = {}
    for wiring in ("fail-open", "fail-closed"):
        c = dict(missed=[], quarantined=[], correct=0)
        for r in rows:
            v = override.get(r["child"], r["verdict"]); v = None if v == "no verdict" else v
            cl, pl = LIC.get(r["child"]) or "apache-2.0", LIC.get(r["parent"]) or "apache-2.0"
            adv = ["ADV-2026-0001 poisoned-checkpoint disclosure against the ancestor"]
            got = gate(emit(r["child"], r["parent"], v, child_licence=cl, parent_licence=pl,
                            on_no_verdict=wiring, parent_advisories=adv))
            tru = gate(emit(r["child"], r["parent"], "Confirmed Match" if r["truly_derived"] else "Not Matched",
                            child_licence=cl, parent_licence=pl, on_no_verdict=wiring, parent_advisories=adv))
            if got == tru: c["correct"] += 1
            elif got and not tru: c["missed"].append(r["child"])
            else: c["quarantined"].append(r["child"])
        out[wiring] = dict(missed=len(c["missed"]), quarantined=len(c["quarantined"]), correct=c["correct"],
                           quarantined_models=c["quarantined"])
    return out
res = dict(as_served=evaluate({}),
           legalbert_weak=evaluate({"nlpaueb/legal-bert-base-uncased": "Weak Match"}))
(ROOT/"M8"/"e43_testbed_format.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(k, {w: (x["missed"], x["quarantined"], x["correct"]) for w, x in v.items()})
