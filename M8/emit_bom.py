"""Turn a verifier verdict into a standards-conformant ML-BOM.

This is the join the paper has been asserting rather than showing. Section I claims
lineage verification "populates bills of materials, propagates licence obligations and
scopes inherited vulnerabilities". Those are not our inventions: CycloneDX 1.6 defines
`component.pedigree.ancestors` as "zero or more components in which a component is
derived from", and `machine-learning-model` is a first-class component type. A lineage
verdict is exactly what fills that field.

Every BOM emitted here is validated against the OFFICIAL pinned schema
(M8/bom-1.6.schema.json, fetched from CycloneDX/specification), so conformance is
checked rather than claimed.

THE LOAD-BEARING MODELLING CHOICE, made explicit because everything downstream turns on
it: how does a verdict become an edge?

    Confirmed / High-Confidence / Weak Match -> pedigree.ancestors = [parent]
    Not Matched                              -> no pedigree edge
    NO VERDICT (verifier raised, no output)  -> depends on the consumer's wiring

The third case is the whole point. The tool returns nothing, so a consumer must decide
what "nothing" means, and there are only two defensible choices:

    fail-open   : treat as no relationship  -> no edge, obligations do not attach
    fail-closed : treat as unknown          -> pedigree.variants, quarantine

We emit both and evaluate both, because a reviewer will rightly say a careful pipeline
would fail closed, and the interesting question is whether that actually rescues it.
"""
import json, pathlib, argparse, datetime
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
SCHEMA = json.loads((_ROOT/"M8"/"bom-1.6.schema.json").read_text())

# CycloneDX requires licence.id to be an SPDX identifier and offers licence.name for
# anything else. Model licences that carry the strongest derivative obligations -- the
# BigScience RAIL family above all -- are NOT SPDX identifiers, so a conformant ML-BOM
# must record them as free text. That is a real property of the deployment surface: the
# obligations most in need of automated propagation are the ones the standard can only
# carry unstructured. We take the SPDX set from the pinned schema rather than hardcoding.
SPDX = set(SCHEMA["definitions"]["license"]["properties"]["id"].get("enum", []))
SPDX_LOWER = {s.lower(): s for s in SPDX}

def _licence_entry(lic):
    """SPDX ids go in `id`; everything else in `name`, as the schema requires."""
    if lic is None: return None
    canon = SPDX_LOWER.get(lic.lower())
    return {"license": {"id": canon}} if canon else {"license": {"name": lic}}

MATCH_VERDICTS = {"Confirmed Match", "High-Confidence Match", "Weak Match"}

def component(name, licence=None, advisories=None, ctype="machine-learning-model"):
    c = {"type": ctype, "name": name, "bom-ref": name}
    if licence:
        e = _licence_entry(licence)
        if e: c["licenses"] = [e]
    if advisories:
        c["properties"] = [{"name": "laundromat:advisory", "value": a} for a in advisories]
    return c

def emit(child, parent, verdict, *, child_licence, parent_licence,
         parent_advisories=None, on_no_verdict="fail-open", timestamp="2026-08-31T00:00:00Z"):
    """One BOM describing the child, with whatever lineage edge the verdict supports."""
    c = component(child, licence=child_licence)
    if verdict in MATCH_VERDICTS:
        c["pedigree"] = {"ancestors": [component(parent, licence=parent_licence,
                                                 advisories=parent_advisories)]}
    elif verdict is None:                      # the verifier produced no output
        if on_no_verdict == "fail-closed":
            c["pedigree"] = {"variants": [component(parent, licence=parent_licence,
                                                    advisories=parent_advisories)],
                             "notes": "verifier returned no verdict; relationship unknown"}
        # fail-open: no pedigree at all, indistinguishable from "not matched"
    # "Not Matched": no pedigree edge
    return {"bomFormat": "CycloneDX", "specVersion": "1.6", "version": 1,
            "metadata": {"timestamp": timestamp,
                         "component": {"type": "application", "name": "consumer-pipeline"}},
            "components": [c]}

def validate(bom):
    import jsonschema
    jsonschema.validate(bom, SCHEMA)     # raises on non-conformance
    return True

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--child", required=True); a.add_argument("--parent", required=True)
    a.add_argument("--verdict", default=None)
    a.add_argument("--child-licence", default="Apache-2.0")
    a.add_argument("--parent-licence", default="Apache-2.0")
    a.add_argument("--on-no-verdict", default="fail-open", choices=["fail-open","fail-closed"])
    n = a.parse_args()
    b = emit(n.child, n.parent, n.verdict or None,
             child_licence=n.child_licence, parent_licence=n.parent_licence,
             on_no_verdict=n.on_no_verdict)
    validate(b)
    print(json.dumps(b, indent=1))
