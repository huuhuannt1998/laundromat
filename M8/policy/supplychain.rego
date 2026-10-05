# Consumer-side supply-chain policy over a CycloneDX 1.6 ML-BOM.
#
# These are the two obligations the paper's own framing names --- propagating licence
# obligations and scoping inherited vulnerabilities --- expressed as an admission gate,
# plus the quarantine rule a fail-closed consumer needs. Nothing here is specific to the
# verifier under study: the policy reads only standard pedigree fields, so any lineage
# source that populates them is subject to the same outcomes.
package supplychain
import rego.v1

# Licences whose terms bind derivative works. These are the real licences carried by
# models in our corpus, not stand-ins: BLOOM ships under the BigScience RAIL licence,
# whose Attachment A use restrictions explicitly apply to derivatives, and legal-bert
# ships under CC-BY-SA-4.0, which is share-alike.
binds_derivatives := {"bigscience-bloom-rail-1.0", "cc-by-sa-4.0",
                      "gpl-3.0", "gpl-3.0-only", "agpl-3.0"}

# A licence may be an SPDX id or, for the model-specific licences the SPDX list does not
# cover, free-text name. A consumer policy must read both or it silently misses exactly
# the licences that bind derivatives most tightly.
lic(c) := l if { some e in c.licenses; l := lower(e.license.id) }
lic(c) := l if { some e in c.licenses; not e.license.id; l := lower(e.license.name) }

# ---- Obligation propagation -------------------------------------------------------
# A component asserted to derive from an ancestor under a derivative-binding licence
# must carry that licence too.
deny contains msg if {
    some c in input.components
    some a in c.pedigree.ancestors
    binds_derivatives[lic(a)]
    lic(c) != lic(a)
    msg := sprintf("LICENCE: %s carries %s but derives from %s under %s, which binds derivatives",
                   [c.name, lic(c), a.name, lic(a)])
}

# ---- Inherited advisory scope ------------------------------------------------------
# A component inherits any advisory attached to an ancestor.
deny contains msg if {
    some c in input.components
    some a in c.pedigree.ancestors
    some p in a.properties
    p.name == "laundromat:advisory"
    msg := sprintf("ADVISORY: %s inherits %s from ancestor %s", [c.name, p.value, a.name])
}

# ---- Fail-closed quarantine --------------------------------------------------------
# A consumer that treats "no verdict" as "unknown" records the candidate under
# pedigree.variants and must quarantine rather than admit.
deny contains msg if {
    some c in input.components
    some v in c.pedigree.variants
    msg := sprintf("QUARANTINE: %s has an unresolved relationship to %s", [c.name, v.name])
}

admit if count(deny) == 0
