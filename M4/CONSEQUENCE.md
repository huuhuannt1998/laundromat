# The attestation-stack consequence — and why it runs the opposite way to the threat model

## The inversion

LAUNDROMAT was scoped around a **laundering** adversary: a publisher who breaks the derivation edge
to escape obligations. M2 measured that adversary and it **failed** — no zero-training-compute
transform evades the detector at any usable capability, and Cisco's Constitution C1 holds with margin.

But the same experiments surfaced the opposite error at a **measured 4/4 rate**: MPK returns
*Confirmed Match* — its highest verdict — on four independently-trained model pairs, and the effect
strengthens with scale. So the risk to the attestation stack is not primarily that a guilty model
**escapes** an edge. It is that an innocent model **acquires** one.

That inversion matters because every control below is built to act on the *presence* of an edge, and
each is therefore fail-loud in the direction that is common and fail-silent in the direction that is rare.

## The propagation mechanism: transitivity turns one bad edge into a merged tree

Constitution §2.2 condition **C5** makes provenance formally transitive: if A→B→C→D each satisfy
C1–C4, then A is provenance-linked to D. §10.1.3 concedes the detection consequence but keeps the
logical property.

A false edge therefore does not mislabel one model. It **merges two lineage trees**: every descendant
of the falsely-attached model inherits everything upstream of the attachment point. One false positive
between two independently-trained bases contaminates both subtrees, and the contamination is invisible
because each individual edge looks locally plausible.

## Four consumers, both directions

| consumer | false edge (over-attribution) — **measured 4/4** | missing edge (laundering) — **not achieved at κ=0** |
|---|---|---|
| **ML-BOM inheritance** (CycloneDX) | Independent model inherits the full upstream component list. Via C5, the error propagates to every descendant. The BOM is *more* wrong the deeper the tree. | Component omitted; downstream consumers under-report. Requires an adversary who can afford what M2 shows they cannot. |
| **License obligation** | An independent lab that followed a *public recipe* is attributed as a derivative of a restrictively-licensed model, and acquires obligations — attribution, share-alike, field-of-use limits — it does not owe. This is legal exposure created by a similarity score. | A genuine derivative escapes copyleft/attribution terms. The original threat model; not reachable at κ=0. |
| **Inherited-vulnerability (CVE) mapping** | Model inherits CVEs it does not have → remediation work against a phantom parent, and alert fatigue. **The realistic outcome is that the control gets disabled**, which also destroys the true-positive value. | A real inherited vulnerability goes unflagged. Higher per-instance severity, far lower measured rate. |
| **Region-of-origin policy** (Cerberus → Secure Access) | An independent model is linked to one from a blocked region and is **blocked**. This is a denial of service against a legitimate publisher, and the trigger is *training on a shared public recipe* — no adversary required. | A model from a blocked region is admitted. |

## Why the over-attribution direction is the one to design against

1. **It has a measured rate.** 4/4 on the same-recipe arm; over-attribution is not hypothetical.
2. **It needs no adversary.** It fires on ordinary, honest publishing behaviour — following a public recipe.
3. **It is self-amplifying via C5**, whereas a missing edge is self-limiting (one absent link).
4. **It degrades the control's own credibility.** False blocks and phantom CVEs are what get a control
   switched off, at which point the true positives are lost too.
5. **The failure is silent to the vendor's own benchmark** — the one diagnostic pair is labelled a
   positive, so the reported accuracy cannot see this mode (M1 finding B5).

## What the stack should require

The controls above should not act on a **verdict**; they should act on a verdict **plus the evidence
class that produced it**. Concretely, and matching what M4's defence measures:

- A derivation edge asserted on **recipe-determined evidence alone** (EAS/END/LEP/NLF) is not a
  sufficient basis for a license obligation, a CVE inheritance, or a region-of-origin block.
- Obligations should attach only where **positional weight evidence** is present — LAP-aligned so that
  a free permutation cannot remove it (margin +0.2996 across fine-tune, multitask *and* distillation,
  versus +0.0117 for MPK's best signal).
- Where shapes do not match and the positional check cannot run, the correct output is
  **"insufficient evidence"**, not a confidence score. MPK currently has no such state: every pair
  receives one of four verdicts, and the weakest is "Not Matched", which asserts a negative rather
  than declining to answer.

## Honest limits of this analysis

This is an **argument from measured detector behaviour to deployment consequence**, not a measurement
of the consequence itself. I have not instrumented a real ML-BOM pipeline, a license checker, or a
Secure Access policy engine, and I make no claim about how any specific product consumes a provenance
edge. The consumer behaviours above are taken from Cisco's own published descriptions of the stack.
Confirming that a false edge actually produces a false block would require access to the deployed
policy engine, and that is out of scope here. The measured 4/4 false-positive rate and the C5
transitivity property are the parts that rest on evidence.
