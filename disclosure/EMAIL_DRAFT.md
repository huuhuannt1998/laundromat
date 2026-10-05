# Disclosure e-mail — draft, NOT SENT

**Status: awaiting PI authorisation. Do not send without it.**

## Where this address came from

The project states its own reporting policy in `SECURITY.md` at the pinned commit
`a75007d5ca4aaf8df2e3f055f318557a965b93aa`:

> `Model Provenance Kit` leverages GitHub's private vulnerability reporting. […] If you've been
> unable to successfully draft a vulnerability report via GitHub or have not received a response
> during the allotted response window, please reach out via the
> [Cisco Open security contact email](mailto:oss-security@cisco.com).

Verified 2026-09-08 against the local clone at the pinned commit and against the file as served on
`github.com/cisco-ai-defense/model-provenance-kit`. Both read the same.

**Channel of record.** GitHub private vulnerability reporting on
`cisco-ai-defense/model-provenance-kit` (the project's stated primary channel).
**E-mail fallback.** `oss-security@cisco.com` (Cisco Open security contact, named in `SECURITY.md`).
**Escalation if unacknowledged.** `psirt@cisco.com`, the Cisco Product Security Incident Response
Team address published on Cisco's Security Vulnerability Policy page; PSIRT states an
around-the-clock intake and supports PGP/GPG. Use only if the two channels above go unanswered
past the three-business-day acknowledgement `SECURITY.md` commits to.

The `SECURITY.md` acknowledgement window is three business days, with a detailed response within a
further three.

---

## Draft

**To:** oss-security@cisco.com
**Cc:** (GitHub private advisory thread, once opened)
**Subject:** Model Provenance Kit 1.1.0 (a75007d) — two defects: no-verdict on absent `architectures`, and tier-1 Confirmed Match with null weight signals

Hello,

We are an academic group evaluating publicly released model-lineage verification tools. We have a
paper under submission to IEEE S&P 2027 that reports two defects in Model Provenance Kit, and we
are sending you the full report before we submit, in line with the conference's disclosure
requirement.

We opened a private vulnerability report on the repository as `SECURITY.md` directs. This e-mail is
the fallback channel that document names, sent so the report does not sit unnoticed.

The attached document, `DISCLOSURE_REPORT.md`, describes both issues with root cause, affected
public models, prevalence, and reproduction commands against version 1.1.0 at commit
`a75007d5ca4aaf8df2e3f055f318557a965b93aa`. In brief:

1. **No verdict when `architectures` is absent from `config.json`.** The CLI aborts with
   "Failed to extract base features" and returns nothing. The cause is at
   `core/signals/metadata.py:126`: `getattr(config, "architectures", ["unknown"])` cannot reach its
   default, because `transformers.PretrainedConfig` declares the attribute as `None`, and the MFI
   model requires `list[str]`. Eight public models from five publishers, including four AllenAI
   domain-adaptive rungs and a first-party Google BERT release, return no verdict as published; in
   a sample of 545 consumer-side configurations, 17 ordinary models (3.1%) lack the key, eleven
   of them DeBERTa. An empty `architectures` list already degrades cleanly to tier 2, so the
   degradation path exists and is simply not reached.

2. **Tier-1 `Confirmed Match` issued with all weight signals `null`.**
   `facebook/bart-large-cnn` against `sshleifer/distilbart-cnn-12-6` returns
   `pipeline_score` 1.0 and `Confirmed Match` with `identity_score` and all five weight signals
   `null`. The same tier pin also overrides weight evidence that was computed and disagreed:
   `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`, whose card states it was pretrained from
   scratch, is confirmed against `bert-base-uncased` at tier 1 while its identity score, 0.5248,
   sits below your own 0.65 weak-match line. Because `architectures` and `model_type` are
   publisher-written and unattested, one field edit with weights byte-identical moves
   `legal-bert-base-uncased` from tier 3 (`Not Matched` as the hub serves it) to tier-1
   `Confirmed Match` against `bert-base-uncased`.

Both are properties of published Apache-2.0 source that any reader can inspect, so we do not
believe an embargo is needed, and we are not asking for one. We would value an acknowledgement we
can cite, and we will record your response, or its absence, accurately in the paper. If you would
prefer a different timeline or a different channel, tell us and we will follow it.

We hold no ambiguity about what we are reporting: this is an evaluation of a released tool, not a
claim that anyone is attacking it. Our measurements were made on one laptop against public
checkpoints, with no access requested and no Cisco system touched. We are happy to share the
reproduction artifact, and to review any fix against the same corpus.

Thank you for publishing the tool and the constitution as open source. The second issue is one we
could only find and describe precisely because the source and the design rationale are both public.

Regards,

[AUTHORS — de-anonymise only after the paper is out of review, or send under a role address and
state that the submission is under double-blind review]

**Attachment:** `DISCLOSURE_REPORT.md`

---

## Checklist before sending

- [ ] PI has authorised transmission.
- [ ] GitHub private vulnerability report opened first; its URL pasted into the e-mail.
- [ ] Decide the sender identity. The S&P submission is anonymous; sending from an institutional
      address before the rebuttal is compatible with double-blind review (the vendor is not a
      reviewer), but the choice is the PI's.
- [ ] `DISCLOSURE_REPORT.md` attached, with the "Report date" line filled in.
- [ ] Record the send date, the acknowledgement date, and any response in
      `REVIEW_CHECKLIST.md` and in the paper's Ethical Considerations section.
- [ ] Re-verify the pin immediately before sending. The local clone at
      `M1/oracle/model-provenance-kit` is present, checked out at `a75007d` (verified 2026-09-25;
      its `uv.lock` carries a local modification). `make check-version` fetches `origin` and
      compares `HEAD` with `origin/main`, so it will print MOVED while upstream's tip is the
      lockfile-only `87fe4b7`; confirm with `git diff a75007d origin/main --stat` that only
      `uv.lock` differs. Without network access to the clone, use
      `curl -sS https://api.github.com/repos/cisco-ai-defense/model-provenance-kit/commits/main`.
      Checked 2026-09-17: tip is `87fe4b7`, a `uv.lock`-only bump whose parent is `a75007d`, so
      no source has changed and both issues still reproduce. If the tip moves again, diff it
      before sending and say which commit you tested.
