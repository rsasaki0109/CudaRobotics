# Paper Claim-to-Evidence Contracts

These manifests are the authoritative readiness ledger for the CudaRobotics
systems paper and the contact-rich Diff-MPPI paper. They separate three facts
that prose drafts often blur:

- `supported`: every referenced artifact exists, matches its SHA-256, and
  passes its declared numeric assertions;
- `partial` or `planned`: implementation or pilot evidence exists, but the
  submission claim is not yet proved;
- `ready`: every submission-required claim is supported by complete valid
  evidence.

Validate schema, hashes, numeric assertions, and claim consistency:

```bash
python3 scripts/validate_paper_artifacts.py
```

This command succeeds when the ledger is internally valid even if a paper is
not ready. The output always reports `ready: false` until every required claim
is proved. The release/submission gate is deliberately stricter:

```bash
python3 scripts/validate_paper_artifacts.py --require-ready
```

Text artifacts may declare `normalization: text_lf` so the same tracked content
has one hash on LF and CRLF worktrees. Binary artifacts must use their raw byte
hash without normalization.

Never change a claim from `planned` or `partial` to `supported` merely because
the implementation exists. First attach the generated artifact, freeze its
SHA-256, and add assertions that test the exact metric used in the prose.
Negative results should remain declared evidence rather than being removed.

For the ready contact-rich ledger, the submission bundle has a second,
portable validation layer:

```bash
python3 scripts/assemble_contact_submission_bundle.py \
  --output-dir build/contact_submission_bundle \
  --venue VENUE --artifact-url https://ANONYMOUS_ARTIFACT_URL
python3 scripts/validate_contact_submission_bundle.py \
  build/contact_submission_bundle/submission_manifest.json \
  --commit "$(git rev-parse HEAD)" --require-ready
python3 scripts/archive_contact_submission_bundle.py \
  build/contact_submission_bundle/submission_manifest.json \
  --output build/contact-rich-diff-mppi-submission.zip \
  --commit "$(git rev-parse HEAD)"
python3 scripts/validate_contact_submission_archive.py \
  build/contact-rich-diff-mppi-submission.zip \
  --checksum build/contact-rich-diff-mppi-submission.zip.sha256 \
  --commit "$(git rev-parse HEAD)"
```

This does not replace the source ledger. It produces an anonymous copy whose
machine-specific absolute paths are redacted and rehashed, then reruns the same
claim assertions against the copied evidence. The final canonical ZIP and its
SHA-256 sidecar are the submission artifacts. The archive validator checks the
anonymous manifest, exact inventory, source commit, checksum, CRC, canonical
metadata, and safe paths before extracting any member.
