# Next Phase Runbook (KT10–KT15)

**Generated:** 2026-01-15

This runbook assumes:
- You already completed the original issues set from the starter repo archive.
- You are targeting IJCAI-ECAI 2026 and must convert the work from “concept/demo” into:
  **implemented system + formal core + experiments + reproducibility/compliance**.

## Golden rules
1) Keep `dataset=toy` smoke test green.
2) No PII. Avoid LinkedIn as a primary data source.
3) No unverified references (verify_bib gate).
4) Don’t let LLMs write your final submission text; use them for scaffolding, tests, and automation.

## Order (recommended)
### Phase 1: KT10–KT11 (paper scaffolding + related work + bib gate)
- KT10: build claim→evidence matrix + formal core checklist.
- KT11: related-work coverage matrix + verified-only references.

### Phase 2: KT12 (final experiment freeze)
- lock experiment matrix, multi-seed runs, paper artifacts manifest.

### Phase 3: KT13 (human-centred evaluation or simulation)
- pick user-study or seeded simulation; output one table + one figure.

### Phase 4: KT14 (submission packaging + preflight)
- anonymization checks, dist folder build, preflight script.

### Phase 5: KT15 (release bundle)
- reproduce_small, license/citation, release zip.

## Control points
- CP1: KT10 merged: clear contributions and evidence mapping.
- CP2: KT11 merged: verified references and novelty map.
- CP3: KT12 merged: paper artifacts frozen with hashes.
- CP4: KT13 merged: HAI evaluation artifact exists.
- CP5: KT14 merged: submission bundle passes preflight.
- CP6: KT15 merged: reproduce_small works for others.

