# [KT2] Embedding pipeline + disk cache + model versioning

## Why
Embeddings are the backbone of ADS. We need deterministic caching, model_id versioning, and the ability to run ablations across encoders.

## Scope
### In scope
- Implement SentenceTransformers encoder option (behind an extra dependency)
- Add disk cache keyed by sha256(text)
- Store model_id in outputs and caches
- Document how to run stub vs sbert

### Out of scope
- Remote proprietary embeddings (unless strictly necessary)

## Deliverables
- [ ] ads_core.embed.SentenceTransformersEncoder
- [ ] DiskEmbeddingCache that avoids recomputation
- [ ] Docs update: reproducibility checklist

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `python -m experiments.run dataset=toy embedding=stub lenses=identity`
- [ ] `python -m experiments.run dataset=toy embedding=sbert_allminilm lenses=identity  # requires extras`

## Expected artifacts (examples)
- `data/embeddings/<model_id>/index.json + vecs.npy (or equivalent cache folder)`
- `experiments/reports/<run_id>/config_resolved.yaml includes embedding.model_id`

## Implementation notes
- Keep stub encoder for CI and quick tests.

## Prompt for Cursor Composer
> Add sentence-transformers encoder + disk cache by sha256(text), with model_id versioning. Prove cache hits on second run.

## Prompt for Claude Code
> Implement embedding cache and sbert encoder. Run a toy experiment twice and show that the second run reuses the cache.
