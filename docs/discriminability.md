# Discriminability report

Historical notes from ablation-style factor contrasts (see also `results/discriminability.json` and `results/embeddings_contrast.json`).

Label set / run context: prior `run_v2` analysis cells.

## Key factors (summary)

| factor | claim |
| --- | --- |
| retrieval | discriminative (off vs on) |
| rerank | often non-discriminative on this tiny corpus |
| chunk_strategy | weak / non-discriminative between fixed and recursive on many pairs |
| top_k | discriminative at low k (2 vs 4/8) |
| threshold | non-discriminative in tested grid |
| embeddings (hash vs minilm) | semantic lift observed in contrast artifact |

## Embeddings contrast (primary)

From `results/embeddings_contrast.json` (shipped historical / regenerated analysis):

- type: `hash_vs_minilm_semantic`
- MiniLM typically higher recall than hash on the hard label set used for contrast

## Interpretation limits

- Small n → wide CIs; “non-discriminative” ≠ “useless in production”
- Pre-registered arm selection does not re-open the ablation grid after holdout
- Prefer dual regression + holdout confirmation as release gates over factor storytelling
