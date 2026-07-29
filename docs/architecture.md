# Architecture

## Purpose

`rag-bench` is an **evaluable** retrieval-augmented QA bench with:

- Frozen evaluation arms (`config/arms.yaml`)
- Pre-registered selection rules (`config/selection_rules.yaml`)
- Dev shortlist + holdout confirm-only (no holdout re-rank shopping)
- External-judge metrics (no LLM-as-judge)
- LangGraph stateful orchestration with multi-stage process-kill recovery
- Session isolation concurrency probe

It is **not** a production multi-tenant service.

## Pipeline

```
retrieve → rerank → generate → finalize
```

Orchestrated as a LangGraph state machine with SQLite checkpointing (`langgraph-checkpoint-sqlite`). Stage sidecars support process-kill freeze/resume tests.

## Layout

| Path | Role |
| --- | --- |
| `src/rag_bench/` | Package: index, retrieve, rerank, generate, eval, arms, selection, recovery, concurrency |
| `config/` | Frozen YAML: categories, arms, selection_rules, selected, ablation_grid |
| `data/corpus/` | Synthetic handbook documents |
| `data/dev/`, `data/holdout/`, `data/regression_v1/` | Label splits |
| `results/` | Metrics and reports (public whitelist) |
| `tests/` | Unit/integration tests (holdout seal, metrics, recovery, concurrency) |
| `docs/` | Human documentation |

## Embeddings & retrieval

| Backend | Notes |
| --- | --- |
| `hash` | Deterministic local dense baseline (no model download) |
| `minilm` | `sentence-transformers` all-MiniLM-L6-v2 (optional `[semantic]`) |
| `bm25` | Sparse via `rank-bm25` |
| `hybrid_rrf` | Reciprocal rank fusion of dense + sparse |

Chunk strategies: `fixed_256`, `fixed_512`, `recursive_400_80`.

Query strategies: `raw`, `multi_query_static` (static synonym table only — no LLM expansion).

Abstain: optional margin threshold vs dev median top-1 (`minilm_dense_k8_r1_margin`).

## Metrics (external judge)

| Metric | Definition |
| --- | --- |
| **recall_at_k** | Any retrieved chunk span overlaps gold (answerable only) |
| **attribution_rate** | Cited gold-overlap **and** all `must_contain` (canonical for selection) |
| **refusal_f1** | Abstain `[ABSTAIN]` vs unanswerable |
| **unsupported_answer_rate** | Non-empty answer without gold cite / unanswerable non-refuse |

No silent gold edits after scores.

## Selection protocol

1. Evaluate all frozen arms on **DEV only**
2. Rank by pre-registered rules → shortlist size 1
3. Holdout evaluate primary (+ baseline) **confirm only** — no re-rank
4. Write `config/selected.yaml` and `results/holdout_confirmation.json`
5. Regression-v1 dual gate: 35/35 recall **and** attribution on immutable labels

Holdout files require `RAG_HOLDOUT_UNLOCK=1` for direct reads; `finalize_selection` is the intended unlock path.

## Recovery & concurrency

- **Multi-stage recovery:** kill worker at retrieve/rerank/generate holds; resume from checkpoint; hash match vs control; corrupt checkpoint must raise typed error.
- **Concurrency:** ≥20 parallel sessions; assert 0 cross-talk and kill isolation.

## One-command recompute

```bash
python -m rag_bench.run_all
```

Stages M0→M6 inside the package (freeze → labels → arms → shortlist/holdout/regression → recovery → concurrency → docs).
