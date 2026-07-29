# rag-bench (production-hardening)

[![CI](https://github.com/My-Denia/rag-production-hardening/actions/workflows/ci.yml/badge.svg)](https://github.com/My-Denia/rag-production-hardening/actions/workflows/ci.yml)

Evaluable retrieval-augmented QA with **LangChain / LangGraph** stateful orchestration, **external-judge** metrics, **frozen arms**, **holdout-confirmed selection**, **multi-stage process-kill recovery**, and **session isolation**.

Synthetic corpus only (fictional company handbook). Offline-first — MiniLM local semantic OK; no paid LLM API required.

> **Honest scope:** research / engineering bench for retrieval evaluation and recovery behavior. **Not** a production multi-tenant deploy, compliance certification, or chat product.

| | |
| --- | --- |
| Version | **0.2.0** |
| License | [MIT](LICENSE) |
| Data card | [DATA_CARD.md](DATA_CARD.md) |
| Evidence index | [docs/public-evidence.md](docs/public-evidence.md) |

## Install

Requires Python ≥3.11. **Recommended: Python 3.12**.

```bash
cd <clone-dir>
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Unix
# source .venv/bin/activate

python -m pip install -U pip
pip install -e ".[dev,semantic]"
```

## One recompute command

```bash
python -m rag_bench.run_all
# equivalent:
python -m rag_bench
```

This runs M0→M6:

1. **M0** Freeze `data/regression_v1/` (35 immutable labels) + verify `config/{categories,arms,selection_rules}.yaml`
2. **M1** Build ≥100 stratified labels → `data/dev/` + sealed `data/holdout/`
3. **M2** Evaluate all 10 frozen arms on **DEV only**; margin calib; determinism check
4. **M3** Dev shortlist (size=1) → holdout confirm-only (no re-rank) → quality report + regression-v1
5. **M4** Multi-stage process-kill recovery (retrieve/rerank/generate) + corrupt checkpoint
6. **M5** ≥20 concurrent sessions, 0 cross-talk
7. **M6** Docs, traces, status

### Holdout seal

- Holdout labels require `RAG_HOLDOUT_UNLOCK=1`
- Sole intended unlock path: `finalize_selection` after `results/dev_shortlist.json`
- Fail → `NEEDS_REPLAN` (no arm shopping on holdout)

## Frozen configs

| File | Role |
| --- | --- |
| `config/categories.yaml` | 11 taxonomy IDs, min 8 each |
| `config/arms.yaml` | Exactly 10 evaluation arms |
| `config/selection_rules.yaml` | Pre-registered lex + bootstrap significance |
| `config/selected.yaml` | Holdout-confirmed winner (or `holdout_failed`) |

Baseline arm: `hash_dense_k8_r1`. Query axis: `minilm_mq_k8_r1` (`multi_query_static`). Abstain: `minilm_dense_k8_r1_margin`.

## Metrics (external judge)

| Metric | Definition |
| --- | --- |
| **recall_at_k** | Any retrieved chunk span overlaps gold (answerable only) |
| **attribution_rate** | Cited gold-overlap **and** all `must_contain` (canonical for selection) |
| **refusal_f1** | Abstain `[ABSTAIN]` vs unanswerable |
| **unsupported_answer_rate** | Non-empty answer without gold cite / unanswerable non-refuse |

No LLM-as-judge. No silent gold edits after scores.

**Dual regression gate:** report both `recall_hits=35` and `attribution_hits=35` on `data/regression_v1/` — recall-only is not dual success.

## Pipeline

```
retrieve → rerank → generate → finalize
```

- **Embeddings:** hash | minilm (dense); BM25 sparse; **hybrid_rrf**
- **Query:** raw | multi_query_static (static synonym table only)
- **Abstain:** margin_0.05 vs dev median top-1
- **Orchestration:** LangGraph + SqliteSaver; process-kill at stage sidecar hold; resume from checkpoint

## Key artifacts

| Artifact | Path |
| --- | --- |
| Freeze report | `results/freeze_report.json` |
| Dev arm metrics | `results/arms/dev_<arm_id>.json` |
| Shortlist | `results/dev_shortlist.json` |
| Holdout confirmation | `results/holdout_confirmation.json` |
| Quality report | `results/quality_report.json` |
| Regression | `results/regression_metrics.json` |
| Recovery | `results/recovery_report.json` |
| Concurrency | `results/concurrency_report.json` |
| Determinism | `results/determinism_check.json` |
| Security scan | `results/release/security-scan.json` |
| Status | `docs/status.md` |

History preserved: `results/run_v1/`, `results/run_v2/` (metrics + selected snapshots).

## Documentation

| Doc | Path |
| --- | --- |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Evaluation protocol | [docs/evaluation-protocol.md](docs/evaluation-protocol.md) |
| Reproduction | [docs/reproduction.md](docs/reproduction.md) |
| Datasets | [docs/datasets.md](docs/datasets.md) |
| Tradeoffs | [docs/tradeoffs.md](docs/tradeoffs.md) |
| Public evidence | [docs/public-evidence.md](docs/public-evidence.md) |
| Preflight | [docs/preflight.md](docs/preflight.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Security | [SECURITY.md](SECURITY.md) |
| Third-party | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |
| Citation | [CITATION.cff](CITATION.cff) |

## Tests

```bash
pytest -q
```

## Sanitize (maintainers)

After recompute, strip absolute machine paths and prune non-whitelisted results:

```bash
python scripts/sanitize_public_tree.py --prune --write-reproduction --write-manifest
```

## Non-goals

Paid API default; real PII; multi-tenant production deploy; hardcoding gold answers; holdout re-rank shopping; claiming dual 35/35 from recall alone.
