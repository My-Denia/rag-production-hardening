# Public evidence index

Pointers to machine-checkable artifacts in this repository. Prefer these files over narrative claims.

## Release packaging

| Artifact | Path |
| --- | --- |
| Security scan | `results/release/security-scan.json` |
| Reproduction summary | `results/release/reproduction.json` |
| File checksums | `results/release/public-manifest.sha256` |
| Preflight | `docs/preflight.md` |
| Data card | `DATA_CARD.md` |
| License | `LICENSE` |

## Core gates

| Gate | Path | Pass reading |
| --- | --- | --- |
| Freeze | `results/freeze_report.json` | `ok` / verified regression freeze |
| Selection rules hash | `results/m0_selection_rules.sha256` | matches rules content at M0 |
| Dev shortlist | `results/dev_shortlist.json` | `shortlist_size=1`, primary arm |
| Holdout confirmation | `results/holdout_confirmation.json` | `pass=true`, `re_ranked_on_holdout=false` |
| Quality report | `results/quality_report.json` | aggregates arms + selection |
| Regression metrics | `results/regression_metrics.json` | `recall_hits=35`, `attribution_hits=35` |
| Regression RCA | `results/regression_rca.json` | `dual_35_35` / `ok` |
| Determinism | `results/determinism_check.json` | identical rerun |
| Recovery | `results/recovery_report.json` | stages + corrupt ok |
| Concurrency | `results/concurrency_report.json` | `n_ok=20`, `cross_talk=0` |
| Embedding backend | `results/embedding_backend.json` | `minilm_success` (if semantic install) |
| Hybrid vs dense | `results/hybrid_vs_dense.json` | differs on some qid |
| Metrics (holdout winner) | `results/metrics.json` | holdout primary metrics mirror |

## Per-arm metrics

All under `results/arms/`:

- `dev_<arm_id>.json` — dev evaluation
- `holdout_<arm_id>.json` — holdout for primary/baseline as produced
- `minilm_dense_k8_r1_margin_calib.json` — margin calibration

## Historical archives

| Path | Role |
| --- | --- |
| `results/run_v1/metrics.json`, `selected.yaml` | v1 snapshot |
| `results/run_v2/metrics.json`, `selected.yaml` | v2 snapshot |
| `results/difficulty_gates.json` | Difficulty gate historical |
| `results/discriminability.json` | Factor discriminability |
| `results/embeddings_contrast.json` | Hash vs MiniLM contrast |
| `results/side_by_side.md`, `.json` | v1/v2 comparison narrative data |

## Human docs

| Doc | Path |
| --- | --- |
| Status (live) | `docs/status.md` |
| Tradeoffs | `docs/tradeoffs.md` |
| Architecture | `docs/architecture.md` |
| Evaluation protocol | `docs/evaluation-protocol.md` |
| Reproduction | `docs/reproduction.md` |
| Datasets | `docs/datasets.md` |
| Discriminability notes | `docs/discriminability.md` |

## How to verify quickly

```bash
pip install -e ".[dev,semantic]"
python -m rag_bench.run_all
pytest -q
python -c "import json; m=json.load(open('results/regression_metrics.json')); assert m['recall_hits']==35 and m['attribution_hits']==35"
python -c "import json; h=json.load(open('results/holdout_confirmation.json')); assert h['pass'] is True"
python scripts/sanitize_public_tree.py --scan-only
```
