# Design tradeoffs (v2 production-hardening)

Selection is **pre-registered** (`config/selection_rules.yaml`) with dev-only shortlist
and holdout confirm-only (no holdout re-rank). shortlist_size=1.

## Frozen arms

See `config/arms.yaml` (exactly 10). Baseline arm: `hash_dense_k8_r1`.

## Selection outcome (stable summary)

Full arm metrics live under `results/arms/` and `results/dev_shortlist.json`.
This section intentionally omits wall-clock and embedding float dumps so
`python -m rag_bench.run_all` leaves a clean tracked tree.

```json
{
  "baseline_arm_id": "hash_dense_k8_r1",
  "bootstrap": {
    "B": 1000,
    "ci_high": 0.30303030303030304,
    "ci_low": 0.030303030303030304,
    "point_delta": 0.15151515151515152,
    "seed": 42
  },
  "holdout_pass": true,
  "primary_arm_id": "recursive_minilm_k8_r1",
  "re_ranked_on_holdout": false,
  "selection_rules_sha256": "ab76cb29861f9310746ecb261a02c258499a249ef04693daca94d16b2c57b04b",
  "shortlist_size": 1,
  "winner_id": "recursive_minilm_k8_r1"
}
```

## History

- `results/run_v1/`, `results/run_v2/` preserved as historical archives.
- `data/regression_v1/` is the immutable 35-label regression freeze.

## Recompute

```bash
python -m rag_bench.run_all
```
