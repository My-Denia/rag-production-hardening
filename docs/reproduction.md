# Reproduction guide

## Environment

- OS: Windows (primary; CI) or any OS with Python ≥3.11
- Recommended: Python 3.12
- Network: optional for first MiniLM model download; hash/BM25 paths work offline

## Install

```bash
git clone <public-repo-url> rag-production-hardening
cd rag-production-hardening

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Unix
# source .venv/bin/activate

python -m pip install -U pip
pip install -e ".[dev,semantic]"
```

Omit `semantic` only if you accept MiniLM arm failures / fallbacks (selection evidence in this release expects MiniLM success).

## Full recompute

```bash
python -m rag_bench.run_all
```

This freezes regression fixtures, rebuilds stratified labels, evaluates 10 arms on dev, confirms holdout, scores regression-v1, runs recovery and concurrency probes, and refreshes `docs/status.md` / `docs/tradeoffs.md`.

## Tests

```bash
pytest -q
```

## Expected gates (honest)

| Check | Where | Expect |
| --- | --- | --- |
| Dual 35/35 | `results/regression_metrics.json` | `recall_hits=35`, `attribution_hits=35` |
| Holdout | `results/holdout_confirmation.json` | `"pass": true` |
| Winner recall | holdout primary metrics | `recall_at_k` ≈ 1.0 |
| Concurrency | `results/concurrency_report.json` | `n_ok=20`, `cross_talk=0` |
| Recovery | `results/recovery_report.json` | `"ok": true` |
| Status | `docs/status.md` | `STATUS: OK` after clean recompute |

Machine-readable summary after release packaging:

- `results/release/reproduction.json` (**schema 2** — recompute protocol + gates; no `public_commit`)
- `results/release/public-manifest.sha256`
- `results/release/security-scan.json`

Final release commit SHA is **not** stored in tracked files. It appears in the GitHub Release asset `release-attestation-v0.2.1.json` after tag creation (see `docs/release-attestation.md`).

Verify gates from artifacts:

```bash
python -m rag_bench.verify_public_evidence
```

## Sanitize (maintainers)

After recompute on a machine with absolute paths in JSON:

```bash
python scripts/sanitize_public_tree.py --prune --write-reproduction --write-manifest
```

## Timing

Full `run_all` with MiniLM on a laptop is typically several minutes (arm grid dominates). Recovery uses multi-second stage holds.

## Non-goals for repro

- Matching absolute temp directories or trace IDs
- Bit-identical floating scores across CPU/OS (selection uses thresholds + bootstrap; regression dual 35/35 is hit counts)
- Claiming production SLAs from this synthetic bench
