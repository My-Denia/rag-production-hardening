# Contributing

Thanks for interest in improving `rag-bench`.

## Ground rules

1. **Do not edit frozen gold** under `data/regression_v1/` to chase metrics.
2. **Do not re-rank on holdout** or add arms only to pass holdout.
3. Prefer external-judge metrics; no silent metric redefinition.
4. Keep synthetic corpus free of real PII and secrets.
5. Public PRs must not introduce absolute local user-profile paths or credentials.

## Dev setup

```bash
python -m venv .venv
# activate venv for your shell
pip install -U pip
pip install -e ".[dev,semantic]"
pytest -q
```

Optional full recompute (slow):

```bash
python -m rag_bench.run_all
```

## Before opening a PR

- [ ] `pytest -q` green
- [ ] No secrets or machine paths (`python scripts/sanitize_public_tree.py --scan-only`)
- [ ] Docs updated if protocol/metrics change
- [ ] CHANGELOG entry for user-visible changes

## Code style

- Python 3.11+; match existing module layout under `src/rag_bench/`
- Keep changes minimal and test-backed
- Avoid committing `.venv/`, caches, `*.db`, raw traces with local paths

## Security reports

See [SECURITY.md](SECURITY.md). Do not file public issues for secrets.
