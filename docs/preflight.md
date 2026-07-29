# Preflight inventory — public release 0.2.0

Machine + human checklist before public remote create/push (M7+).

## Source provenance

| Item | Value |
| --- | --- |
| Internal baseline (read-only) | commit `9f8ff41` on private workspace (not shipped) |
| Public tree | Fresh `git init` under public mirror only — no private `.git` history |
| Package version | `0.2.0` (`pyproject.toml` + `rag_bench.__version__`) |

## INCLUDE roots (must ship)

| Path | Purpose |
| --- | --- |
| `src/rag_bench/` | Library + CLI (`python -m rag_bench.run_all`) |
| `tests/` | pytest suite |
| `config/` | Frozen arms, categories, selection rules, selected |
| `data/` | Corpus + labels including holdout (for recompute) |
| `pyproject.toml` | Packaging |
| `README.md` | Public entry |
| `.gitignore` | Hygiene |
| `LICENSE` | MIT |
| `DATA_CARD.md` | Dataset card |
| `THIRD_PARTY_NOTICES.md` | Dependency inventory |
| `docs/` | Architecture, protocol, evidence, limits |
| `scripts/sanitize_public_tree.py` | Redaction + scan + prune |
| `.github/workflows/ci.yml` | Windows CI |
| `results/` (whitelist only) | Recomputed + historical evidence |

## EXCLUDE always

| Path / pattern | Reason |
| --- | --- |
| `.venv/`, `.git/` (from internal) | Environment / private history |
| `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.cache/`, `*.db` | Build/runtime noise |
| `results/traces/` raw | May embed local workdirs; not on exclusive whitelist |
| `results/recovery.json` | Duplicate of recovery with temp paths |
| Absolute Windows user profile paths | Path leakage |
| Private internal workspace root (Files/rag) | Path leakage |
| Absolute public mirror path before relativizing | Path leakage |
| Credential material (classic GH tokens, fine-grained PATs, OpenAI-style keys, PEM private keys) | Credential risk |
| Agent harness absolute paths | Internal automation |

## Field redaction inventory

| Location | Field / content | Replacement |
| --- | --- | --- |
| `results/**/*.json` | `work_dir` | `"<REDACTED_TEMP_DIR>"` |
| any text | Windows user profile absolute paths | `<REDACTED_USER_PATH>` |
| any text | Private internal workspace root | `<REPO_ROOT>/` |
| any text | Absolute public mirror path | `<REPO_ROOT>/` |
| `embedding_backend.json` | `install_commands` with full python path | `["python -m pip install sentence-transformers"]` |
| README / docs | private `cd` to internal workspace | `cd <clone-dir>` |
| any | private host username as path segment | redacted |
| any | harness run-directory absolute paths | redacted / generic harness wording |

## Exclusive results whitelist

See plan rev4 / `scripts/sanitize_public_tree.py` `RESULTS_WHITELIST` + `arms/` + `release/`.

## AC2 machine scan

```bash
python scripts/sanitize_public_tree.py --scan-only
```

**Pass criteria:** `results/release/security-scan.json` has `path_hits=0` and `high_confidence_secrets=0`.

Also scan `git log -p` of the **public** repo after commits.

## Functional gates (recompute)

| Gate | Evidence | Pass |
| --- | --- | --- |
| Dual regression 35/35 | `results/regression_metrics.json` | `recall_hits=35` and `attribution_hits=35` |
| Holdout | `results/holdout_confirmation.json` | `pass=true`; winner recall_at_k ≈ 1.0 |
| Recovery | `results/recovery_report.json` | `ok=true` (stages + corrupt) |
| Concurrency | `results/concurrency_report.json` | `n_ok=20`, `cross_talk=0` |
| Determinism | `results/determinism_check.json` | identical / ok |
| Tests | `pytest -q` | green |

## Forbidden claims

- Production multi-tenant deploy readiness
- Paid LLM judge equivalence
- Secret/PII-free beyond scan of this tree
- Force-push or rewrite of public history after secret exposure without owner gate

## Owner-only remaining (post-M6)

1. Independent execution audit (M7)
2. `gh repo create` + push main (M8) — **not** done in M0–M6
3. Remote clone reverify (M9)
4. Tag `v0.2.0` + Release (M10)
