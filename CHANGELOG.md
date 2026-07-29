# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-07-29

### Added

- Public open-source packaging: MIT `LICENSE`, `DATA_CARD.md`, `THIRD_PARTY_NOTICES.md`
- Public documentation set: architecture, evaluation protocol, reproduction, datasets, public evidence index, preflight
- `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff`
- `scripts/sanitize_public_tree.py` — path/secret redaction, exclusive results whitelist prune, security scan, reproduction + manifest
- GitHub Actions CI (Windows, Python 3.11 and 3.12): install, pytest, freeze verify
- `results/release/` packaging artifacts (security-scan, reproduction, manifest)

### Changed

- Package version **0.2.0**
- README rewritten for public clone paths (no private machine roots)
- Results tree limited to exclusive evidence whitelist for release

### Security

- Redact `work_dir` and absolute user/private paths from published JSON/docs
- Machine AC2 scan gate: `path_hits=0`, `high_confidence_secrets=0`

### Notes

- Functional pipeline continues production-hardening v2 behavior (dual regression 35/35 target, holdout confirm-only).
- Gold labels under `data/regression_v1/` unchanged by packaging.

## [0.1.0] — prior internal

Internal evaluable RAG bench baseline (not published as this public tree).
