# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.2.x | Yes (current public release line) |
| < 0.2 | Best-effort only |

This repository is an evaluation bench, not a networked multi-tenant service. Still, treat dependency and secret hygiene seriously.

## Reporting a vulnerability

If you discover a security issue (e.g. accidental credential in history, unsafe deserialization, dependency RCE):

1. **Do not** open a public GitHub issue with exploit details or secrets.
2. Contact the repository owner via GitHub Security Advisories (preferred) or a private channel listed on the owner profile.
3. Include: affected commit/tag, impact, minimal reproduction without live secrets.

## Secrets and path hygiene

Maintainers must ensure public trees pass:

```bash
python scripts/sanitize_public_tree.py --scan-only
```

Fail on high-confidence secrets or absolute private paths (`results/release/security-scan.json`).

## Scope notes

- Synthetic data only; no real customer documents intended.
- Optional MiniLM download is a third-party model; review upstream model terms.
- Process-kill recovery tests use local temp dirs — never commit raw temp paths.
