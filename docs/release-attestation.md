# Release attestation vs tracked evidence

## Problem

A tracked file such as `results/release/reproduction.json` must not store the
"current release commit SHA". Editing that field creates a new commit, so the
SHA is immediately stale (self-reference loop). Schema 1's `public_commit` field
was also easy to misread as "this tag's commit" when it actually pointed at an
older packaging commit.

## Schema 2 design

Tracked `results/release/reproduction.json` uses **schema 2**:

| Section | Role |
| --- | --- |
| `schema: 2` | Explicit format version |
| `package` / `version` | Package identity (matches `pyproject.toml`) |
| `recompute_command` / `install` / `python_requires` | How to recompute |
| `commit_binding` | Points at a **post-tag** Release asset, not a git SHA |
| `gates` | Snapshot of gate values from committed machine artifacts |
| `artifacts` | Paths of source JSON used for gates |

There is **no** `public_commit` (or any equally ambiguous "current release commit"
field) in tracked metadata.

### `commit_binding`

```json
{
  "type": "release-attestation",
  "asset": "release-attestation-v0.2.1.json",
  "reason": "The final release commit is recorded after tag creation..."
}
```

After the `vX.Y.Z` tag and GitHub Release exist, maintainers attach:

- `release-attestation-vX.Y.Z.json` — includes `release_commit` (tag peel SHA),
  CI/full-recompute run IDs, remote-clone gates, metrics, artifact SHA-256s
- `release-attestation-vX.Y.Z.sha256` — checksum of the JSON asset

These files are **Release assets only** (not committed to git).

## Verification chain

1. **Tracked artifacts** — `results/*.json` under the tag tree  
2. **Verifier** — `python -m rag_bench.verify_public_evidence` (or `scripts/verify_public_evidence.py`)  
3. **Fast CI** — pytest + freeze + security scan on every push  
4. **Full recompute workflow** — `.github/workflows/full-recompute.yml` (manual dispatch)  
5. **Remote clean clone** — independent `run_all` evidence JSON  
6. **Release attestation** — post-tag asset binding `release_commit` == tag SHA  

## Compatibility

Schema 1 (`public_commit`) is **not** accepted by the public evidence verifier.
Upgrade by regenerating with:

```bash
python scripts/sanitize_public_tree.py --write-reproduction --write-manifest
```
