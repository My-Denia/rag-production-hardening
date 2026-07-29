#!/usr/bin/env python3
"""Sanitize public release tree: redaction, exclusive results prune, security scan.

Run from repo root:
  python scripts/sanitize_public_tree.py [--prune] [--scan-only]
  python scripts/sanitize_public_tree.py --prune --write-reproduction --write-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF_REL = "scripts/sanitize_public_tree.py"

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".toml",
    ".cfg",
    ".ini",
    ".csv",
    ".tsv",
    ".cff",
    ".sh",
    ".ps1",
    ".rst",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".lock",
    ".sha256",
}

# Exclusive results whitelist (plan rev4). Paths relative to results/.
RESULTS_WHITELIST = {
    "regression_metrics.json",
    "regression_rca.json",
    "holdout_confirmation.json",
    "quality_report.json",
    "recovery_report.json",
    "concurrency_report.json",
    "determinism_check.json",
    "freeze_report.json",
    "dev_shortlist.json",
    "hybrid_vs_dense.json",
    "embedding_backend.json",
    "m0_selection_rules.sha256",
    "metrics.json",
    "side_by_side.md",
    "side_by_side.json",
    "difficulty_gates.json",
    "discriminability.json",
    "embeddings_contrast.json",
    "run_v1/metrics.json",
    "run_v1/selected.yaml",
    "run_v2/metrics.json",
    "run_v2/selected.yaml",
}

RESULTS_WHITELIST_DIRS = {
    "arms",
    "release",
}

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".cache",
    "node_modules",
}


def _frag(*parts: str) -> str:
    """Join fragments so source does not contain full forbidden literals continuously."""
    return "".join(parts)


# Forbidden path fragments assembled at runtime (avoid self-hits in this file).
_DRIVE_USERS = _frag("C:", "\\", "Users", "\\")
_DRIVE_USERS_FWD = _frag("C:", "/", "Users", "/")
_PRIV_RAG_BS = _frag("C:", "\\", "Files", "\\", "rag", "\\")
_PRIV_RAG_FWD = _frag("C:", "/", "Files", "/", "rag", "/")
_PRIV_RAG_BARE_BS = _frag("C:", "\\", "Files", "\\", "rag")
_PRIV_RAG_BARE_FWD = _frag("C:", "/", "Files", "/", "rag")
_PUB_BS = _frag("C:", "\\", "Files", "\\", "public-rag-release")
_PUB_FWD = _frag("C:", "/", "Files", "/", "public-rag-release")
_LOCAL_UID = _frag("25", "725")  # path-segment username token from private host
_GHP = _frag("ghp", "_")
_GH_PAT = _frag("github", "_", "pat", "_")
_SK = _frag("sk", "-")
_OPENSSH = _frag("BEGIN ", "OPENSSH ", "PRIVATE ", "KEY")
_RSA = _frag("BEGIN ", "RSA ", "PRIVATE ", "KEY")
_GOAL = _frag("goal", "-", "runs")


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if path.name in {".gitignore", "LICENSE", "Makefile", "Dockerfile"}:
        return True
    if path.name.endswith(".sha256"):
        return True
    return False


def should_skip_dir(name: str) -> bool:
    if name in SKIP_DIR_NAMES:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def redact_text(text: str) -> str:
    """Apply field-level and path redactions to free text."""
    # Absolute user home paths
    text = re.sub(
        r"[A-Za-z]:\\Users\\[^\\\s\"']+(?:\\[^\\\"'\s,]*)*",
        "<REDACTED_USER_PATH>",
        text,
    )
    text = re.sub(
        r"[A-Za-z]:/Users/[^/\s\"']+(?:/[^/\"'\s,]*)*",
        "<REDACTED_USER_PATH>",
        text,
    )
    # Temp under user profile
    text = re.sub(
        r"[A-Za-z]:\\Users\\[^\\]+\\AppData\\Local\\Temp\\[^\s\"']*",
        "<REDACTED_TEMP_DIR>",
        text,
        flags=re.IGNORECASE,
    )

    # Private internal root (any drive letter)
    text = re.sub(
        r"[A-Za-z]:\\Files\\rag\\?",
        "<REPO_ROOT>/",
        text,
    )
    text = re.sub(
        r"[A-Za-z]:/Files/rag/?",
        "<REPO_ROOT>/",
        text,
    )

    # Public mirror absolute paths
    text = re.sub(
        r"[A-Za-z]:\\Files\\public-rag-release\\?",
        "<REPO_ROOT>/",
        text,
    )
    text = re.sub(
        r"[A-Za-z]:/Files/public-rag-release/?",
        "<REPO_ROOT>/",
        text,
    )

    # Username path segment (private host) — built from fragments
    uid = re.escape(_LOCAL_UID)
    text = re.sub(rf"(?<=[/\\]){uid}(?=[/\\])", "<REDACTED_USER>", text)
    text = re.sub(rf"(?<=[/\\]){uid}(?=[\"'\s,]|$)", "<REDACTED_USER>", text)

    # Agent harness paths (fragment-built so this file is not a scan self-hit)
    _gr = re.escape(_GOAL)
    text = re.sub(
        rf"[A-Za-z]:\\Users\\[^\\]+\\.grok\\{_gr}\\[^\s\"']*",
        "<REDACTED_AGENT_HARNESS_PATH>",
        text,
    )
    text = re.sub(
        rf"[^\s\"']*[/\\]{_gr}[/\\][^\s\"']*",
        "<REDACTED_AGENT_HARNESS_PATH>",
        text,
    )
    text = re.sub(
        r"[A-Za-z]:\\Users\\[^\\]+\\.grok\\[^\s\"']*",
        "<REDACTED_AGENT_HARNESS_PATH>",
        text,
    )

    # Common private clone instruction
    text = re.sub(
        r"cd\s+[A-Za-z]:\\Files\\rag\b",
        "cd <clone-dir>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"cd\s+[A-Za-z]:/Files/rag\b",
        "cd <clone-dir>",
        text,
        flags=re.IGNORECASE,
    )

    return text


def redact_json_obj(obj: Any) -> Any:
    """Deep-walk JSON: redact work_dir fields and string values."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "work_dir" and isinstance(v, str):
                out[k] = "<REDACTED_TEMP_DIR>"
            elif k == "install_commands" and isinstance(v, list):
                out[k] = ["python -m pip install sentence-transformers"]
            elif isinstance(v, str):
                out[k] = redact_text(v)
            else:
                out[k] = redact_json_obj(v)
        return out
    if isinstance(obj, list):
        return [redact_json_obj(x) for x in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


def sanitize_file(path: Path) -> bool:
    """Sanitize one text file in place. Returns True if modified."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False

    original = raw
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
            data = redact_json_obj(data)
            new = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            if new != original:
                path.write_text(new, encoding="utf-8", newline="\n")
                return True
            return False
        except json.JSONDecodeError:
            pass

    new = redact_text(raw)
    if new != original:
        path.write_text(new, encoding="utf-8", newline="\n")
        return True
    return False


def iter_text_files(root: Path, *, include_self: bool = False):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(should_skip_dir(part) for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        if not include_self and rel == SELF_REL:
            continue
        if is_text_file(p):
            yield p


def sanitize_tree(root: Path) -> dict[str, Any]:
    changed: list[str] = []
    scanned = 0
    # Never rewrite this script (patterns must remain operational)
    for p in iter_text_files(root, include_self=False):
        scanned += 1
        if sanitize_file(p):
            changed.append(str(p.relative_to(root)).replace("\\", "/"))
    return {"scanned_files": scanned, "modified_files": changed, "n_modified": len(changed)}


def prune_results(root: Path) -> dict[str, Any]:
    results = root / "results"
    if not results.exists():
        return {"pruned": [], "kept": [], "note": "no results dir"}

    kept: list[str] = []
    pruned: list[str] = []

    for p in sorted(results.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(results).as_posix()
        top = rel.split("/")[0] if "/" in rel else rel
        if top in RESULTS_WHITELIST_DIRS:
            kept.append(rel)
            continue
        if rel in RESULTS_WHITELIST:
            kept.append(rel)
            continue
        p.unlink()
        pruned.append(rel)

    for d in sorted(results.rglob("*"), reverse=True):
        if d.is_dir() and d != results:
            try:
                next(d.iterdir())
            except StopIteration:
                rel = d.relative_to(results).as_posix()
                if rel in RESULTS_WHITELIST_DIRS:
                    continue
                try:
                    d.rmdir()
                except OSError:
                    pass

    return {"kept": kept, "pruned": pruned, "n_kept": len(kept), "n_pruned": len(pruned)}


def _build_scan_patterns() -> list[tuple[str, re.Pattern[str], str]]:
    """Return (name, pattern, kind) with kinds path|secret."""
    uid = re.escape(_LOCAL_UID)
    return [
        (
            "users_path",
            re.compile(
                re.escape(_DRIVE_USERS)
                + r"|"
                + re.escape(_DRIVE_USERS_FWD)
                + r"|[A-Za-z]:\\Users\\|[A-Za-z]:/Users/"
            ),
            "path",
        ),
        (
            "appdata_temp",
            re.compile(
                r"[A-Za-z]:\\Users\\[^\\]+\\AppData\\Local\\Temp\\",
                re.I,
            ),
            "path",
        ),
        (
            "private_rag",
            re.compile(
                re.escape(_PRIV_RAG_BS)
                + r"|"
                + re.escape(_PRIV_RAG_FWD)
                + r"|[A-Za-z]:\\Files\\rag\\|[A-Za-z]:/Files/rag/"
            ),
            "path",
        ),
        (
            "private_rag_bare",
            re.compile(
                re.escape(_PRIV_RAG_BARE_BS)
                + r"(?![A-Za-z0-9_-])|"
                + re.escape(_PRIV_RAG_BARE_FWD)
                + r"(?![A-Za-z0-9_-])|"
                + r"[A-Za-z]:\\Files\\rag(?![A-Za-z0-9_-])|"
                + r"[A-Za-z]:/Files/rag(?![A-Za-z0-9_-])"
            ),
            "path",
        ),
        (
            "public_abs",
            re.compile(
                re.escape(_PUB_BS)
                + r"|"
                + re.escape(_PUB_FWD)
                + r"|[A-Za-z]:\\Files\\public-rag-release"
            ),
            "path",
        ),
        (
            "user_uid",
            re.compile(rf"[/\\]{uid}[/\\]|[/\\]{uid}[\"'\s,]"),
            "path",
        ),
        (
            "ghp_token",
            re.compile(re.escape(_GHP) + r"[A-Za-z0-9]{20,}"),
            "secret",
        ),
        (
            "github_pat",
            re.compile(re.escape(_GH_PAT)),
            "secret",
        ),
        (
            "sk_token",
            re.compile(re.escape(_SK) + r"[A-Za-z0-9]{20,}"),
            "secret",
        ),
        (
            "openssh_key",
            re.compile(re.escape(_OPENSSH)),
            "secret",
        ),
        (
            "rsa_key",
            re.compile(re.escape(_RSA)),
            "secret",
        ),
        (
            "goal_runs",
            re.compile(r"[/\\]" + re.escape(_GOAL) + r"[/\\]|" + re.escape(_GOAL) + r"[/\\]"),
            "path",
        ),
    ]


# Files that document scan rules; skip self-hit noise for pattern inventory only.
_SCAN_SKIP_RELS = {
    SELF_REL,
    "src/rag_bench/verify_public_evidence.py",
}


def security_scan(root: Path) -> dict[str, Any]:
    patterns = _build_scan_patterns()
    hits: list[dict[str, Any]] = []
    commands: list[str] = []
    files_scanned = 0

    for p in iter_text_files(root, include_self=True):
        rel = p.relative_to(root).as_posix()
        # Always skip scanning this script body (pattern assembly only)
        if rel in _SCAN_SKIP_RELS:
            continue
        # Skip prior scan report content (rewritten at end)
        if rel == "results/release/security-scan.json":
            continue
        files_scanned += 1
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pat, kind in patterns:
            for m in pat.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                snippet = text[max(0, m.start() - 40) : m.end() + 40].replace("\n", " ")
                hits.append(
                    {
                        "pattern": name,
                        "file": rel,
                        "line": line_no,
                        "match": m.group(0)[:120],
                        "snippet": snippet[:200],
                        "kind": kind,
                    }
                )

    git_dir = root / ".git"
    if git_dir.exists():
        import subprocess

        try:
            proc = subprocess.run(
                ["git", "-C", str(root), "log", "-p", "--all"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            commands.append("git log -p --all")
            log_text = proc.stdout or ""
            for name, pat, kind in patterns:
                for m in pat.finditer(log_text):
                    # Ignore hits only inside this script path in history if needed
                    hits.append(
                        {
                            "pattern": name,
                            "file": "<git-log>",
                            "line": log_text.count("\n", 0, m.start()) + 1,
                            "match": m.group(0)[:120],
                            "snippet": log_text[
                                max(0, m.start() - 40) : m.end() + 40
                            ].replace("\n", " ")[:200],
                            "kind": kind,
                        }
                    )
        except Exception as e:
            commands.append(f"git log -p --all FAILED: {e}")

    path_hits = [h for h in hits if h["kind"] == "path"]
    secret_hits = [h for h in hits if h["kind"] == "secret"]

    return {
        "schema": 1,
        "high_confidence_secrets": len(secret_hits),
        "path_hits": len(path_hits),
        "files_scanned": files_scanned,
        "commands": commands
        + [
            "python scripts/sanitize_public_tree.py --scan-only",
            "scan root=<REPO_ROOT>",
        ],
        "secret_hits": secret_hits,
        "path_hit_details": path_hits,
        "pass": len(secret_hits) == 0 and len(path_hits) == 0,
    }


def write_reproduction(root: Path) -> Path:
    """Write schema-2 reproduction metadata (no public_commit / no self-ref SHA).

    Final release commit is recorded only in the post-tag Release asset
    ``release-attestation-vX.Y.Z.json`` (see docs/release-attestation.md).
    """
    release = root / "results" / "release"
    release.mkdir(parents=True, exist_ok=True)

    def loadj(rel: str) -> Any:
        p = root / rel
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    reg = loadj("results/regression_metrics.json") or {}
    hold = loadj("results/holdout_confirmation.json") or {}
    conc = loadj("results/concurrency_report.json") or {}
    rec = loadj("results/recovery_report.json") or {}
    det = loadj("results/determinism_check.json") or {}
    freeze = loadj("results/freeze_report.json") or {}
    emb = loadj("results/embedding_backend.json") or {}

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    ver_m = re.search(r'version\s*=\s*"([^"]+)"', pyproject)
    version = ver_m.group(1) if ver_m else "unknown"
    tag_name = f"v{version}"
    attestation_asset = f"release-attestation-{tag_name}.json"

    primary = hold.get("holdout_primary_metrics") or {}
    boot = hold.get("bootstrap") or {}

    payload = {
        "schema": 2,
        "package": "rag-bench",
        "version": version,
        "recompute_command": "python -m rag_bench.run_all",
        "install": 'pip install -e ".[dev,semantic]"',
        "python_requires": ">=3.11",
        "commit_binding": {
            "type": "release-attestation",
            "asset": attestation_asset,
            "reason": (
                "The final release commit is recorded after tag creation in the "
                "GitHub Release attestation asset to avoid a tracked-file "
                "self-reference loop. Tracked reproduction.json describes the "
                "recompute protocol and gate snapshot only."
            ),
        },
        "gates": {
            "regression_recall_hits": reg.get("recall_hits"),
            "regression_attribution_hits": reg.get("attribution_hits"),
            "dual_35_35": (
                int(reg.get("recall_hits") or 0) >= 35
                and int(reg.get("attribution_hits") or 0) >= 35
            ),
            "holdout_pass": hold.get("pass"),
            "holdout_winner": hold.get("winner_id"),
            "holdout_recall_at_k": primary.get("recall_at_k"),
            "holdout_attribution_rate": primary.get("attribution_rate"),
            "re_ranked_on_holdout": hold.get("re_ranked_on_holdout"),
            "bootstrap_point_delta": boot.get("point_delta"),
            "bootstrap_ci_low": boot.get("ci_low"),
            "bootstrap_ci_high": boot.get("ci_high"),
            "concurrency_ok": conc.get("ok"),
            "concurrency_n_ok": conc.get("n_ok"),
            "concurrency_cross_talk": conc.get("cross_talk"),
            "recovery_ok": rec.get("ok"),
            "determinism_ok": det.get("ok") if "ok" in det else det.get("identical"),
            "freeze_ok": freeze.get("ok"),
            "minilm_success": emb.get("minilm_success"),
        },
        "artifacts": {
            "regression_metrics": "results/regression_metrics.json",
            "holdout_confirmation": "results/holdout_confirmation.json",
            "quality_report": "results/quality_report.json",
            "recovery_report": "results/recovery_report.json",
            "concurrency_report": "results/concurrency_report.json",
            "freeze_report": "results/freeze_report.json",
            "security_scan": "results/release/security-scan.json",
        },
    }
    # Schema 2 forbids ambiguous commit fields
    assert "public_commit" not in payload
    out = release / "reproduction.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def write_manifest(root: Path) -> Path:
    release = root / "results" / "release"
    release.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for p in sorted(iter_text_files(root, include_self=True)):
        rel = p.relative_to(root).as_posix()
        if rel == "results/release/public-manifest.sha256":
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {rel}")
    out = release / "public-manifest.sha256"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sanitize public rag-bench tree")
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--no-sanitize", action="store_true")
    ap.add_argument("--write-reproduction", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    if not (root / "src" / "rag_bench").exists():
        print(f"ERROR: not a rag-bench root: {root}", file=sys.stderr)
        return 2

    if args.prune:
        pr = prune_results(root)
        print(f"prune: kept={pr.get('n_kept')} pruned={pr.get('n_pruned')}")

    if not args.scan_only and not args.no_sanitize:
        san = sanitize_tree(root)
        print(f"sanitize: scanned={san['scanned_files']} modified={san['n_modified']}")
        for f in san["modified_files"][:30]:
            print(f"  - {f}")

    if args.write_reproduction:
        p = write_reproduction(root)
        print(f"reproduction -> {p}")

    if args.write_manifest:
        p = write_manifest(root)
        print(f"manifest -> {p}")

    scan = security_scan(root)
    release = root / "results" / "release"
    release.mkdir(parents=True, exist_ok=True)
    scan_path = release / "security-scan.json"
    # Write scan without embedding raw secret material beyond counts when pass
    out_scan = {
        "schema": scan["schema"],
        "high_confidence_secrets": scan["high_confidence_secrets"],
        "path_hits": scan["path_hits"],
        "files_scanned": scan["files_scanned"],
        "commands": scan["commands"],
        "secret_hits": scan["secret_hits"] if not scan["pass"] else [],
        "path_hit_details": scan["path_hit_details"] if not scan["pass"] else [],
        "pass": scan["pass"],
    }
    # --scan-only must not dirty the working tree (full-recompute git diff gate).
    if not args.scan_only:
        scan_path.write_text(json.dumps(out_scan, indent=2) + "\n", encoding="utf-8")
        dest = str(scan_path)
    else:
        dest = "(not written; --scan-only)"
    print(
        f"scan: path_hits={scan['path_hits']} secrets={scan['high_confidence_secrets']} "
        f"pass={scan['pass']} -> {dest}"
    )
    if not scan["pass"]:
        for h in (scan.get("path_hit_details") or [])[:20]:
            print(f"  PATH {h['file']}:{h['line']} [{h['pattern']}] {h['match']!r}")
        for h in (scan.get("secret_hits") or [])[:20]:
            print(f"  SECRET {h['file']}:{h['line']} [{h['pattern']}] {h['match']!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
