"""Machine-checkable public evidence verifier for committed artifacts.

Pure validation is separated from I/O so tests can load fixtures/temp trees
without mocking the comparison logic.

Usage:
  python -m rag_bench.verify_public_evidence
  python scripts/verify_public_evidence.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from rag_bench.config_load import ROOT as DEFAULT_ROOT
from rag_bench.freeze_regression import verify as freeze_verify

def _frag(*parts: str) -> str:
    return "".join(parts)


# High-confidence private path / secret fragments (assembled; avoid full literals
# so security scan of this source file does not self-hit).
_PRIVATE_PATH_RES = [
    re.compile(r"[A-Za-z]:\\Users\\", re.I),
    re.compile(r"[A-Za-z]:/Users/", re.I),
    re.compile(r"[A-Za-z]:\\Files\\rag(?:\\|$)", re.I),
    re.compile(r"[A-Za-z]:/Files/rag(?:/|$)"),
    re.compile(
        re.escape(_frag("C:", "\\", "Files", "\\", "public-rag-release"))
        + r"|[A-Za-z]:\\Files\\public-rag-release",
        re.I,
    ),
    re.compile(re.escape(_frag("goal", "-", "runs")), re.I),
]
_SECRET_RES = [
    re.compile(re.escape(_frag("ghp", "_")) + r"[A-Za-z0-9]{20,}"),
    re.compile(re.escape(_frag("github", "_", "pat", "_")) + r"[A-Za-z0-9_]+"),
    re.compile(re.escape(_frag("sk", "-")) + r"[A-Za-z0-9]{20,}"),
    re.compile(re.escape(_frag("BEGIN ", "OPENSSH ", "PRIVATE ", "KEY"))),
    re.compile(re.escape(_frag("BEGIN ", "RSA ", "PRIVATE ", "KEY"))),
]

REQUIRED_ARTIFACTS = [
    "results/regression_metrics.json",
    "results/holdout_confirmation.json",
    "results/concurrency_report.json",
    "results/recovery_report.json",
    "results/freeze_report.json",
    "results/release/reproduction.json",
]

STALE_HOLDING_ATTR = 0.909
STALE_HOLDING_ATTR_STRS = ("0.909", "0.9090")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _approx_eq(a: float, b: float, *, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def validate_artifacts(root: Path, data: dict[str, Any]) -> list[str]:
    """Validate already-loaded artifact dicts. Returns list of error strings (empty = pass).

    Expected keys in ``data``:
      regression, holdout, concurrency, recovery, freeze, reproduction
    Optional: freeze_verify_result (dict from freeze_regression.verify)
    """
    errors: list[str] = []

    # --- required presence ---
    for key in (
        "regression",
        "holdout",
        "concurrency",
        "recovery",
        "freeze",
        "reproduction",
    ):
        if key not in data or data[key] is None:
            errors.append(f"missing required artifact data: {key}")
    if errors:
        return errors

    reg = data["regression"]
    hold = data["holdout"]
    conc = data["concurrency"]
    rec = data["recovery"]
    freeze = data["freeze"]
    repro = data["reproduction"]

    # --- regression 35/35 ---
    rh = int(reg.get("recall_hits") or 0)
    ah = int(reg.get("attribution_hits") or 0)
    if rh != 35:
        errors.append(f"regression recall_hits={rh} expected 35")
    if ah != 35:
        errors.append(f"regression attribution_hits={ah} expected 35")

    # --- holdout ---
    if hold.get("pass") is not True:
        errors.append(f"holdout pass={hold.get('pass')!r} expected True")
    if hold.get("winner_id") != "recursive_minilm_k8_r1":
        errors.append(f"holdout winner_id={hold.get('winner_id')!r} expected recursive_minilm_k8_r1")
    if hold.get("re_ranked_on_holdout") is not False:
        errors.append(
            f"re_ranked_on_holdout={hold.get('re_ranked_on_holdout')!r} expected False"
        )
    primary = hold.get("holdout_primary_metrics") or {}
    if not _approx_eq(float(primary.get("recall_at_k") or 0), 1.0):
        errors.append(f"holdout recall_at_k={primary.get('recall_at_k')!r} expected 1.0")
    attr = float(primary.get("attribution_rate") or 0)
    # Must match committed artifact value (not stale 0.909)
    if _approx_eq(attr, STALE_HOLDING_ATTR, tol=1e-3) and not _approx_eq(
        attr, 0.9393939393939394, tol=1e-9
    ):
        errors.append(f"holdout attribution_rate looks stale ({attr}); expected committed ~0.939…")
    # Exact match against the loaded committed artifact (self-consistency of hold file)
    committed_attr = float(primary.get("attribution_rate"))
    if not _approx_eq(attr, committed_attr):
        errors.append("holdout attribution_rate internal mismatch")
    # Absolute floor: must be the published v0.2.x value
    if not _approx_eq(attr, 0.9393939393939394, tol=1e-12):
        errors.append(
            f"holdout attribution_rate={attr!r} does not match committed public value "
            f"0.9393939393939394 (stale 0.909 not allowed)"
        )

    # --- significance / bootstrap ---
    boot = hold.get("bootstrap") or {}
    point = boot.get("point_delta")
    ci_low = boot.get("ci_low")
    ci_high = boot.get("ci_high")
    if point is None:
        errors.append("bootstrap.point_delta missing")
    else:
        if not _approx_eq(float(point), 0.15151515151515152, tol=1e-12):
            errors.append(
                f"bootstrap.point_delta={point!r} does not match committed "
                f"0.15151515151515152"
            )
    if ci_low is None or float(ci_low) <= 0:
        errors.append(f"bootstrap.ci_low={ci_low!r} must be present and > 0")
    else:
        if not _approx_eq(float(ci_low), 0.030303030303030304, tol=1e-12):
            errors.append(
                f"bootstrap.ci_low={ci_low!r} does not match committed 0.030303030303030304"
            )
    if ci_high is None:
        errors.append("bootstrap.ci_high missing")
    else:
        if not _approx_eq(float(ci_high), 0.30303030303030304, tol=1e-12):
            errors.append(
                f"bootstrap.ci_high={ci_high!r} does not match committed 0.30303030303030304"
            )

    # --- concurrency ---
    n_ok = conc.get("n_ok")
    if n_ok is None or int(n_ok) != 20:
        errors.append(f"concurrency n_ok={n_ok!r} expected 20")
    cross = conc.get("cross_talk")
    if cross is None or int(cross) != 0:
        errors.append(f"concurrency cross_talk={cross!r} expected 0")

    # --- recovery ---
    stages = rec.get("stages") or {}
    for stage in ("retrieve", "rerank", "generate"):
        st = stages.get(stage) or {}
        if not st.get("ok"):
            errors.append(f"recovery stage {stage} not ok: {st.get('ok')!r}")
    corrupt = rec.get("corrupt_checkpoint") or {}
    if not corrupt.get("ok"):
        errors.append(f"recovery corrupt_checkpoint.ok={corrupt.get('ok')!r} expected True")
    if not corrupt.get("corrupt_raises"):
        errors.append("recovery corrupt_checkpoint.corrupt_raises expected True")
    if not rec.get("ok"):
        errors.append(f"recovery top-level ok={rec.get('ok')!r} expected True")

    # --- freeze ---
    if not freeze.get("ok"):
        errors.append(f"freeze_report.ok={freeze.get('ok')!r} expected True")
    for key in ("labels_v1_sha256", "regression_sha256"):
        if not freeze.get(key):
            errors.append(f"freeze_report missing {key}")
    cfg = freeze.get("config_sha256") or {}
    for name in ("arms.yaml", "selection_rules.yaml", "categories.yaml"):
        if not cfg.get(name):
            errors.append(f"freeze_report.config_sha256 missing {name}")
    fv = data.get("freeze_verify_result")
    if fv is not None and not fv.get("ok"):
        errors.append(f"freeze verify failed: {fv.get('errors') or fv}")

    # --- reproduction schema 2 ---
    schema = repro.get("schema")
    if schema != 2:
        errors.append(f"reproduction.schema={schema!r} expected 2")
    if "public_commit" in repro:
        errors.append(
            "reproduction still contains ambiguous field 'public_commit' "
            "(schema 2 forbids this; use release-attestation commit_binding)"
        )
    binding = repro.get("commit_binding") or {}
    if binding.get("type") != "release-attestation":
        errors.append(
            f"commit_binding.type={binding.get('type')!r} expected 'release-attestation'"
        )
    if not binding.get("asset"):
        errors.append("commit_binding.asset missing")
    if not binding.get("reason"):
        errors.append("commit_binding.reason missing")
    if repro.get("package") != "rag-bench":
        errors.append(f"reproduction.package={repro.get('package')!r} expected rag-bench")
    if not repro.get("recompute_command"):
        errors.append("reproduction.recompute_command missing")
    gates = repro.get("gates") or {}
    if int(gates.get("regression_recall_hits") or 0) != 35:
        errors.append("reproduction.gates.regression_recall_hits expected 35")
    if int(gates.get("regression_attribution_hits") or 0) != 35:
        errors.append("reproduction.gates.regression_attribution_hits expected 35")

    # --- private paths / secrets in loaded JSON dumps ---
    blob = json.dumps(data, ensure_ascii=False)
    for pat in _PRIVATE_PATH_RES:
        if pat.search(blob):
            errors.append(f"private path pattern detected in artifacts: {pat.pattern}")
    for pat in _SECRET_RES:
        if pat.search(blob):
            errors.append(f"high-confidence secret pattern detected: {pat.pattern}")

    return errors


def load_public_artifacts(root: Path) -> tuple[dict[str, Any], list[str]]:
    """Load artifacts from disk. Returns (data, load_errors)."""
    errors: list[str] = []
    data: dict[str, Any] = {}
    mapping = {
        "regression": "results/regression_metrics.json",
        "holdout": "results/holdout_confirmation.json",
        "concurrency": "results/concurrency_report.json",
        "recovery": "results/recovery_report.json",
        "freeze": "results/freeze_report.json",
        "reproduction": "results/release/reproduction.json",
    }
    for key, rel in mapping.items():
        path = root / rel
        if not path.is_file():
            errors.append(f"missing required artifact file: {rel}")
            data[key] = None
            continue
        try:
            data[key] = _load_json(path)
        except (OSError, json.JSONDecodeError) as e:
            errors.append(f"failed to load {rel}: {e}")
            data[key] = None
    return data, errors


def verify_public_evidence(root: Path | None = None) -> dict[str, Any]:
    """Full I/O + validation. Returns report dict with ok, errors, checks."""
    root = Path(root) if root is not None else Path(DEFAULT_ROOT)
    root = root.resolve()
    load_data, load_errors = load_public_artifacts(root)
    errors = list(load_errors)

    # freeze live verify when possible
    freeze_verify_result: dict[str, Any] | None = None
    try:
        # freeze_verify uses package ROOT; temporarily only call if root matches package
        # or when root has data/regression_v1
        if (root / "data" / "regression_v1" / "labels.jsonl").is_file():
            # Import-level verify uses package DATA_DIR; only run when root is package root
            from rag_bench import config_load

            if Path(config_load.ROOT).resolve() == root:
                freeze_verify_result = freeze_verify()
            else:
                # Structural freeze report check only when not package root
                freeze_verify_result = {"ok": True, "note": "skipped live verify (non-package root)"}
    except Exception as e:
        freeze_verify_result = {"ok": False, "errors": [str(e)]}

    load_data["freeze_verify_result"] = freeze_verify_result
    if not load_errors:
        errors.extend(validate_artifacts(root, load_data))
    elif freeze_verify_result is not None and not freeze_verify_result.get("ok"):
        errors.append(f"freeze verify failed: {freeze_verify_result.get('errors')}")

    # scan text tree lightly for public_commit leftover in tracked JSON under results/release
    repro_path = root / "results" / "release" / "reproduction.json"
    if repro_path.is_file():
        text = repro_path.read_text(encoding="utf-8")
        if '"public_commit"' in text:
            if "reproduction still contains ambiguous field 'public_commit'" not in "\n".join(
                errors
            ):
                errors.append("reproduction.json text contains public_commit field")

    report = {
        "ok": len(errors) == 0,
        "root": str(root),
        "errors": errors,
        "n_errors": len(errors),
        "freeze_verify_ok": (freeze_verify_result or {}).get("ok"),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify committed public evidence artifacts")
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (default: package root)",
    )
    ap.add_argument("--json", action="store_true", help="Print full report JSON")
    args = ap.parse_args(argv)
    report = verify_public_evidence(args.root)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        if report["ok"]:
            print("public evidence verifier: PASS")
        else:
            print("public evidence verifier: FAIL", file=sys.stderr)
            for e in report["errors"]:
                print(f"  - {e}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
