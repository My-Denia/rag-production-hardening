"""Freeze immutable regression-v1 from labels_v1.jsonl with SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from rag_bench.config_load import CONFIG_DIR, DATA_DIR, RESULTS_DIR, ensure_dirs

REGRESSION_DIR = DATA_DIR / "regression_v1"
REGRESSION_LABELS = REGRESSION_DIR / "labels.jsonl"
REGRESSION_MANIFEST = REGRESSION_DIR / "manifest.json"
FREEZE_REPORT = RESULTS_DIR / "freeze_report.json"
SOURCE_V1 = DATA_DIR / "labels_v1.jsonl"


def normalize_newlines(data: bytes) -> bytes:
    """Universal-newline normalize to LF for stable content hashes across OSes."""
    # CRLF → LF, then lone CR → LF
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha256_file(path: Path) -> str:
    """Raw byte SHA-256 (no newline normalization). Prefer sha256_text_file for fixtures."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text_file(path: Path) -> str:
    """SHA-256 of file content after normalizing newlines to LF.

    Windows checkouts with core.autocrlf=true may materialize CRLF on disk while
    git stores LF. Freeze manifests record the LF-normalized hash so verify() is
    stable on both LF and CRLF working trees.
    """
    return sha256_bytes(normalize_newlines(path.read_bytes()))


def freeze(force: bool = False) -> dict[str, Any]:
    """Copy labels_v1 → regression_v1 and write manifest. Refuse overwrite unless force."""
    ensure_dirs()
    if not SOURCE_V1.exists():
        raise FileNotFoundError(f"missing {SOURCE_V1}")
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

    src_hash = sha256_text_file(SOURCE_V1)
    src_bytes = normalize_newlines(SOURCE_V1.read_bytes())
    n = sum(1 for line in src_bytes.splitlines() if line.strip())

    if REGRESSION_LABELS.exists() and not force:
        existing = sha256_text_file(REGRESSION_LABELS)
        if existing != src_hash:
            # immutability: never overwrite divergent freeze
            report = {
                "ok": False,
                "error": "regression-v1 already frozen with different hash; refuse overwrite",
                "existing_sha256": existing,
                "source_sha256": src_hash,
                "hash_mode": "lf_normalized",
            }
            FREEZE_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report
        # same content — refresh manifest only
    else:
        # Write LF-normalized bytes so working tree matches manifest hash mode
        REGRESSION_LABELS.write_bytes(src_bytes)

    labels_hash = sha256_text_file(REGRESSION_LABELS)
    # also hash related frozen configs snapshot (LF-normalized)
    cfg_hashes = {}
    for name in ("categories.yaml", "arms.yaml", "selection_rules.yaml"):
        p = CONFIG_DIR / name
        if p.exists():
            cfg_hashes[name] = sha256_text_file(p)

    labels_v2_hash = (
        sha256_text_file(DATA_DIR / "labels.jsonl")
        if (DATA_DIR / "labels.jsonl").exists()
        else None
    )

    manifest = {
        "schema": 1,
        "name": "regression-v1",
        "source": "data/labels_v1.jsonl",
        "n_labels": n,
        "labels_sha256": labels_hash,
        "source_sha256": src_hash,
        "immutable": True,
        "hash_mode": "lf_normalized",
    }
    REGRESSION_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = {
        "ok": True,
        "regression_path": "data/regression_v1/labels.jsonl",
        "n_labels": n,
        "labels_v1_sha256": src_hash,
        "labels_jsonl_sha256": labels_v2_hash,
        "regression_sha256": labels_hash,
        "config_sha256": cfg_hashes,
        "match_source": labels_hash == src_hash,
        "hash_mode": "lf_normalized",
    }
    FREEZE_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def verify() -> dict[str, Any]:
    """Verify regression-v1 exists, n==35, LF-normalized SHA matches manifest."""
    ensure_dirs()
    errors: list[str] = []
    if not REGRESSION_LABELS.exists():
        errors.append("missing regression_v1/labels.jsonl")
    if not REGRESSION_MANIFEST.exists():
        errors.append("missing regression_v1/manifest.json")
    if errors:
        return {"ok": False, "errors": errors}

    manifest = json.loads(REGRESSION_MANIFEST.read_text(encoding="utf-8"))
    # LF-normalized hash: equal for CRLF and LF working trees with same logical lines
    actual = sha256_text_file(REGRESSION_LABELS)
    n = sum(
        1
        for line in normalize_newlines(REGRESSION_LABELS.read_bytes()).splitlines()
        if line.strip()
    )
    if actual != manifest.get("labels_sha256"):
        errors.append(
            f"sha mismatch: actual={actual} manifest={manifest.get('labels_sha256')}"
        )
    if n != 35:
        errors.append(f"expected 35 labels, found {n}")
    if n != manifest.get("n_labels"):
        errors.append(f"n_labels manifest={manifest.get('n_labels')} actual={n}")
    # categories / arms / selection_rules presence
    for name in ("categories.yaml", "arms.yaml", "selection_rules.yaml"):
        if not (CONFIG_DIR / name).exists():
            errors.append(f"missing config/{name}")
    # arms count
    if (CONFIG_DIR / "arms.yaml").exists():
        import yaml

        arms = yaml.safe_load((CONFIG_DIR / "arms.yaml").read_text(encoding="utf-8")) or {}
        arm_list = arms.get("arms") or []
        if len(arm_list) != 10:
            errors.append(f"arms.yaml expected 10 arms, found {len(arm_list)}")
        ids = [a.get("id") for a in arm_list]
        if "hash_dense_k8_r1" not in ids:
            errors.append("baseline_arm_id hash_dense_k8_r1 missing from arms")
    if (CONFIG_DIR / "selection_rules.yaml").exists():
        import yaml

        rules = yaml.safe_load((CONFIG_DIR / "selection_rules.yaml").read_text(encoding="utf-8")) or {}
        if rules.get("baseline_arm_id") != "hash_dense_k8_r1":
            errors.append(f"baseline_arm_id={rules.get('baseline_arm_id')} != hash_dense_k8_r1")
    if (CONFIG_DIR / "categories.yaml").exists():
        import yaml

        cats = yaml.safe_load((CONFIG_DIR / "categories.yaml").read_text(encoding="utf-8")) or {}
        clist = cats.get("categories") or []
        if len(clist) != 11:
            errors.append(f"categories expected 11, found {len(clist)}")

    # Preserve freeze() field names so public evidence / recompute stay stable.
    cfg_hashes: dict[str, str] = {}
    for name in ("categories.yaml", "arms.yaml", "selection_rules.yaml"):
        p = CONFIG_DIR / name
        if p.exists():
            cfg_hashes[name] = sha256_text_file(p)
    src_hash = (
        sha256_text_file(SOURCE_V1) if SOURCE_V1.exists() else manifest.get("source_sha256")
    )
    labels_v2_hash = (
        sha256_text_file(DATA_DIR / "labels.jsonl")
        if (DATA_DIR / "labels.jsonl").exists()
        else None
    )
    result = {
        "ok": len(errors) == 0,
        "errors": errors,
        "regression_path": "data/regression_v1/labels.jsonl",
        "n_labels": n,
        "labels_v1_sha256": src_hash,
        "labels_jsonl_sha256": labels_v2_hash,
        "regression_sha256": actual,
        "labels_sha256": actual,
        "config_sha256": cfg_hashes,
        "match_source": actual == src_hash,
        "hash_mode": "lf_normalized",
        "manifest": manifest,
    }
    # Only rewrite freeze_report when content changes (avoid CRLF/timestamp noise).
    text = json.dumps(result, indent=2) + "\n"
    if not FREEZE_REPORT.exists() or FREEZE_REPORT.read_text(encoding="utf-8") != text:
        FREEZE_REPORT.write_text(text, encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--verify" in argv:
        r = verify()
        print(json.dumps(r, indent=2))
        return 0 if r.get("ok") else 1
    force = "--force" in argv
    r = freeze(force=force)
    print(json.dumps(r, indent=2))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
