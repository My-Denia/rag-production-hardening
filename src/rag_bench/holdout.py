"""Holdout seal: load requires RAG_HOLDOUT_UNLOCK=1; finalize_selection is sole unlock path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rag_bench.config_load import DATA_DIR, RESULTS_DIR

HOLDOUT_DIR = DATA_DIR / "holdout"
HOLDOUT_LABELS = HOLDOUT_DIR / "labels.jsonl"
HOLDOUT_MANIFEST = HOLDOUT_DIR / "manifest.json"
DEV_LABELS = DATA_DIR / "dev" / "labels.jsonl"
DEV_SHORTLIST = RESULTS_DIR / "dev_shortlist.json"
UNLOCK_ENV = "RAG_HOLDOUT_UNLOCK"


class HoldoutSealedError(PermissionError):
    """Raised when holdout labels are loaded without unlock."""


def holdout_is_unlocked() -> bool:
    return os.environ.get(UNLOCK_ENV, "").strip() in {"1", "true", "yes"}


def load_labels(path: Path | str, *, require_unlock: bool | None = None) -> list[dict[str, Any]]:
    """
    Load JSONL labels. If path is the sealed holdout path, require unlock env.
    """
    p = Path(path)
    is_holdout = False
    try:
        is_holdout = p.resolve() == HOLDOUT_LABELS.resolve()
    except Exception:
        is_holdout = str(p).replace("\\", "/").endswith("data/holdout/labels.jsonl")

    if require_unlock is None:
        require_unlock = is_holdout

    if require_unlock and not holdout_is_unlocked():
        raise HoldoutSealedError(
            f"Holdout sealed: set {UNLOCK_ENV}=1 only via finalize_selection after dev_shortlist.json"
        )

    labels: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(json.loads(line))
    return labels


def holdout_bytes_sha256() -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(HOLDOUT_LABELS.read_bytes())
    return h.hexdigest()


def verify_holdout_immutable() -> bool:
    """True if holdout file bytes match manifest sha."""
    if not HOLDOUT_LABELS.exists() or not HOLDOUT_MANIFEST.exists():
        return False
    manifest = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    return holdout_bytes_sha256() == manifest.get("labels_sha256")
