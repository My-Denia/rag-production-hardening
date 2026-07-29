"""Holdout seal tests (plan: self-contained)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_bench.config_load import RESULTS_DIR
from rag_bench.holdout import (
    HOLDOUT_LABELS,
    HoldoutSealedError,
    load_labels,
    verify_holdout_immutable,
)


def test_holdout_load_without_unlock_raises():
    if not HOLDOUT_LABELS.exists():
        pytest.skip("holdout labels not built yet")
    os.environ.pop("RAG_HOLDOUT_UNLOCK", None)
    with pytest.raises(HoldoutSealedError):
        load_labels(HOLDOUT_LABELS)


def test_finalize_requires_shortlist(monkeypatch):
    if not HOLDOUT_LABELS.exists():
        pytest.skip("holdout labels not built yet")
    from rag_bench import selection as sel

    monkeypatch.delenv("RAG_HOLDOUT_UNLOCK", raising=False)
    # Without shortlist → refuse
    shortlist = RESULTS_DIR / "dev_shortlist.json"
    if shortlist.exists():
        # temporarily rename
        bak = shortlist.with_suffix(".json.bak_test")
        shortlist.rename(bak)
        try:
            with pytest.raises(FileNotFoundError):
                sel.finalize_selection()
        finally:
            bak.rename(shortlist)
    else:
        with pytest.raises(FileNotFoundError):
            sel.finalize_selection()


def test_holdout_load_with_unlock():
    if not HOLDOUT_LABELS.exists():
        pytest.skip("holdout labels not built yet")
    os.environ["RAG_HOLDOUT_UNLOCK"] = "1"
    try:
        labs = load_labels(HOLDOUT_LABELS)
        assert len(labs) >= 1
    finally:
        os.environ.pop("RAG_HOLDOUT_UNLOCK", None)


def test_holdout_immutable_manifest():
    if not HOLDOUT_LABELS.exists():
        pytest.skip("holdout labels not built yet")
    assert verify_holdout_immutable()


def test_ablate_and_run_arms_dev_never_reference_holdout_path():
    """grep-style: ablate / arms modules must not hardcode holdout path string."""
    root = Path(__file__).resolve().parents[1] / "src" / "rag_bench"
    forbidden = "data/holdout/labels.jsonl"
    for name in ("ablate.py", "arms.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert forbidden not in text.replace("\\", "/")
        assert "holdout/labels" not in text.replace("\\", "/")
