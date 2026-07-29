"""Multi-stage process-kill recovery + corrupt checkpoint semantics."""

from __future__ import annotations

import json

import pytest

from rag_bench.config_load import RESULTS_DIR, ensure_dirs
from rag_bench.multi_stage_recovery import (
    CheckpointError,
    corrupt_checkpoint_test,
    kill_at_stage,
    run_multi_stage_report,
)


def test_checkpoint_error_type_exists():
    assert issubclass(CheckpointError, RuntimeError)


def test_corrupt_checkpoint_raises():
    out = corrupt_checkpoint_test()
    assert out.get("corrupt_raises") is True
    assert out.get("missing_raises") is True
    assert out.get("ok") is True


@pytest.mark.slow
def test_kill_at_retrieve():
    ensure_dirs()
    r = kill_at_stage("retrieve", hold_sec=3.0, poll_timeout_sec=90.0)
    assert r["hash_match"] is True
    assert r["answer_match_control"] is True
    assert r["ok"] is True


@pytest.mark.slow
def test_multi_stage_report_smoke():
    """Use pre-written recovery_report if present from run_all; else run stages."""
    path = RESULTS_DIR / "recovery_report.json"
    if path.exists():
        rep = json.loads(path.read_text(encoding="utf-8"))
        assert "stages" in rep
        for st in ("retrieve", "rerank", "generate"):
            assert st in rep["stages"]
            assert rep["stages"][st].get("hash_match") is True
            assert rep["stages"][st].get("answer_match_control") is True
        assert rep.get("corrupt_checkpoint", {}).get("ok") is True
    else:
        rep = run_multi_stage_report(hold_sec=3.0, write=True)
        assert rep.get("ok") is True
