"""Concurrency isolation evidence."""

from __future__ import annotations

import json

from rag_bench.config_load import RESULTS_DIR, ensure_dirs
from rag_bench.concurrency_probe import run_concurrency_probe


def test_concurrency_20_sessions():
    ensure_dirs()
    # Prefer fresh probe; fall back to artifact if very slow env
    rep = run_concurrency_probe(20, write=True)
    assert rep["n_sessions"] == 20
    assert rep["n_ok"] == 20
    assert rep["cross_talk"] == 0
    assert rep["kill_isolation_ok"] is True
    assert rep["ok"] is True
    disk = json.loads((RESULTS_DIR / "concurrency_report.json").read_text(encoding="utf-8"))
    assert disk["ok"] is True
