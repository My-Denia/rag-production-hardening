"""Primary AC1 evidence: non-cooperative process kill + resume_only."""

from __future__ import annotations

import inspect
import json

from rag_bench.config_load import RESULTS_DIR, ensure_dirs
from rag_bench.process_recovery import (
    REQUIRED_RECOVERY_KEYS,
    resume_only,
    run_process_kill_probe,
)


def test_resume_only_signature_has_no_interrupt_param():
    sig = inspect.signature(resume_only)
    for name in sig.parameters:
        assert "interrupt" not in name.lower()


def test_process_kill_recovery_probe():
    ensure_dirs()
    payload = run_process_kill_probe(hold_sec=3.5, write_results=True, poll_timeout_sec=90.0)

    for k in REQUIRED_RECOVERY_KEYS:
        assert k in payload, f"missing recovery key: {k}"

    assert payload["mode"] == "process_kill"
    assert payload["cooperative_interrupt_used"] is False
    assert payload["worker_dead"] is True
    assert payload["pre_kill_stage"] == "retrieve"
    assert payload["pre_kill_answer_empty"] is True
    assert payload["pre_kill_n_retrieved"] >= 1
    assert payload["hash_match"] is True
    assert payload["pre_kill_hash"] == payload["post_restart_hash"]
    assert payload["recoverer_received_interrupt_node"] is False
    assert payload["final_stage"] == "finalize"
    assert payload["answer_present"] is True
    assert payload["db_freeze_verified"] is True
    assert payload["hold_sec"] > 0
    assert payload["kill_method"] in ("taskkill", "Process.kill")
    assert payload.get("ok") is True

    # File evidence
    out = RESULTS_DIR / "recovery.json"
    assert out.exists()
    disk = json.loads(out.read_text(encoding="utf-8"))
    assert disk["mode"] == "process_kill"
    assert disk["cooperative_interrupt_used"] is False
    assert disk["hash_match"] is True
