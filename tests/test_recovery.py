# LEGACY — not AC1 evidence
"""Legacy cooperative interrupt_after recovery (non-primary)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rag_bench.config_load import RESULTS_DIR, ensure_dirs
from rag_bench.graph import (
    build_graph,
    close_checkpointer,
    critical_state_hash,
    get_sqlite_checkpointer,
)
from rag_bench.index import RAGIndex


def test_legacy_cooperative_interrupt_recovery():
    """Cooperative interrupt_after path — NOT primary AC1 process-kill evidence."""
    ensure_dirs()
    index = RAGIndex.build(strategy_name="fixed_512", embeddings_kind="hash")
    flags = {
        "top_k": 4,
        "threshold": None,
        "retriever": "dense",
        "retrieval": True,
        "rerank": True,
    }
    query = "What is the Nova Enterprise SLA uptime?"

    td = tempfile.mkdtemp()
    cp = None
    try:
        db = Path(td) / "recovery.db"
        cp = get_sqlite_checkpointer(db)
        thread_id = "test-recovery-thread"
        config = {"configurable": {"thread_id": thread_id}}

        app_a = build_graph(
            index,
            top_k=4,
            rerank_enabled=True,
            retrieval_enabled=True,
            checkpointer=cp,
            interrupt_after=["retrieve"],
        )
        app_a.invoke(
            {"query": query, "config_flags": flags, "stage": "start"},
            config=config,
        )
        state_a = dict(app_a.get_state(config).values)
        assert state_a.get("retrieved_docs"), "expected docs after retrieve"
        hash_a = critical_state_hash(state_a)

        app_b = build_graph(
            index,
            top_k=4,
            rerank_enabled=True,
            retrieval_enabled=True,
            checkpointer=cp,
            interrupt_after=["retrieve"],
        )
        state_b = dict(app_b.get_state(config).values)
        hash_b = critical_state_hash(state_b)
        assert hash_a == hash_b, f"hash mismatch {hash_a} != {hash_b}"

        app_c = build_graph(
            index,
            top_k=4,
            rerank_enabled=True,
            retrieval_enabled=True,
            checkpointer=cp,
        )
        final = app_c.invoke(None, config=config)
        assert final.get("stage") == "finalize"
        assert final.get("answer")
        assert isinstance(final.get("source_chunk_ids"), list)

        # Do NOT overwrite primary recovery.json (process_kill). Write legacy sidecar.
        payload = {
            "ok": True,
            "mode": "cooperative_interrupt",
            "legacy": True,
            "not_ac1_evidence": True,
            "pre_interrupt_hash": hash_a,
            "post_restart_hash": hash_b,
            "hash_match": hash_a == hash_b,
            "final_stage": final.get("stage"),
            "answer_present": bool(final.get("answer")),
            "n_retrieved": len(state_a.get("retrieved_docs") or []),
            "thread_id": thread_id,
            "source": "tests/test_recovery.py",
        }
        out = RESULTS_DIR / "recovery_legacy_cooperative.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    finally:
        if cp is not None:
            close_checkpointer(cp)
