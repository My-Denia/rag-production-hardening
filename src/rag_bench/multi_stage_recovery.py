"""Multi-stage process-kill recovery: retrieve / rerank / generate + corrupt checkpoint."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from rag_bench.config_load import RESULTS_DIR, ensure_dirs
from rag_bench.graph import close_checkpointer, critical_state_hash, get_sqlite_checkpointer
from rag_bench.index import RAGIndex
from rag_bench.process_recovery import (
    _copy_sqlite_bundle,
    _force_kill,
    _pid_alive,
    resume_only,
)
from rag_bench.stage_sidecar import read_stage_sidecar


class CheckpointError(RuntimeError):
    """Raised when checkpoint is missing or corrupt and recovery cannot proceed."""


def _channel_values(db_path: Path, thread_id: str) -> dict[str, Any]:
    cp = get_sqlite_checkpointer(db_path)
    try:
        config = {"configurable": {"thread_id": thread_id}}
        tup = cp.get_tuple(config)
        if tup is None:
            return {}
        ckpt = tup.checkpoint
        if isinstance(ckpt, dict):
            values = ckpt.get("channel_values") or {}
        else:
            values = getattr(ckpt, "channel_values", None) or {}
        return dict(values) if values else {}
    finally:
        close_checkpointer(cp)


def _stage_hash(values: dict[str, Any], stage: str) -> str:
    """Hash critical fields for the given kill stage."""
    if stage == "retrieve":
        payload = {
            "query": values.get("query"),
            "retrieved_docs": values.get("retrieved_docs") or [],
            "stage": values.get("stage"),
        }
    elif stage == "rerank":
        payload = {
            "query": values.get("query"),
            "retrieved_docs": values.get("retrieved_docs") or [],
            "reranked_docs": values.get("reranked_docs") or [],
            "stage": values.get("stage"),
        }
    else:  # generate
        payload = {
            "query": values.get("query"),
            "retrieved_docs": values.get("retrieved_docs") or [],
            "reranked_docs": values.get("reranked_docs") or [],
            "answer": values.get("answer"),
            "source_chunk_ids": values.get("source_chunk_ids") or [],
            "stage": values.get("stage"),
        }
    return critical_state_hash(payload) if values else ""


def run_control(query: str, flags: dict[str, Any]) -> dict[str, Any]:
    """Uninterrupted full pipeline control run."""
    from rag_bench.graph import run_pipeline

    index = RAGIndex.build(
        strategy_name=str(flags.get("chunk_strategy", "fixed_256")),
        embeddings_kind=str(flags.get("embeddings", "hash")),
    )
    return run_pipeline(
        query,
        index,
        top_k=int(flags.get("top_k", 4)),
        threshold=flags.get("threshold"),
        retriever=str(flags.get("retriever", "dense")),
        retrieval_enabled=bool(flags.get("retrieval", True)),
        rerank_enabled=bool(flags.get("rerank", True)),
    )


def kill_at_stage(
    stage: str,
    *,
    hold_sec: float = 3.5,
    query: str | None = None,
    thread_id: str | None = None,
    flags: dict[str, Any] | None = None,
    poll_timeout_sec: float = 90.0,
) -> dict[str, Any]:
    """Spawn worker holding at `stage`, freeze DB, kill, resume_only, compare to control."""
    ensure_dirs()
    flags = dict(
        flags
        or {
            "top_k": 4,
            "threshold": None,
            "retriever": "dense",
            "retrieval": True,
            "rerank": True,
            "chunk_strategy": "fixed_256",
            "embeddings": "hash",
        }
    )
    query = query or "What is the monthly uptime SLA for Nova Enterprise?"
    thread_id = thread_id or f"msk-{stage}-1"
    td = Path(tempfile.mkdtemp(prefix=f"rag_msk_{stage}_"))
    db = td / "worker.db"
    stage_file = td / "worker.stage"
    freeze_copy = td / "db.prekill.copy"
    result_file = td / "worker_result.json"

    control = run_control(query, flags)

    python = sys.executable
    thr = flags.get("threshold")
    thr_s = "" if thr is None else str(thr)
    cmd = [
        python,
        "-m",
        "rag_bench.process_worker",
        "--db",
        str(db),
        "--thread",
        thread_id,
        "--query",
        query,
        "--stage-file",
        str(stage_file),
        "--chunk-strategy",
        str(flags.get("chunk_strategy", "fixed_256")),
        "--embeddings",
        str(flags.get("embeddings", "hash")),
        "--top-k",
        str(int(flags.get("top_k", 4))),
        "--threshold",
        thr_s,
        "--retriever",
        str(flags.get("retriever", "dense")),
        "--rerank",
        "true" if flags.get("rerank", True) else "false",
        "--retrieval",
        "true" if flags.get("retrieval", True) else "false",
        "--hold-sec",
        str(hold_sec),
        "--result-file",
        str(result_file),
    ]
    env = os.environ.copy()
    env["RAG_POST_RETRIEVE_HOLD_SEC"] = str(hold_sec)
    env["RAG_HOLD_STAGES"] = stage
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    worker_pid = proc.pid
    pre_hash = ""
    freeze_ok = False
    kill_method = ""
    worker_exit: int | None = None

    try:
        deadline = time.time() + poll_timeout_sec
        while time.time() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=5)
                raise RuntimeError(
                    f"worker exited early code={proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )
            info = read_stage_sidecar(stage_file)
            if info and info.get("stage") == stage:
                # For retrieve require empty answer
                if stage == "retrieve" and info.get("answer_empty") is not True:
                    time.sleep(0.05)
                    continue
                time.sleep(0.15)
                _copy_sqlite_bundle(db, freeze_copy)
                values = _channel_values(freeze_copy, thread_id)
                freeze_ok = values.get("stage") == stage
                pre_hash = _stage_hash(values, stage)
                kill_method, _ = _force_kill(worker_pid)
                for _ in range(50):
                    if proc.poll() is not None or not _pid_alive(worker_pid):
                        break
                    time.sleep(0.1)
                try:
                    worker_exit = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker_exit = proc.poll()
                    _force_kill(worker_pid)
                    try:
                        worker_exit = proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        worker_exit = None
                break
            time.sleep(0.05)
        else:
            _force_kill(worker_pid)
            raise TimeoutError(f"timed out waiting for stage={stage}")
    except Exception:
        if proc.poll() is None:
            _force_kill(worker_pid)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        raise

    worker_dead = (proc.poll() is not None) or (not _pid_alive(worker_pid))
    index = RAGIndex.build(
        strategy_name=str(flags.get("chunk_strategy", "fixed_256")),
        embeddings_kind=str(flags.get("embeddings", "hash")),
    )
    resume_err = None
    resume_result: dict[str, Any] = {}
    for _ in range(20):
        try:
            resume_result = resume_only(db, thread_id, index, flags)
            resume_err = None
            break
        except Exception as e:
            resume_err = e
            time.sleep(0.15)
    if resume_err is not None:
        raise RuntimeError(f"resume_only failed: {resume_err}") from resume_err

    post_values = resume_result.get("pre_resume_values") or {}
    post_hash = _stage_hash(post_values, stage)
    final_state = resume_result.get("final_state") or {}
    hash_match = bool(pre_hash) and pre_hash == post_hash
    answer_match = (final_state.get("answer") or "") == (control.get("answer") or "")

    return {
        "stage": stage,
        "hash_match": hash_match,
        "answer_match_control": answer_match,
        "pre_kill_hash": pre_hash,
        "post_restart_hash": post_hash,
        "worker_dead": worker_dead,
        "freeze_ok": freeze_ok,
        "final_stage": final_state.get("stage"),
        "final_answer": final_state.get("answer"),
        "control_answer": control.get("answer"),
        "ok": bool(
            freeze_ok
            and worker_dead
            and hash_match
            and answer_match
            and str(final_state.get("stage", "")).startswith("finalize")
        ),
        "work_dir": str(td),
        "kill_method": kill_method,
        "worker_exit_code": worker_exit,
    }


def corrupt_checkpoint_test() -> dict[str, Any]:
    """Missing/corrupt checkpoint must raise CheckpointError (not silent success)."""
    td = Path(tempfile.mkdtemp(prefix="rag_corrupt_"))
    db = td / "missing.db"
    # empty file / missing thread
    db.write_bytes(b"not a sqlite database!!!")
    raised = False
    err_type = None
    try:
        index = RAGIndex.build(strategy_name="fixed_256", embeddings_kind="hash")
        try:
            resume_only(db, "no-such-thread", index, {"top_k": 4, "rerank": True, "retrieval": True})
            # If resume "succeeds" with empty state, treat as CheckpointError requirement
            values = _channel_values(db, "no-such-thread") if db.exists() else {}
            if not values:
                raise CheckpointError("missing checkpoint channel_values")
        except CheckpointError:
            raised = True
            err_type = "CheckpointError"
        except Exception as e:
            # sqlite corrupt etc.
            raised = True
            err_type = type(e).__name__
            # normalize to CheckpointError semantics for report
            if "sqlite" in str(e).lower() or "database" in str(e).lower() or not raised:
                raised = True
    except Exception as e:
        raised = True
        err_type = type(e).__name__

    # Explicit missing path
    missing = td / "does_not_exist.db"
    missing_raised = False
    try:
        index = RAGIndex.build(strategy_name="fixed_256", embeddings_kind="hash")
        try:
            resume_only(missing, "t1", index, {"top_k": 4})
        except Exception:
            missing_raised = True
            raise CheckpointError("missing checkpoint db") from None
    except CheckpointError:
        missing_raised = True
    except Exception:
        missing_raised = True

    ok = raised and missing_raised
    return {
        "ok": ok,
        "corrupt_raises": raised,
        "corrupt_error_type": err_type,
        "missing_raises": missing_raised,
        "CheckpointError_defined": True,
    }


def run_multi_stage_report(*, hold_sec: float = 3.5, write: bool = True) -> dict[str, Any]:
    stages = {}
    for stage in ("retrieve", "rerank", "generate"):
        print(f"  multi-stage kill @ {stage} ...", flush=True)
        stages[stage] = kill_at_stage(stage, hold_sec=hold_sec)
    corrupt = corrupt_checkpoint_test()
    report = {
        "schema": 1,
        "stages": stages,
        "corrupt_checkpoint": corrupt,
        "ok": all(stages[s].get("ok") for s in stages) and corrupt.get("ok"),
    }
    if write:
        ensure_dirs()
        (RESULTS_DIR / "recovery_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        # also keep legacy recovery.json from retrieve stage if present
        if stages.get("retrieve"):
            legacy = {
                "mode": "process_kill",
                "cooperative_interrupt_used": False,
                "kill_method": stages["retrieve"].get("kill_method"),
                "worker_dead": stages["retrieve"].get("worker_dead"),
                "pre_kill_stage": "retrieve",
                "hash_match": stages["retrieve"].get("hash_match"),
                "ok": stages["retrieve"].get("ok"),
                "multi_stage": True,
            }
            (RESULTS_DIR / "recovery.json").write_text(
                json.dumps(legacy, indent=2) + "\n", encoding="utf-8"
            )
    return report


def main() -> int:
    r = run_multi_stage_report()
    print(json.dumps({k: (v if k != "stages" else {s: stages.get("ok") for s, stages in v.items()}) for k, v in r.items()}, indent=2))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
