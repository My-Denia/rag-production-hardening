"""Parent controller: spawn worker, freeze DB, process-kill, resume_only, write recovery.json.

Primary AC1 evidence path — cooperative interrupt_after is NOT used.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rag_bench.config_load import RESULTS_DIR, ensure_dirs
from rag_bench.graph import (
    build_graph,
    close_checkpointer,
    critical_state_hash,
    get_sqlite_checkpointer,
)
from rag_bench.index import RAGIndex
from rag_bench.stage_sidecar import read_stage_sidecar


REQUIRED_RECOVERY_KEYS = [
    "mode",
    "cooperative_interrupt_used",
    "kill_method",
    "worker_pid",
    "worker_exit_code",
    "worker_dead",
    "pre_kill_stage",
    "pre_kill_answer_empty",
    "pre_kill_n_retrieved",
    "pre_kill_hash",
    "post_restart_hash",
    "hash_match",
    "recoverer_received_interrupt_node",
    "final_stage",
    "answer_present",
    "hold_sec",
    "db_freeze_verified",
]


def _copy_sqlite_bundle(src: Path, dest: Path) -> list[str]:
    """Copy SQLite main file + WAL/SHM sidecars if present."""
    copied: list[str] = []
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    copied.append(str(dest))
    for suffix in ("-wal", "-shm"):
        sp = Path(str(src) + suffix)
        if sp.exists():
            dp = Path(str(dest) + suffix)
            shutil.copy2(sp, dp)
            copied.append(str(dp))
    return copied


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # tasklist is reliable enough for tests
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
            text = (out.stdout or "") + (out.stderr or "")
            return str(pid) in text and "No tasks" not in text
        except Exception:
            pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _force_kill(pid: int) -> tuple[str, int | None]:
    """Non-cooperative kill. Returns (kill_method, exit_code_if_known)."""
    if sys.platform == "win32":
        r = subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        return "taskkill", r.returncode
    try:
        os.kill(pid, 9)
        return "Process.kill", None
    except ProcessLookupError:
        return "Process.kill", None


def _channel_values_from_db(db_path: Path, thread_id: str) -> dict[str, Any]:
    """
    Read durable checkpoint channel_values via SqliteSaver.get_tuple.

    Prefer this over compiled app.get_state(), which can surface pending writes
    / internal branch channels and mis-report stage mid-flight.
    """
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


def verify_checkpoint_at_retrieve(db_path: Path, thread_id: str) -> dict[str, Any]:
    """Open DB (or freeze copy) and verify retrieve-stage checkpoint values."""
    values = _channel_values_from_db(db_path, thread_id)
    stage = values.get("stage")
    answer = values.get("answer") or ""
    retrieved = values.get("retrieved_docs") or []
    ok = (
        stage == "retrieve"
        and not str(answer).strip()
        and isinstance(retrieved, list)
        and len(retrieved) > 0
    )
    # Hash only critical durable fields (ignore internal branch:* channels)
    hash_state = {
        "query": values.get("query"),
        "retrieved_docs": values.get("retrieved_docs") or [],
        "stage": values.get("stage"),
    }
    h = critical_state_hash(hash_state) if values else ""
    return {
        "ok": ok,
        "stage": stage,
        "answer_empty": not bool(str(answer).strip()),
        "n_retrieved": len(retrieved) if isinstance(retrieved, list) else 0,
        "hash": h,
        "values": values,
    }


def resume_only(
    db_path: str | Path,
    thread_id: str,
    index: RAGIndex,
    flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resume a checkpointed run from db+thread_id only.

    Signature deliberately excludes interrupt node names.
    recoverer_received_interrupt_node is always False for this API.
    """
    # Guard: signature must not accept interrupt node parameters.
    sig = inspect.signature(resume_only)
    for name in sig.parameters:
        if "interrupt" in name.lower():
            raise RuntimeError("resume_only must not take interrupt parameters")

    flags = flags or {}
    top_k = int(flags.get("top_k", 4))
    threshold = flags.get("threshold")
    retriever = str(flags.get("retriever", "dense"))
    retrieval_enabled = bool(flags.get("retrieval", True))
    rerank_enabled = bool(flags.get("rerank", True))

    # Hash from durable checkpoint before any resume invoke.
    values = _channel_values_from_db(Path(db_path), thread_id)
    hash_state = {
        "query": values.get("query"),
        "retrieved_docs": values.get("retrieved_docs") or [],
        "stage": values.get("stage"),
    }
    post_hash = critical_state_hash(hash_state) if values else ""

    cp = get_sqlite_checkpointer(db_path)
    try:
        app = build_graph(
            index,
            top_k=top_k,
            threshold=threshold,
            retriever=retriever,
            retrieval_enabled=retrieval_enabled,
            rerank_enabled=rerank_enabled,
            checkpointer=cp,
            interrupt_after=None,
        )
        config = {"configurable": {"thread_id": thread_id}}
        final = app.invoke(None, config=config)
        final_state = dict(final)
        return {
            "post_restart_hash": post_hash,
            "pre_resume_values": values,
            "final_state": final_state,
            "recoverer_received_interrupt_node": False,
            "resume_only_params": list(sig.parameters.keys()),
        }
    finally:
        close_checkpointer(cp)


def run_process_kill_probe(
    *,
    hold_sec: float = 4.0,
    query: str | None = None,
    thread_id: str = "process-kill-probe-1",
    work_dir: str | Path | None = None,
    flags: dict[str, Any] | None = None,
    write_results: bool = True,
    poll_timeout_sec: float = 60.0,
) -> dict[str, Any]:
    """
    Spawn worker → wait stage=retrieve → freeze DB verify → kill → resume_only → recovery.json
    """
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

    import tempfile

    td = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="rag_proc_"))
    td.mkdir(parents=True, exist_ok=True)
    db = td / "worker.db"
    stage_file = td / "worker.stage"
    freeze_copy = td / "db.prekill.copy"
    result_file = td / "worker_result.json"

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
    # Ensure package importable
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
    kill_method = ""
    worker_exit: int | None = None
    pre_kill_hash = ""
    freeze_ok = False
    stage_info: dict[str, Any] = {}
    pre_kill_n = 0

    try:
        deadline = time.time() + poll_timeout_sec
        while time.time() < deadline:
            if proc.poll() is not None:
                # Worker exited early — capture output for diagnostics
                out, err = proc.communicate(timeout=5)
                raise RuntimeError(
                    f"worker exited early code={proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )
            info = read_stage_sidecar(stage_file)
            if (
                info
                and info.get("stage") == "retrieve"
                and info.get("answer_empty") is True
            ):
                stage_info = info
                # Freeze durability: copy DB + verify on copy
                # Small settle for WAL
                time.sleep(0.15)
                _copy_sqlite_bundle(db, freeze_copy)
                verified = verify_checkpoint_at_retrieve(freeze_copy, thread_id)
                freeze_ok = bool(verified.get("ok"))
                pre_kill_hash = str(verified.get("hash") or "")
                pre_kill_n = int(verified.get("n_retrieved") or 0)
                if not freeze_ok:
                    # Still kill/cleanup but mark fail
                    pass
                kill_method, _ = _force_kill(worker_pid)
                # Wait for death
                for _ in range(50):
                    if proc.poll() is not None:
                        break
                    if not _pid_alive(worker_pid):
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
            try:
                out, err = proc.communicate(timeout=5)
            except Exception:
                out, err = "", ""
            raise TimeoutError(
                f"timed out waiting for retrieve stage; stage={read_stage_sidecar(stage_file)}\n"
                f"stdout={out}\nstderr={err}"
            )
    except Exception:
        if proc.poll() is None:
            _force_kill(worker_pid)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        raise

    worker_dead = (proc.poll() is not None) or (not _pid_alive(worker_pid))

    # Resume in NEW process API (in-process new compile; no interrupt node)
    # Retry open if Windows still releasing locks briefly.
    index = RAGIndex.build(
        strategy_name=str(flags.get("chunk_strategy", "fixed_256")),
        embeddings_kind=str(flags.get("embeddings", "hash")),
    )
    resume_err = None
    resume_result: dict[str, Any] = {}
    for attempt in range(20):
        try:
            resume_result = resume_only(db, thread_id, index, flags)
            resume_err = None
            break
        except Exception as e:
            resume_err = e
            time.sleep(0.15)
    if resume_err is not None:
        raise RuntimeError(f"resume_only failed after retries: {resume_err}") from resume_err

    post_hash = str(resume_result.get("post_restart_hash") or "")
    final_state = resume_result.get("final_state") or {}
    hash_match = bool(pre_kill_hash) and pre_kill_hash == post_hash

    payload: dict[str, Any] = {
        "mode": "process_kill",
        "cooperative_interrupt_used": False,
        "kill_method": kill_method or "taskkill",
        "worker_pid": worker_pid,
        "worker_exit_code": worker_exit,
        "worker_dead": bool(worker_dead),
        "pre_kill_stage": "retrieve",
        "pre_kill_answer_empty": True,
        "pre_kill_n_retrieved": pre_kill_n or int(stage_info.get("n_retrieved") or 0),
        "pre_kill_hash": pre_kill_hash,
        "post_restart_hash": post_hash,
        "hash_match": hash_match,
        "recoverer_received_interrupt_node": bool(
            resume_result.get("recoverer_received_interrupt_node", False)
        ),
        "final_stage": final_state.get("stage"),
        "answer_present": bool(final_state.get("answer")),
        "hold_sec": float(hold_sec),
        "db_freeze_verified": bool(freeze_ok),
        "ok": bool(
            freeze_ok
            and worker_dead
            and hash_match
            and final_state.get("stage") == "finalize"
            and bool(final_state.get("answer"))
            and not resume_result.get("recoverer_received_interrupt_node", False)
        ),
        "work_dir": str(td),
        "thread_id": thread_id,
        "query": query,
        "resume_only_params": resume_result.get("resume_only_params"),
    }

    # Required key presence check
    missing = [k for k in REQUIRED_RECOVERY_KEYS if k not in payload]
    payload["required_keys_missing"] = missing

    if write_results:
        out = RESULTS_DIR / "recovery.json"
        # Never persist absolute temp/user paths in results artifacts
        disk = dict(payload)
        disk["work_dir"] = "<REDACTED_TEMP_DIR>"
        with out.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(disk, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    payload = run_process_kill_probe()
    print(json.dumps({k: payload[k] for k in REQUIRED_RECOVERY_KEYS + ["ok"]}, indent=2))
    return 0 if payload.get("ok") and not payload.get("required_keys_missing") else 1


if __name__ == "__main__":
    sys.exit(main())
