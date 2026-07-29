"""OS worker: run full LangGraph (no interrupt_after) with durable stage sidecar.

Usage:
  python -m rag_bench.process_worker --db PATH --thread ID --query TEXT --stage-file PATH
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG process worker (full graph)")
    p.add_argument("--db", required=True, help="Sqlite checkpoint DB path")
    p.add_argument("--thread", required=True, help="thread_id")
    p.add_argument("--query", required=True, help="user query")
    p.add_argument("--stage-file", required=True, help="atomic stage sidecar path")
    p.add_argument("--chunk-strategy", default="fixed_256")
    p.add_argument("--embeddings", default="hash")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--threshold", default="", help="empty = none")
    p.add_argument("--retriever", default="dense")
    p.add_argument("--rerank", default="true")
    p.add_argument("--retrieval", default="true")
    p.add_argument("--hold-sec", type=float, default=None, help="override RAG_POST_RETRIEVE_HOLD_SEC")
    p.add_argument("--result-file", default="", help="optional final state JSON path")
    return p


def _as_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.hold_sec is not None:
        os.environ["RAG_POST_RETRIEVE_HOLD_SEC"] = str(args.hold_sec)

    from rag_bench.graph import build_graph, close_checkpointer, get_sqlite_checkpointer
    from rag_bench.index import RAGIndex
    from rag_bench.stage_checkpointer import StageAwareCheckpointer

    thr = args.threshold.strip()
    threshold = None if thr in ("", "null", "None", "none") else float(thr)
    flags = {
        "top_k": int(args.top_k),
        "threshold": threshold,
        "retriever": args.retriever,
        "retrieval": _as_bool(args.retrieval),
        "rerank": _as_bool(args.rerank),
    }

    index = RAGIndex.build(
        strategy_name=args.chunk_strategy,
        embeddings_kind=args.embeddings,
    )

    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    raw_cp = get_sqlite_checkpointer(db)
    # Prefer WAL for concurrent freeze reads while worker holds connection.
    conn = getattr(raw_cp, "conn", None)
    if conn is not None:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
        except Exception:
            pass

    hold = float(os.environ.get("RAG_POST_RETRIEVE_HOLD_SEC", "0") or "0")
    cp = StageAwareCheckpointer(raw_cp, args.stage_file, hold_sec=hold)

    try:
        # Primary path: NO interrupt_after — full graph intended.
        app = build_graph(
            index,
            top_k=flags["top_k"],
            threshold=flags["threshold"],
            retriever=flags["retriever"],
            retrieval_enabled=flags["retrieval"],
            rerank_enabled=flags["rerank"],
            checkpointer=cp,
            interrupt_after=None,
        )
        config = {"configurable": {"thread_id": args.thread}}
        final = app.invoke(
            {
                "query": args.query,
                "config_flags": flags,
                "stage": "start",
            },
            config=config,
        )
        if args.result_file:
            out = Path(args.result_file)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump(dict(final), f, ensure_ascii=False, default=str)
                f.write("\n")
        return 0
    finally:
        close_checkpointer(raw_cp)


if __name__ == "__main__":
    sys.exit(main())
