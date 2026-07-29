"""LangGraph orchestration: retrieve → rerank → generate → finalize + SqliteSaver."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag_bench.generate import extractive_answer
from rag_bench.index import RAGIndex
from rag_bench.metrics import ABSTAIN_MARK
from rag_bench.rerank import rerank_documents
from rag_bench.retrieve import RetrieveConfig, docs_to_state, retrieve, state_to_docs


class RAGState(TypedDict, total=False):
    query: str
    retrieved_docs: list[dict[str, Any]]
    reranked_docs: list[dict[str, Any]]
    answer: str
    source_chunk_ids: list[str]
    stage: str
    config_flags: dict[str, Any]
    error: str
    top1_score: float
    top2_score: float
    abstained: bool


def critical_state_hash(state: dict[str, Any]) -> str:
    """Hash query + retrieved_docs (chunk ids/content) for recovery assertions."""
    payload = {
        "query": state.get("query"),
        "retrieved_docs": state.get("retrieved_docs") or [],
        "stage": state.get("stage"),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _top_scores(docs: list[Document]) -> tuple[float, float]:
    scores = []
    for d in docs:
        s = d.metadata.get("score")
        if s is not None:
            scores.append(float(s))
    s1 = scores[0] if len(scores) > 0 else 0.0
    s2 = scores[1] if len(scores) > 1 else 0.0
    return s1, s2


def build_graph(
    index: RAGIndex,
    *,
    top_k: int = 4,
    threshold: float | None = None,
    retriever: str = "dense",
    retrieval_enabled: bool = True,
    rerank_enabled: bool = True,
    rrf_k: int = 60,
    query_strategy: str = "raw",
    abstain: str = "none",
    median_dev: float | None = None,
    margin: float = 0.05,
    checkpointer=None,
    interrupt_after: list[str] | None = None,
    post_node_hold_sec: float | None = None,
):
    """
    Compile LangGraph with nodes retrieve → rerank → generate → finalize.

    abstain=margin_0.05: abstain if (s1-s2)<margin AND s1 < median_dev.
    """
    _ = post_node_hold_sec

    def node_retrieve(state: RAGState) -> dict[str, Any]:
        flags = state.get("config_flags") or {}
        cfg = RetrieveConfig(
            top_k=int(flags.get("top_k", top_k)),
            threshold=flags.get("threshold", threshold),
            retriever=str(flags.get("retriever", retriever)),
            retrieval_enabled=bool(flags.get("retrieval", retrieval_enabled)),
            rrf_k=int(flags.get("rrf_k", rrf_k)),
            query_strategy=str(flags.get("query_strategy", query_strategy)),
        )
        docs = retrieve(state["query"], index, cfg)
        s1, s2 = _top_scores(docs)
        return {
            "retrieved_docs": docs_to_state(docs),
            "stage": "retrieve",
            "top1_score": s1,
            "top2_score": s2,
        }

    def node_rerank(state: RAGState) -> dict[str, Any]:
        flags = state.get("config_flags") or {}
        enabled = bool(flags.get("rerank", rerank_enabled))
        docs = state_to_docs(state.get("retrieved_docs"))
        k = int(flags.get("top_k", top_k))
        reranked = rerank_documents(state["query"], docs, top_n=k, enabled=enabled)
        s1, s2 = _top_scores(reranked)
        # Prefer rerank scores when present
        out: dict[str, Any] = {
            "reranked_docs": docs_to_state(reranked),
            "stage": "rerank",
        }
        if reranked:
            out["top1_score"] = s1
            out["top2_score"] = s2
        return out

    def node_generate(state: RAGState) -> dict[str, Any]:
        flags = state.get("config_flags") or {}
        abstain_mode = str(flags.get("abstain", abstain) or "none")
        med = flags.get("median_dev", median_dev)
        mar = float(flags.get("margin", margin))

        docs = state_to_docs(state.get("reranked_docs") or state.get("retrieved_docs"))
        s1 = float(state.get("top1_score") or 0.0)
        s2 = float(state.get("top2_score") or 0.0)
        if not docs:
            s1, s2 = 0.0, 0.0
        else:
            # recompute from docs if needed
            ts1, ts2 = _top_scores(docs)
            if ts1 or ts2:
                s1, s2 = ts1, ts2

        if abstain_mode.startswith("margin") and med is not None:
            if (s1 - s2) < mar and s1 < float(med):
                return {
                    "answer": ABSTAIN_MARK,
                    "source_chunk_ids": [],
                    "stage": "generate_abstain",
                    "abstained": True,
                }

        result = extractive_answer(state["query"], docs)
        return {
            "answer": result["answer"],
            "source_chunk_ids": result["source_chunk_ids"],
            "stage": "generate",
            "abstained": False,
        }

    def node_finalize(state: RAGState) -> dict[str, Any]:
        stage = "finalize"
        if state.get("abstained") or (state.get("answer") or "").strip() == ABSTAIN_MARK:
            stage = "finalize_abstain"
        return {"stage": stage}

    g = StateGraph(RAGState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("rerank", node_rerank)
    g.add_node("generate", node_generate)
    g.add_node("finalize", node_finalize)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "finalize")
    g.add_edge("finalize", END)

    kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if interrupt_after:
        kwargs["interrupt_after"] = interrupt_after
    return g.compile(**kwargs)


def get_sqlite_checkpointer(db_path: str | Path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=60.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        conn.commit()
    except Exception:
        pass
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def close_checkpointer(checkpointer) -> None:
    conn = getattr(checkpointer, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def get_memory_checkpointer():
    return MemorySaver()


def run_pipeline(
    query: str,
    index: RAGIndex,
    *,
    top_k: int = 4,
    threshold: float | None = None,
    retriever: str = "dense",
    retrieval_enabled: bool = True,
    rerank_enabled: bool = True,
    rrf_k: int = 60,
    query_strategy: str = "raw",
    abstain: str = "none",
    median_dev: float | None = None,
    margin: float = 0.05,
    thread_id: str | None = None,
    checkpointer=None,
) -> dict[str, Any]:
    """Run full graph once; optional thread_id for checkpointed runs."""
    app = build_graph(
        index,
        top_k=top_k,
        threshold=threshold,
        retriever=retriever,
        retrieval_enabled=retrieval_enabled,
        rerank_enabled=rerank_enabled,
        rrf_k=rrf_k,
        query_strategy=query_strategy,
        abstain=abstain,
        median_dev=median_dev,
        margin=margin,
        checkpointer=checkpointer,
    )
    flags = {
        "top_k": top_k,
        "threshold": threshold,
        "retriever": retriever,
        "retrieval": retrieval_enabled,
        "rerank": rerank_enabled,
        "rrf_k": rrf_k,
        "query_strategy": query_strategy,
        "abstain": abstain,
        "median_dev": median_dev,
        "margin": margin,
    }
    inv_cfg: dict[str, Any] = {}
    if thread_id and checkpointer is not None:
        inv_cfg["configurable"] = {"thread_id": thread_id}
    result = app.invoke(
        {"query": query, "config_flags": flags, "stage": "start"},
        config=inv_cfg or None,
    )
    return dict(result)
