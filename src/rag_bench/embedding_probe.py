"""Mandatory MiniLM attempt + backend report for AC4."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from rag_bench.config_load import RESULTS_DIR, ensure_dirs


def probe_embedding_backends(
    *,
    try_install: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    ensure_dirs()
    report: dict[str, Any] = {
        "attempted_minilm": True,
        "minilm_success": False,
        "minilm_error": None,
        "install_commands": [],
        "install_ok": None,
        "tfidf_available": True,
        "hash_available": True,
        "primary_semantic_backend": None,
        "AC4_semantic": "unmet",
        "fallback_label": None,
        "notes": [],
    }

    # 1) Try import sentence_transformers
    try:
        import sentence_transformers  # noqa: F401

        st_ok = True
    except Exception as e:
        st_ok = False
        report["notes"].append(f"initial import failed: {type(e).__name__}: {e}")
        if try_install:
            cmd = [sys.executable, "-m", "pip", "install", "sentence-transformers", "--quiet"]
            report["install_commands"].append(" ".join(cmd))
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                report["install_ok"] = r.returncode == 0
                if r.returncode != 0:
                    report["notes"].append(
                        f"pip install failed rc={r.returncode}: {(r.stderr or r.stdout)[:500]}"
                    )
                else:
                    try:
                        import sentence_transformers  # noqa: F401

                        st_ok = True
                    except Exception as e2:
                        report["notes"].append(f"post-install import failed: {e2}")
            except Exception as e:
                report["install_ok"] = False
                report["notes"].append(f"pip install exception: {e}")

    if st_ok:
        from rag_bench.embeddings_minilm import try_load_minilm

        emb, mini_report = try_load_minilm()
        report["minilm_probe"] = mini_report
        if mini_report.get("success") and emb is not None:
            report["minilm_success"] = True
            report["primary_semantic_backend"] = "minilm"
            report["AC4_semantic"] = "met"
        else:
            report["minilm_error"] = mini_report.get("error")
            report["AC4_semantic"] = "unmet"
            report["fallback_label"] = "content_aware_lexical"
            report["notes"].append("MiniLM load failed after package available; using tfidf fallback")
    else:
        report["minilm_error"] = "sentence-transformers unavailable"
        report["AC4_semantic"] = "unmet"
        report["fallback_label"] = "content_aware_lexical"
        report["notes"].append("MiniLM unavailable; hash vs tfidf labeled content_aware_lexical only")

    # Always verify tfidf works
    try:
        from rag_bench.embeddings_tfidf import TfidfEmbeddings

        t = TfidfEmbeddings(dim=32)
        t.fit(["alpha beta", "beta gamma"])
        _ = t.embed_query("alpha")
        report["tfidf_available"] = True
    except Exception as e:
        report["tfidf_available"] = False
        report["notes"].append(f"tfidf failed: {e}")

    if write:
        path = RESULTS_DIR / "embedding_backend.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return report
