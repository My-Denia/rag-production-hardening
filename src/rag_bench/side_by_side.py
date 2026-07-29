"""Side-by-side v1 vs v2 comparison tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rag_bench.config_load import RESULTS_DIR, ensure_dirs


def build_side_by_side(
    v1: dict[str, Any],
    v2: dict[str, Any],
    *,
    disc_v1: dict[str, Any] | None = None,
    disc_v2: dict[str, Any] | None = None,
    difficulty: dict[str, Any] | None = None,
    embeddings_contrast: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    keys = sorted(set(v1.keys()) | set(v2.keys()))
    for k in keys:
        rows.append({"metric": k, "v1": v1.get(k), "v2": v2.get(k)})
    fac_rows = []
    d1 = (disc_v1 or {}).get("factors") or {}
    d2 = (disc_v2 or {}).get("factors") or {}
    for fac in sorted(set(d1) | set(d2)):
        fac_rows.append(
            {
                "factor": fac,
                "v1_claim": (d1.get(fac) or {}).get("claim_label"),
                "v2_claim": (d2.get(fac) or {}).get("claim_label"),
            }
        )
    return {
        "metrics": rows,
        "factors": fac_rows,
        "difficulty": difficulty or {},
        "embeddings_contrast": embeddings_contrast or {},
        "v1_summary": v1,
        "v2_summary": v2,
    }


def write_side_by_side(payload: dict[str, Any], results_dir: Path | None = None) -> tuple[Path, Path]:
    ensure_dirs()
    root = results_dir or RESULTS_DIR
    md_path = root / "side_by_side.md"
    csv_path = root / "side_by_side.csv"
    json_path = root / "side_by_side.json"

    lines = [
        "# Side-by-side: run_v1 vs run_v2",
        "",
        "## Metrics",
        "",
        "| metric | v1 | v2 |",
        "| --- | --- | --- |",
    ]
    for r in payload.get("metrics") or []:
        lines.append(f"| {r['metric']} | {r['v1']} | {r['v2']} |")
    lines += ["", "## Factor discriminability claims", "", "| factor | v1 | v2 |", "| --- | --- | --- |"]
    for r in payload.get("factors") or []:
        lines.append(f"| {r['factor']} | {r.get('v1_claim')} | {r.get('v2_claim')} |")
    diff = payload.get("difficulty") or {}
    if diff:
        lines += [
            "",
            "## Difficulty gates",
            "",
            f"- jaccard_v1: {diff.get('jaccard_v1')}",
            f"- jaccard_v2: {diff.get('jaccard_v2')}",
            f"- jaccard_drop: {diff.get('jaccard_drop')}",
            f"- hash_recall_v1: {diff.get('hash_recall_v1')}",
            f"- hash_recall_v2: {diff.get('hash_recall_v2')}",
            f"- hash_recall_drop: {diff.get('hash_recall_drop')}",
            f"- gates_passed: {diff.get('passed')}",
            "",
        ]
    emb = payload.get("embeddings_contrast") or {}
    if emb:
        lines += ["## Embeddings contrast", "", "```json", json.dumps(emb, indent=2), "```", ""]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "v1", "v2"])
        w.writeheader()
        for r in payload.get("metrics") or []:
            w.writerow(r)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return md_path, csv_path
