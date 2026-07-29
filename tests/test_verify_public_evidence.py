"""Tests for schema-2 reproduction metadata and public evidence verifier.

Negative cases corrupt real loaded fixtures (not hard-coded expected digests of
the checker) and assert the shipped validate_artifacts / verify_public_evidence
paths fail.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from rag_bench.verify_public_evidence import (
    load_public_artifacts,
    validate_artifacts,
    verify_public_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_live() -> dict:
    data, errors = load_public_artifacts(ROOT)
    assert not errors, errors
    return data


def test_reproduction_schema_2_no_public_commit():
    repro_path = ROOT / "results" / "release" / "reproduction.json"
    assert repro_path.is_file()
    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    assert repro.get("schema") == 2
    assert "public_commit" not in repro
    binding = repro.get("commit_binding") or {}
    assert binding.get("type") == "release-attestation"
    assert str(binding.get("asset", "")).startswith("release-attestation-")
    assert binding.get("reason")
    assert repro.get("package") == "rag-bench"
    assert "run_all" in str(repro.get("recompute_command") or "")


def test_verifier_passes_on_committed_tree():
    report = verify_public_evidence(ROOT)
    assert report["ok"] is True, report.get("errors")
    assert report["n_errors"] == 0


def test_validate_artifacts_pass_on_live_data():
    data = _load_live()
    # freeze verify optional for pure path
    data["freeze_verify_result"] = {"ok": True}
    errors = validate_artifacts(ROOT, data)
    assert errors == [], errors


def test_negative_wrong_regression_hits():
    data = _load_live()
    data["freeze_verify_result"] = {"ok": True}
    bad = copy.deepcopy(data)
    bad["regression"]["recall_hits"] = 34
    errors = validate_artifacts(ROOT, bad)
    assert any("recall_hits" in e for e in errors)


def test_negative_restored_public_commit():
    data = _load_live()
    data["freeze_verify_result"] = {"ok": True}
    bad = copy.deepcopy(data)
    bad["reproduction"]["public_commit"] = "deadbeef" * 5
    bad["reproduction"]["schema"] = 1
    errors = validate_artifacts(ROOT, bad)
    assert any("public_commit" in e or "schema" in e for e in errors)


def test_negative_freeze_hash_change():
    data = _load_live()
    data["freeze_verify_result"] = {"ok": True}
    bad = copy.deepcopy(data)
    bad["freeze"]["regression_sha256"] = "0" * 64
    bad["freeze"]["ok"] = False
    errors = validate_artifacts(ROOT, bad)
    assert any("freeze" in e for e in errors)


def test_negative_cross_talk_nonzero():
    data = _load_live()
    data["freeze_verify_result"] = {"ok": True}
    bad = copy.deepcopy(data)
    bad["concurrency"]["cross_talk"] = 1
    errors = validate_artifacts(ROOT, bad)
    assert any("cross_talk" in e for e in errors)


def test_negative_missing_required_artifact(tmp_path: Path):
    # Copy minimal tree missing recovery
    for rel in (
        "results/regression_metrics.json",
        "results/holdout_confirmation.json",
        "results/concurrency_report.json",
        "results/freeze_report.json",
        "results/release/reproduction.json",
    ):
        src = ROOT / rel
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    # deliberately omit recovery_report.json
    data, load_errors = load_public_artifacts(tmp_path)
    assert any("recovery" in e for e in load_errors)
    assert data.get("recovery") is None


def test_negative_stale_attribution_rate():
    data = _load_live()
    data["freeze_verify_result"] = {"ok": True}
    bad = copy.deepcopy(data)
    bad["holdout"]["holdout_primary_metrics"]["attribution_rate"] = 0.909
    errors = validate_artifacts(ROOT, bad)
    assert any("attribution" in e.lower() or "0.909" in e or "stale" in e for e in errors)


def test_negative_recovery_stage_fail():
    data = _load_live()
    data["freeze_verify_result"] = {"ok": True}
    bad = copy.deepcopy(data)
    bad["recovery"]["stages"]["retrieve"]["ok"] = False
    bad["recovery"]["ok"] = False
    errors = validate_artifacts(ROOT, bad)
    assert any("retrieve" in e or "recovery" in e for e in errors)
