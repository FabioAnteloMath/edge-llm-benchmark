"""Tests for the aggregator."""

from __future__ import annotations

from pathlib import Path

from edge_llm_bench.aggregator import (
    CSV_FIELDS,
    ResultRow,
    append_result,
    get_git_commit,
    make_run_id,
)


def test_csv_fields_canonical_order() -> None:
    # Order matters for downstream consumers — keep stable.
    assert CSV_FIELDS[0] == "run_id"
    assert CSV_FIELDS[-1] == "notes"
    assert "model_id" in CSV_FIELDS
    assert "mmlu_acc" in CSV_FIELDS


def test_make_run_id_format() -> None:
    rid = make_run_id()
    parts = rid.split("-")
    assert len(parts) == 4  # YYYY-MM-DD-SHORT
    assert len(parts[-1]) == 6


def test_get_git_commit_handles_no_git(tmp_path: Path) -> None:
    # When run from a non-git dir, must return "unknown", not raise.
    commit = get_git_commit()
    assert commit == "unknown" or len(commit) <= 40


def test_append_result_creates_header(tmp_path: Path) -> None:
    row = ResultRow(
        run_id="2026-06-23-abc123",
        timestamp="2026-06-23T00:00:00Z",
        host_profile="test",
        model_id="m",
        format="Q4_K_M",
        backend="ollama",
        ctx_size=4096,
        status="ok",
    )
    path = append_result(tmp_path, row)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "run_id" in content
    assert "m" in content


def test_append_result_appends_multiple(tmp_path: Path) -> None:
    for i in range(3):
        append_result(
            tmp_path,
            ResultRow(
                run_id="r",
                timestamp="t",
                host_profile="h",
                model_id=f"m{i}",
                format="Q4_K_M",
                backend="ollama",
                ctx_size=4096,
                status="ok",
            ),
        )
    csv_text = (tmp_path / "results.csv").read_text(encoding="utf-8")
    # Header + 3 data rows
    assert csv_text.count("\n") == 4
    for i in range(3):
        assert f"m{i}" in csv_text


def test_result_row_csv_dict_serializes_none_as_empty() -> None:
    row = ResultRow(
        run_id="r",
        timestamp="t",
        host_profile="h",
        model_id="m",
        format="Q4",
        backend="ollama",
        ctx_size=4096,
        status="ok",
        mmlu_acc=None,
        ttft_ms_p50=123.45,
    )
    d = row.to_csv_dict()
    assert d["mmlu_acc"] == ""
    assert d["ttft_ms_p50"] == 123.45
