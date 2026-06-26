"""Tests for the quality benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from edge_llm_bench.inference.base import GenerateResult, InferenceBackend
from edge_llm_bench.quality_bench import (
    _check_ifeval_response,
    _extract_mmlu_answer,
    _extract_python_code,
    _format_mmlu_prompt,
    run_mmlu,
    run_quality,
)

# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


@dataclass
class StaticBackend:
    """Returns a fixed response string for any prompt."""

    name: str = "static"
    response: str = "A"
    raise_on_empty: bool = False
    call_count: int = 0

    def load(self, model_path: str, format: str = "", ctx_size: int = 4096, **kw: Any) -> None:
        pass

    def warmup(self) -> None:
        pass

    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.0
    ) -> GenerateResult:
        self.call_count += 1
        if self.raise_on_empty:
            raise RuntimeError("synthetic")
        return GenerateResult(
            text=self.response,
            ttft_ms=10,
            total_ms=200,
            output_tokens=max_tokens,
            tokens_per_s=20.0,
        )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Prompt formatting & answer extraction
# ---------------------------------------------------------------------------


def test_format_mmlu_prompt_includes_letters() -> None:
    p = _format_mmlu_prompt("What is 2+2?", ["3", "4", "5", "6"])
    assert "A) 3" in p
    assert "D) 6" in p
    assert "Answer:" in p


def test_extract_mmlu_answer_basic() -> None:
    assert _extract_mmlu_answer("The answer is B.") == "B"
    assert _extract_mmlu_answer("\n\nA") == "A"
    assert _extract_mmlu_answer("I don't know") is None
    assert _extract_mmlu_answer("a") == "A"


def test_extract_python_code() -> None:
    text = "Here is the code:\n```python\ndef f(x): return x+1\n```\nDone."
    assert _extract_python_code("def f(x):", text) == "def f(x): return x+1\n"

    text_no_fence = "def f(x): return x+1"
    assert _extract_python_code("def f(x):", text_no_fence) == "def f(x): return x+1"


def test_check_ifeval_response_lowercase() -> None:
    assert _check_ifeval_response("respond in lowercase", "this is lowercase") is True
    assert _check_ifeval_response("respond in lowercase", "This is Mixed") is False


def test_check_ifeval_response_json() -> None:
    assert _check_ifeval_response("return json", '{"key": "value"}') is True
    assert _check_ifeval_response("return json", "not json at all") is False


# ---------------------------------------------------------------------------
# run_mmlu on synthetic data
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_mmlu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = [
        {"question": "Q1", "choices": ["a1", "a2", "a3", "a4"], "answer": 0, "subject": "x"},
        {"question": "Q2", "choices": ["b1", "b2", "b3", "b4"], "answer": 1, "subject": "x"},
        {"question": "Q3", "choices": ["c1", "c2", "c3", "c4"], "answer": 2, "subject": "x"},
        {"question": "Q4", "choices": ["d1", "d2", "d3", "d4"], "answer": 3, "subject": "x"},
    ]
    path = tmp_path / "mmlu_subset.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr("edge_llm_bench.quality_bench.DATASETS_DIR", tmp_path)
    return path


def test_run_mmlu_all_correct(tiny_mmlu: Path) -> None:
    """Backend always returns 'A'; fixture has answers [0,1,2,3].
    Only the first matches → accuracy is 1/4 = 0.25."""
    backend: InferenceBackend = StaticBackend(response="A")
    result = run_mmlu(backend, n=4)
    assert result.n_mmlu == 4
    assert result.mmlu_acc == 0.25


def test_run_mmlu_half_correct(tiny_mmlu: Path) -> None:
    """Backend always returns 'B'; answers [0,1,2,3] → only second matches.
    Accuracy = 0.25."""
    backend: InferenceBackend = StaticBackend(response="B")
    result = run_mmlu(backend, n=4)
    assert result.n_mmlu == 4
    # Only one of four rows has answer=1 ("B")
    assert result.mmlu_acc == 0.25


def test_run_mmlu_missing_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr("edge_llm_bench.quality_bench.DATASETS_DIR", empty)
    backend: InferenceBackend = StaticBackend()
    r = run_mmlu(backend)
    assert r.error and "missing" in r.error


# ---------------------------------------------------------------------------
# run_quality orchestrator
# ---------------------------------------------------------------------------


def test_run_quality_unknown_suite_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    backend: InferenceBackend = StaticBackend()
    result = run_quality(backend, suite="nonexistent")
    assert result.mmlu_acc is None
    assert result.ifeval_strict_acc is None
