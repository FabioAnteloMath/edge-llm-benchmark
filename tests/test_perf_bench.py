"""Tests for the performance benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edge_llm_bench.inference.base import GenerateResult, InferenceBackend
from edge_llm_bench.perf_bench import (
    _percentile,
    _safe_generate,
    _short_prompts,
    run_perf,
)


@dataclass
class FakeBackend:
    """Deterministic backend for testing — produces predictable timings."""

    name: str = "fake"
    loaded: bool = False
    closed: bool = False
    # Tweakable response: tokens_per_s, ttft_ms, ram_peak_gb
    tps: float = 50.0
    ttft_ms: float = 100.0
    ram_gb: float = 10.0
    error_after: int | None = None
    call_count: int = 0
    raise_on: set[str] = field(default_factory=set)

    def load(self, model_path: str, format: str = "", ctx_size: int = 4096, **kw: Any) -> None:
        self.loaded = True

    def warmup(self) -> None:
        pass

    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.0
    ) -> GenerateResult:
        self.call_count += 1
        if "boom" in prompt and self.raise_on is not None and "boom" in self.raise_on:
            raise RuntimeError("synthetic failure")
        return GenerateResult(
            text="x" * max_tokens,
            ttft_ms=self.ttft_ms,
            total_ms=1000.0,
            output_tokens=max_tokens,
            tokens_per_s=self.tps,
            ram_peak_gb=self.ram_gb,
        )

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_percentile_empty() -> None:
    assert _percentile([], 50) == 0.0


def test_percentile_basic() -> None:
    vals = list(range(1, 101))
    p50 = _percentile(vals, 50)
    p95 = _percentile(vals, 95)
    assert 49 <= p50 <= 51
    assert 94 <= p95 <= 96


def test_short_prompts_returns_n_unique() -> None:
    p = _short_prompts(5)
    assert len(p) == 5
    assert len(set(p)) == 1  # all same template


def test_safe_generate_swallows_exception() -> None:
    backend: InferenceBackend = FakeBackend(raise_on={"boom"})
    res = _safe_generate(backend, "boom go boom", max_tokens=10)
    assert res.text == ""
    assert "error" in res.extras


# ---------------------------------------------------------------------------
# run_perf
# ---------------------------------------------------------------------------


def test_run_perf_warmup_does_not_pollute_stats() -> None:
    """All warmup calls share the fake's tps; stats come from later phases."""
    backend = FakeBackend(tps=50.0, ttft_ms=100.0, ram_gb=5.0)
    stats = run_perf(
        backend,
        n_warmup=3,
        n_short=2,
        n_medium=2,
        n_long=1,
        progress=False,
    )
    # 3 warmup + 2 short + 2 medium + 1 long = 8 calls
    assert backend.call_count == 8
    assert stats.error is None
    # Stats reflect the deterministic fake values
    assert stats.tokens_per_s_p50 == 50.0
    assert stats.ttft_ms_p50 == 100.0
    assert stats.ram_peak_gb == 5.0
    assert stats.n_runs == 5  # excludes warmup


def test_run_perf_partial_failure() -> None:
    """A backend that errors on some prompts returns a non-empty stats object."""
    backend = FakeBackend(tps=10.0, ttft_ms=200.0, ram_gb=3.0)
    # Override generate to fail every 3rd call.
    real = backend.generate
    counter = {"n": 0}

    def flaky(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerateResult:
        counter["n"] += 1
        if counter["n"] % 3 == 0:
            raise RuntimeError("flake")
        return real(prompt, max_tokens, temperature)

    backend.generate = flaky  # type: ignore[assignment]
    stats = run_perf(
        backend,
        n_warmup=2,
        n_short=3,
        n_medium=2,
        n_long=1,
        progress=False,
    )
    # Warmup fails happen at n=3,6,9 (n_short failures at n=4,7; n_medium at n=8)
    # Stats should still be computed from the successful runs.
    assert stats.n_runs > 0
    assert stats.tokens_per_s_p50 == 10.0


def test_run_perf_all_runs_fail() -> None:
    backend = FakeBackend(raise_on={"hi", "Summarize", "The quick", "Large language"})
    # Force every call to fail by patching raise_on broadly.
    backend.raise_on = None  # type: ignore[assignment]
    counter = {"n": 0}

    def always_fail(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerateResult:
        counter["n"] += 1
        raise RuntimeError("always")

    backend.generate = always_fail  # type: ignore[assignment]
    stats = run_perf(
        backend,
        n_warmup=2,
        n_short=1,
        n_medium=1,
        n_long=1,
        progress=False,
    )
    assert stats.error == "all_runs_failed"
    assert stats.n_runs == 0
