"""Tests for the inference backend abstraction."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from edge_llm_bench.inference import GenerateResult, InferenceBackend
from edge_llm_bench.inference.base import BackendRegistry as BaseRegistry

# ---------------------------------------------------------------------------
# Mock backend implementing the protocol
# ---------------------------------------------------------------------------


@dataclass
class MockBackend:
    """Minimal stand-in used to validate protocol conformance and registry."""

    name: str = "mock"
    loaded: bool = False
    warmed: bool = False
    closed: bool = False
    last_prompt: str = ""
    last_max_tokens: int = 0

    def load(self, model_path: str, format: str = "", ctx_size: int = 4096, **kw) -> None:
        self.loaded = True

    def warmup(self) -> None:
        self.warmed = True

    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.0
    ) -> GenerateResult:
        self.last_prompt = prompt
        self.last_max_tokens = max_tokens
        return GenerateResult(
            text=f"echo:{prompt}",
            ttft_ms=42.0,
            total_ms=200.0,
            output_tokens=4,
            tokens_per_s=20.0,
            ram_peak_gb=1.5,
        )

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mock_backend_conforms_to_protocol() -> None:
    """Protocol is structural; no inheritance required."""
    backend: InferenceBackend = MockBackend()
    assert isinstance(backend, InferenceBackend)


def test_generate_result_defaults() -> None:
    res = GenerateResult(text="hi")
    assert res.text == "hi"
    assert res.output_tokens == 0
    assert res.ram_peak_gb == 0.0
    assert res.vram_peak_gb is None


def test_backend_registry_register_and_get() -> None:
    BaseRegistry.register("mock-test", MockBackend)
    cls = BaseRegistry.get("mock-test")
    assert cls is MockBackend


def test_backend_registry_unknown_raises_actionable() -> None:
    # Save & restore registry to avoid polluting other tests.
    saved = BaseRegistry._registry.copy()
    try:
        BaseRegistry._registry.clear()
        with pytest.raises(KeyError) as excinfo:
            BaseRegistry.get("does-not-exist")
        assert "Available" in str(excinfo.value)
    finally:
        BaseRegistry._registry.update(saved)


def test_backend_registry_available_lists_known() -> None:
    BaseRegistry.register("mock-1", MockBackend)
    BaseRegistry.register("mock-2", MockBackend)
    names = BaseRegistry.available()
    assert "mock-1" in names and "mock-2" in names


def test_mock_backend_full_lifecycle() -> None:
    b: InferenceBackend = MockBackend()
    b.load("any-model", "Q4_K_M", 4096)
    b.warmup()
    result = b.generate("hello world", max_tokens=10)
    assert result.text == "echo:hello world"
    assert b.last_max_tokens == 10
    b.close()
    assert b.closed is True
