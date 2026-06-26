"""Inference backend abstraction.

Every backend (Ollama, llama.cpp, future vLLM, MLX, …) implements the same
:class:`InferenceBackend` Protocol so the orchestrator never needs to special-case
them. Backends register themselves in :data:`BackendRegistry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable


@dataclass
class GenerateResult:
    """Outcome of a single ``generate`` call."""

    text: str
    ttft_ms: float = 0.0  # time to first token
    total_ms: float = 0.0
    output_tokens: int = 0
    tokens_per_s: float = 0.0
    ram_peak_gb: float = 0.0
    vram_peak_gb: float | None = None
    gpu_util_pct: float | None = None
    prefill_tokens_per_s: float | None = None
    inter_token_jitter_ms: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InferenceBackend(Protocol):
    """Pluggable inference backend interface."""

    name: ClassVar[str]

    def load(
        self,
        model_path: str,
        format: str = "",
        ctx_size: int = 4096,
        **kwargs: Any,
    ) -> None: ...

    def warmup(self) -> None: ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GenerateResult: ...

    def close(self) -> None: ...


class BackendRegistry:
    """Lightweight backend lookup by name."""

    _registry: ClassVar[dict[str, type[InferenceBackend]]] = {}

    @classmethod
    def register(cls, name: str, backend_cls: type[InferenceBackend]) -> None:
        cls._registry[name] = backend_cls

    @classmethod
    def get(cls, name: str) -> type[InferenceBackend]:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise KeyError(
                f"Unknown backend '{name}'. Available: {available}. "
                f"Did you forget to import the backend module?"
            )
        return cls._registry[name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)


__all__ = ["BackendRegistry", "GenerateResult", "InferenceBackend"]
