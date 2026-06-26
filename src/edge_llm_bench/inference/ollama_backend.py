"""Ollama backend — default, simplest, ships the best DX.

Wraps the official ``ollama`` Python client. Tracks RAM via a background sampler.
GPU utilization is read from ``pynvml`` if available, else ``None``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import psutil

from .base import BackendRegistry, GenerateResult, InferenceBackend


class OllamaBackend(InferenceBackend):
    name = "ollama"

    def __init__(
        self,
        host: str | None = None,
        keep_alive: str = "5m",
        track_ram: bool = True,
    ) -> None:
        # Import lazily so the package can be imported even if ollama is missing.
        try:
            import ollama as _ollama

            self._ollama = _ollama
        except ImportError as exc:  # pragma: no cover - exercised on bad installs
            raise RuntimeError(
                "Ollama backend requires the `ollama` Python package. "
                "Install with: pip install ollama"
            ) from exc

        self._client = self._ollama.Client(host=host or os.environ.get("OLLAMA_HOST"))
        self._keep_alive = keep_alive
        self._track_ram = track_ram
        self._model_tag: str | None = None
        self._proc = psutil.Process(os.getpid())

    # -- lifecycle ---------------------------------------------------------
    def load(
        self,
        model_path: str,
        format: str = "",  # noqa: ARG002 — Ollama derives format from tag
        ctx_size: int = 4096,
        **kwargs: Any,
    ) -> None:
        """``model_path`` here is an Ollama tag, e.g. ``"phi4:14b-q8_0"``.

        We pull it if missing, then warm the model into memory with options for
        ``num_ctx``. The tag is what ``generate()`` then references.
        """
        tag = model_path
        self._model_tag = tag

        # Show + pull. ``show`` is a cheap existence check.
        try:
            self._client.show(tag)
        except Exception:  # noqa: BLE001 — model not present, try to pull
            self._client.pull(tag)

        # Warm by issuing a tiny generation. This loads weights into the daemon.
        warm = self._client.generate(
            model=tag,
            prompt="hi",
            options={"num_ctx": ctx_size, "num_predict": 1},
            keep_alive=self._keep_alive,
        )
        _ = warm  # discard

    def warmup(self) -> None:
        if self._model_tag is None:
            return
        for _ in range(2):
            self._client.generate(
                model=self._model_tag,
                prompt="Hello",
                options={"num_predict": 1, "num_ctx": 512},
                keep_alive=self._keep_alive,
            )

    def close(self) -> None:
        if self._model_tag is None:
            return
        try:
            # Tell Ollama to unload — frees RAM/VRAM.
            self._client.generate(
                model=self._model_tag,
                prompt="",
                options={"num_predict": 0},
                keep_alive="0",
            )
        except Exception:  # noqa: BLE001 — best effort
            pass
        self._model_tag = None

    # -- generate ----------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GenerateResult:
        if self._model_tag is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        ram_sampler = _RamSampler(self._proc) if self._track_ram else None
        if ram_sampler:
            ram_sampler.start()

        t0 = time.perf_counter()
        ttft_ms = 0.0
        first_token_at: float | None = None
        text_parts: list[str] = []

        # Use streaming so we can measure TTFT.
        stream = self._client.generate(
            model=self._model_tag,
            prompt=prompt,
            stream=True,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
            keep_alive=self._keep_alive,
        )

        for chunk in stream:
            if first_token_at is None and chunk.get("response"):
                first_token_at = time.perf_counter()
                ttft_ms = (first_token_at - t0) * 1000.0
            text_parts.append(chunk.get("response", ""))
            if chunk.get("done"):
                break

        total_ms = (time.perf_counter() - t0) * 1000.0
        text = "".join(text_parts)
        output_tokens = max(1, len(text_parts))

        ram_peak = ram_sampler.stop() if ram_sampler else 0.0
        tokens_per_s = (output_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        return GenerateResult(
            text=text,
            ttft_ms=round(ttft_ms, 2),
            total_ms=round(total_ms, 2),
            output_tokens=output_tokens,
            tokens_per_s=round(tokens_per_s, 3),
            ram_peak_gb=ram_peak,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RamSampler:
    """Background thread that samples the current process RSS every 100 ms."""

    def __init__(self, proc: psutil.Process, interval_s: float = 0.1) -> None:
        self._proc = proc
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_gb = 0.0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return round(self._peak_rss_gb, 3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss_gb = self._proc.memory_info().rss / (1024**3)
                if rss_gb > self._peak_rss_gb:
                    self._peak_rss_gb = rss_gb
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop.wait(self._interval)


# Register on import.
BackendRegistry.register("ollama", OllamaBackend)
