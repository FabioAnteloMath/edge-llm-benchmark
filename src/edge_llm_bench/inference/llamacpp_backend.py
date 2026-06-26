"""llama.cpp backend — wraps ``llama-cpp-python``.

Install with ``pip install -e ".[llamacpp]"`` (extra). Provides finer control
over ``n_ctx``, ``n_threads``, ``n_gpu_layers`` than Ollama.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from .base import BackendRegistry, GenerateResult, InferenceBackend


class LlamaCppBackend(InferenceBackend):
    name = "llama-cpp"

    def __init__(self, n_threads: int | None = None, track_ram: bool = True) -> None:
        try:
            from llama_cpp import Llama

            self._Llama = Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama.cpp backend requires `llama-cpp-python`. "
                'Install with: pip install -e ".[llamacpp]"'
            ) from exc
        self._n_threads = n_threads or os.cpu_count() or 4
        self._track_ram = track_ram
        self._llama: Any = None
        self._proc = psutil.Process(os.getpid())

    def load(
        self,
        model_path: str,
        format: str = "",  # noqa: ARG002 — determined by file
        ctx_size: int = 4096,
        **kwargs: Any,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"GGUF model file not found: {path}")
        n_gpu_layers = kwargs.pop("n_gpu_layers", -1)  # -1 = all layers on GPU
        self._llama = self._Llama(
            model_path=str(path),
            n_ctx=ctx_size,
            n_threads=self._n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def warmup(self) -> None:
        if self._llama is None:
            return
        for _ in range(2):
            self._llama.create_completion("hi", max_tokens=1, temperature=0.0)

    def close(self) -> None:
        if self._llama is not None:
            try:
                self._llama.close()
            except Exception:  # noqa: BLE001 — best effort
                pass
            self._llama = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GenerateResult:
        if self._llama is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        sampler = _RamSampler(self._proc) if self._track_ram else None
        if sampler:
            sampler.start()

        t0 = time.perf_counter()
        # llama-cpp-python does not natively stream-completion with TTFT.
        # We split into a 1-token "prefill" probe then a normal completion to
        # get an approximate TTFT.
        probe = self._llama.create_completion(prompt, max_tokens=1, temperature=temperature)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        n_probe = len(probe["choices"][0]["text"]) if probe.get("choices") else 0

        remaining = max(0, max_tokens - n_probe)
        completion = self._llama.create_completion(
            prompt, max_tokens=remaining, temperature=temperature
        )
        total_ms = (time.perf_counter() - t0) * 1000.0

        text = completion["choices"][0]["text"] if completion.get("choices") else ""
        # Approximate token count from the llama-cpp-python usage block.
        usage = completion.get("usage", {}) or {}
        output_tokens = int(usage.get("completion_tokens", len(text.split())))

        ram_peak = sampler.stop() if sampler else 0.0
        tokens_per_s = (output_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        return GenerateResult(
            text=text,
            ttft_ms=round(ttft_ms, 2),
            total_ms=round(total_ms, 2),
            output_tokens=output_tokens,
            tokens_per_s=round(tokens_per_s, 3),
            ram_peak_gb=ram_peak,
        )


class _RamSampler:
    """Identical to the Ollama backend's sampler."""

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


# Register on import — only if llama-cpp-python is installed.
try:
    import llama_cpp  # noqa: F401

    BackendRegistry.register("llama-cpp", LlamaCppBackend)
except ImportError:
    pass
