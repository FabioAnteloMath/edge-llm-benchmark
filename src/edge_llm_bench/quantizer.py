"""Quantizer — convert FP16/BF16 weights to a target quantization format.

Dispatches by format family:

* ``Q*_K_*``, ``Q*_0`` → llama.cpp ``convert.py`` + ``quantize`` binary
* ``AWQ`` → ``autoawq`` (CUDA only)
* ``GPTQ`` → ``auto-gptq`` (CUDA only)
* ``MLX-*`` → ``mlx_lm.convert`` (Apple Silicon only)

All backends are *optional imports*; missing backend → actionable error pointing
to the ``pip install`` command. Idempotent: skips when the output artifact
already exists.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .hardware_profiler import HardwareProfile


class QuantizationError(RuntimeError):
    """Raised on any quantization failure with an actionable message."""


# ---------------------------------------------------------------------------
# Disk check
# ---------------------------------------------------------------------------


def _check_disk(output_dir: Path, source_size_gb: float, multiplier: float = 1.5) -> None:
    """Estimate peak disk need (source + quantized output + working)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    need_gb = source_size_gb * multiplier + 1.0
    free_gb = shutil.disk_usage(output_dir).free / (1024**3)
    if free_gb < need_gb:
        raise QuantizationError(
            f"Need ~{need_gb:.0f} GB free at {output_dir} for quantization, "
            f"only {free_gb:.1f} GB available. Free disk space and retry."
        )


def _dir_size_gb(path: Path) -> float:
    """Estimate a directory's size in GB (cheap, may overshoot for symlinks)."""
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total / (1024**3)


# ---------------------------------------------------------------------------
# Per-format implementations
# ---------------------------------------------------------------------------


def _ensure_llamacpp(source_path: Path, target_format: str, output_dir: Path) -> Path:
    """Run llama.cpp's quantize binary (assumes a GGUF source)."""
    src_gguf = source_path / f"{source_path.name}.gguf"
    if not src_gguf.exists():
        candidates = list(source_path.glob("*.gguf"))
        if not candidates:
            raise QuantizationError(
                f"No GGUF source found in {source_path}. "
                f"llama.cpp quantization requires an existing GGUF (typically F16 or F32)."
            )
        src_gguf = candidates[0]

    out_path = output_dir / f"{source_path.name}.{target_format.lower()}.gguf"
    if out_path.exists():
        return out_path

    binary = shutil.which("llama-quantize") or shutil.which("quantize")
    if binary is None:
        raise QuantizationError(
            "llama.cpp `llama-quantize` binary not found on PATH.\n"
            "  Fix: build llama.cpp (`git clone https://github.com/ggerganov/llama.cpp "
            "&& cd llama.cpp && make`) and add to PATH."
        )

    cmd = [binary, str(src_gguf), str(out_path), target_format]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise QuantizationError(
            f"llama-quantize failed (exit {proc.returncode}).\n"
            f"  stdout: {proc.stdout[-500:]}\n  stderr: {proc.stderr[-500:]}"
        )
    return out_path


def _ensure_awq(source_path: Path, target_format: str, output_dir: Path) -> Path:
    """Quantize to AWQ via autoawq (CUDA only)."""
    try:
        import autoawq  # noqa: F401
    except ImportError as exc:
        raise QuantizationError(
            "AWQ requires `autoawq`. Install with: pip install autoawq"
        ) from exc
    raise QuantizationError(
        "AWQ quantization runtime path is not yet implemented in v0.1; "
        "please use the Ollama pipeline which handles AWQ internally."
    )


def _ensure_gptq(source_path: Path, target_format: str, output_dir: Path) -> Path:
    """Quantize to GPTQ via auto-gptq (CUDA only)."""
    try:
        import auto_gptq  # noqa: F401
    except ImportError as exc:
        raise QuantizationError(
            "GPTQ requires `auto-gptq`. Install with: pip install auto-gptq"
        ) from exc
    raise QuantizationError(
        "GPTQ quantization runtime path is not yet implemented in v0.1; "
        "please use the Ollama pipeline which handles GPTQ internally."
    )


def _ensure_mlx(source_path: Path, target_format: str, output_dir: Path) -> Path:
    """Convert to MLX format (Apple Silicon only)."""
    try:
        from mlx_lm import convert
    except ImportError as exc:
        raise QuantizationError(
            "MLX quantization requires `mlx-lm`. Install with: pip install mlx-lm"
        ) from exc
    bits = 4 if target_format.endswith("4bit") else 8
    out_path = output_dir / f"{source_path.name}-mlx-{bits}bit"
    if out_path.exists():
        return out_path
    try:
        convert(source_path, out_path, quantize=True, bits=bits)
    except Exception as exc:
        raise QuantizationError(f"mlx_lm.convert failed: {exc}") from exc
    return out_path


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_FORMAT_DISPATCH = {
    "AWQ": _ensure_awq,
    "GPTQ": _ensure_gptq,
    "MLX-4bit": _ensure_mlx,
    "MLX-8bit": _ensure_mlx,
}


def _is_gguf_format(fmt: str) -> bool:
    return fmt.startswith("Q") and ("_K" in fmt or fmt.endswith("_0"))


def ensure_quantized(
    source_path: Path,
    target_format: str,
    output_dir: Path,
    backend: str = "auto",
    profile: HardwareProfile | None = None,
) -> Path:
    """Ensure a quantized artifact exists at ``output_dir``. Idempotent.

    Parameters
    ----------
    source_path:
        Directory containing the FP16/BF16 weights (or pre-quant GGUF).
    target_format:
        One of ``"Q4_K_M"``, ``"Q5_K_M"``, ``"Q8_0"``, ``"AWQ"``, ``"GPTQ"``,
        ``"MLX-4bit"``, ``"MLX-8bit"``.
    output_dir:
        Destination directory for the quantized artifact.
    backend:
        ``"auto"`` (default) picks based on format; or force a specific backend.
    profile:
        Optional :class:`HardwareProfile` for hardware-aware backend selection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_disk(output_dir, _dir_size_gb(source_path))

    if _is_gguf_format(target_format):
        return _ensure_llamacpp(source_path, target_format, output_dir)

    if target_format in _FORMAT_DISPATCH:
        return _FORMAT_DISPATCH[target_format](source_path, target_format, output_dir)

    raise QuantizationError(
        f"Unknown target_format '{target_format}'. Supported: Q4_K_M, Q5_K_M, Q8_0, "
        f"AWQ, GPTQ, MLX-4bit, MLX-8bit."
    )


__all__ = ["ensure_quantized", "QuantizationError"]
