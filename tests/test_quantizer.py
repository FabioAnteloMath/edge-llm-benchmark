"""Tests for the quantizer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from edge_llm_bench.quantizer import (
    QuantizationError,
    _is_gguf_format,
    ensure_quantized,
)


def test_is_gguf_format_detection() -> None:
    assert _is_gguf_format("Q4_K_M") is True
    assert _is_gguf_format("Q5_K_M") is True
    assert _is_gguf_format("Q8_0") is True
    assert _is_gguf_format("AWQ") is False
    assert _is_gguf_format("GPTQ") is False
    assert _is_gguf_format("MLX-4bit") is False


def test_ensure_quantized_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(QuantizationError, match="Unknown target_format"):
        ensure_quantized(
            source_path=tmp_path,
            target_format="MYSTERY-99",
            output_dir=tmp_path / "out",
        )


def test_ensure_quantized_awq_without_install(tmp_path: Path) -> None:
    with patch.dict("sys.modules", {"autoawq": None}):
        with pytest.raises(QuantizationError, match="autoawq"):
            ensure_quantized(tmp_path, "AWQ", tmp_path / "out")


def test_ensure_quantized_gptq_without_install(tmp_path: Path) -> None:
    with patch.dict("sys.modules", {"auto_gptq": None}):
        with pytest.raises(QuantizationError, match="auto-gptq"):
            ensure_quantized(tmp_path, "GPTQ", tmp_path / "out")


def test_ensure_quantized_mlx_without_install(tmp_path: Path) -> None:
    with patch.dict("sys.modules", {"mlx_lm": None}):
        with pytest.raises(QuantizationError, match="mlx-lm"):
            ensure_quantized(tmp_path, "MLX-4bit", tmp_path / "out")


def test_ensure_quantized_llamacpp_no_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Provide a fake gguf source so the function gets past the source check.
    src = tmp_path / "src"
    src.mkdir()
    (src / "fake.gguf").write_bytes(b"GGUF\x00\x00\x00\x03fake")
    monkeypatch.setattr("shutil.which", lambda x: None)
    with pytest.raises(QuantizationError, match="llama-quantize"):
        ensure_quantized(src, "Q4_K_M", tmp_path / "out")


def test_ensure_quantized_llamacpp_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the output file exists, no subprocess is invoked."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "fake.gguf").write_bytes(b"GGUF")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    expected = out_dir / "src.q4_k_m.gguf"
    expected.write_bytes(b"already-here")

    # Patch shutil.which to return a fake binary; should NOT be called.
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/fake-binary")

    result = ensure_quantized(src, "Q4_K_M", out_dir)
    assert result == expected
