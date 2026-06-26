"""Tests for the hardware profiler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from edge_llm_bench.hardware_profiler import (
    HardwareProfile,
    _detect_apple_metal,
    _detect_disk_free_gb,
    _detect_nvidia_vram,
    _detect_os,
    can_run,
    profile_hardware,
    suggested_quant_max,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_profile() -> HardwareProfile:
    return HardwareProfile(
        os="linux",
        arch="x86_64",
        cpu_cores=8,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        gpu_type="none",
        vram_gb=None,
        disk_free_gb=200.0,
        python_version="3.11.0",
    )


@pytest.fixture
def large_profile() -> HardwareProfile:
    return HardwareProfile(
        os="darwin",
        arch="arm64",
        cpu_cores=12,
        ram_total_gb=36.0,
        ram_available_gb=24.0,
        gpu_type="apple",
        vram_gb=None,
        disk_free_gb=500.0,
        python_version="3.11.0",
    )


@pytest.fixture
def tiny_profile() -> HardwareProfile:
    return HardwareProfile(
        os="linux",
        arch="x86_64",
        cpu_cores=4,
        ram_total_gb=8.0,
        ram_available_gb=4.0,
        gpu_type="none",
        vram_gb=None,
        disk_free_gb=100.0,
        python_version="3.11.0",
    )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def test_detect_os_returns_known_token() -> None:
    os_name, arch = _detect_os()
    assert os_name in {"darwin", "linux", "windows"}
    assert isinstance(arch, str) and arch


def test_detect_disk_free_gb_is_positive() -> None:
    val = _detect_disk_free_gb()
    assert val > 0


def test_detect_apple_metal_on_linux_is_false() -> None:
    with patch("edge_llm_bench.hardware_profiler._detect_os", return_value=("linux", "x86_64")):
        assert _detect_apple_metal() is False


def test_detect_apple_metal_on_arm64_darwin_is_true() -> None:
    with patch("edge_llm_bench.hardware_profiler._detect_os", return_value=("darwin", "arm64")):
        assert _detect_apple_metal() is True


def test_detect_nvidia_vram_without_pynvml() -> None:
    with patch("edge_llm_bench.hardware_profiler._HAS_PYNVML", False):
        assert _detect_nvidia_vram() is None


def test_detect_nvidia_vram_returns_total() -> None:
    fake_handle = MagicMock()
    fake_handle.total = 24 * (1024**3)  # 24 GB
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlDeviceGetCount.return_value = 1
    fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = fake_handle
    fake_pynvml.nvmlDeviceGetMemoryInfo.return_value = fake_handle

    with (
        patch("edge_llm_bench.hardware_profiler._HAS_PYNVML", True),
        patch("edge_llm_bench.hardware_profiler.pynvml", fake_pynvml),
    ):
        vram = _detect_nvidia_vram()
    assert vram == 24.0


# ---------------------------------------------------------------------------
# can_run
# ---------------------------------------------------------------------------


def test_can_run_within_available_ram(small_profile: HardwareProfile) -> None:
    # 8 GB available, 1 GB buffer → max 7 GB usable
    assert can_run(small_profile, 7.0) is True


def test_can_run_exceeds_available_ram(small_profile: HardwareProfile) -> None:
    assert can_run(small_profile, 8.0) is False
    assert can_run(small_profile, 20.0) is False


def test_can_run_zero_requirement(small_profile: HardwareProfile) -> None:
    assert can_run(small_profile, 0) is True


def test_can_run_reserves_1gb_buffer(small_profile: HardwareProfile) -> None:
    # 8 GB available; 7.5 GB should still be fine (8 - 1 = 7)
    assert can_run(small_profile, 7.5) is False


# ---------------------------------------------------------------------------
# suggested_quant_max
# ---------------------------------------------------------------------------


def test_suggested_quant_high_ram(large_profile: HardwareProfile) -> None:
    # large_profile has ram_available_gb=24 → 16 GB tier (≥16, <32)
    # → head is Q4_K_M, tail is Q3_K_M.
    quants = suggested_quant_max(large_profile)
    assert quants[0] == "Q4_K_M"
    assert "Q3_K_M" in quants


def test_suggested_quant_very_high_ram() -> None:
    profile = HardwareProfile(
        os="darwin",
        arch="arm64",
        cpu_cores=12,
        ram_total_gb=128.0,
        ram_available_gb=96.0,
        gpu_type="apple",
        vram_gb=None,
        disk_free_gb=500.0,
        python_version="3.11.0",
    )
    quants = suggested_quant_max(profile)
    # 96 GB available → 64 GB tier → head is Q8_0 / Q6_K / Q5_K_M
    assert quants[0] in {"Q8_0", "Q6_K", "Q5_K_M"}


def test_suggested_quant_mid_ram(small_profile: HardwareProfile) -> None:
    quants = suggested_quant_max(small_profile)
    # 8 GB available lands in the 8.0 tier
    assert quants[0] in {"Q3_K_M", "Q2_K"}


def test_suggested_quant_low_ram(tiny_profile: HardwareProfile) -> None:
    quants = suggested_quant_max(tiny_profile)
    assert quants == ["Q2_K"]


# ---------------------------------------------------------------------------
# profile_hardware (smoke)
# ---------------------------------------------------------------------------


def test_profile_hardware_runs_on_real_machine() -> None:
    """Smoke: real detection on the current machine must not raise."""
    profile = profile_hardware()
    assert profile.os in {"darwin", "linux", "windows"}
    assert profile.cpu_cores >= 1
    assert profile.ram_total_gb > 0
    assert profile.disk_free_gb > 0
    assert profile.python_version
    # gpu_type is always one of the known tokens
    assert profile.gpu_type in {"nvidia", "apple", "amd", "none"}
