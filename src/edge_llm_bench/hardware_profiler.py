"""Hardware profiler — detect and report machine capabilities for LLM runnability.

Cross-platform (Windows, macOS, Linux). Detects RAM, CPU, GPU type (NVIDIA / Apple
Metal / none), disk free, and Python version. Never raises on missing optional deps;
returns ``gpu_type="none"`` instead.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import psutil

# Optional: NVIDIA. Imported lazily so absence is non-fatal.
try:
    import pynvml

    _HAS_PYNVML = True
except ImportError:  # pragma: no cover - exercised on non-CUDA machines
    pynvml = None
    _HAS_PYNVML = False


@dataclass
class HardwareProfile:
    """Snapshot of the machine's LLM-relevant capabilities."""

    os: str  # "darwin" | "linux" | "windows"
    arch: str  # "arm64" | "x86_64" | ...
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_type: str  # "nvidia" | "apple" | "amd" | "none"
    vram_gb: float | None
    disk_free_gb: float
    python_version: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _detect_os() -> tuple[str, str]:
    """Return ``(os, arch)`` normalized to the project vocabulary."""
    sys_platform = sys.platform
    if sys_platform.startswith("darwin"):
        os_name = "darwin"
    elif sys_platform.startswith("linux"):
        os_name = "linux"
    elif sys_platform.startswith("win"):
        os_name = "windows"
    else:
        os_name = sys_platform  # exotic; keep as-is

    arch = platform.machine().lower()
    return os_name, arch


def _detect_nvidia_vram() -> float | None:
    """Return total VRAM in GB across all NVIDIA GPUs, or None on failure."""
    if not _HAS_PYNVML:
        return None
    try:
        pynvml.nvmlInit()
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return None
            total_bytes = 0
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_bytes += info.total
            return round(total_bytes / (1024**3), 2)
        finally:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
    except Exception:  # noqa: BLE001 — any pynvml error → treat as no GPU
        return None


def _detect_apple_metal() -> bool:
    """Return True if the machine is Apple Silicon (Metal-capable)."""
    os_name, arch = _detect_os()
    if os_name != "darwin":
        return False
    if arch == "arm64":
        return True
    # Intel Macs do not have unified GPU memory accessible to PyTorch easily.
    return False


def _detect_disk_free_gb() -> float:
    """Return free disk space in GB for the current working directory's volume."""
    usage = shutil.disk_usage(os.getcwd())
    return round(usage.free / (1024**3), 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def profile_hardware() -> HardwareProfile:
    """Profile the current machine and return a :class:`HardwareProfile`."""
    os_name, arch = _detect_os()
    vm = psutil.virtual_memory()
    cpu_cores = psutil.cpu_count(logical=True) or 1

    # GPU detection — priority: NVIDIA > Apple Metal > none.
    gpu_type = "none"
    vram_gb: float | None = None
    extras: dict[str, Any] = {}

    vram = _detect_nvidia_vram()
    if vram is not None:
        gpu_type = "nvidia"
        vram_gb = vram
    elif _detect_apple_metal():
        gpu_type = "apple"
        # Apple Silicon uses unified memory; vram is reported via ram_total_gb.
        vram_gb = None
        extras["unified_memory"] = True

    return HardwareProfile(
        os=os_name,
        arch=arch,
        cpu_cores=cpu_cores,
        ram_total_gb=round(vm.total / (1024**3), 2),
        ram_available_gb=round(vm.available / (1024**3), 2),
        gpu_type=gpu_type,
        vram_gb=vram_gb,
        disk_free_gb=_detect_disk_free_gb(),
        python_version=platform.python_version(),
        extras=extras,
    )


def can_run(profile: HardwareProfile, required_ram_gb: float) -> bool:
    """Return True if this profile has enough *available* RAM for the model.

    We compare against *available* RAM (not total) so we don't over-promise.
    A small buffer (1 GB) is reserved for the OS and the runner itself.
    """
    if required_ram_gb <= 0:
        return True
    buffer_gb = 1.0
    return profile.ram_available_gb >= required_ram_gb + buffer_gb


# Ordered from highest fidelity / largest size to smallest.
_QUANT_TIERS: list[tuple[float, list[str]]] = [
    (64.0, ["Q8_0", "Q6_K", "Q5_K_M"]),
    (32.0, ["Q5_K_M", "Q4_K_M"]),
    (16.0, ["Q4_K_M", "Q3_K_M"]),
    (8.0, ["Q3_K_M", "Q2_K"]),
]


def suggested_quant_max(profile: HardwareProfile) -> list[str]:
    """Return a list of quantization tiers appropriate for this machine.

    Heuristic: more available RAM → higher quality quants. The list is ordered
    from most-preferred to least-preferred; callers usually take ``[0]``.
    """
    avail = profile.ram_available_gb
    for threshold, quants in _QUANT_TIERS:
        if avail >= threshold:
            return list(quants)
    return ["Q2_K"]


__all__ = [
    "HardwareProfile",
    "profile_hardware",
    "can_run",
    "suggested_quant_max",
]
