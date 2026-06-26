"""Tests for the runner — primarily dry-run resolution + resume behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_llm_bench.runner import build_parser, load_matrix, load_profile, resolve_matrix


@pytest.fixture
def tmp_matrix(tmp_path: Path) -> Path:
    p = tmp_path / "matrix.yaml"
    p.write_text(
        """version: 1
defaults:
  ctx_size: 4096
models:
  - id: org/small
    family: small
    formats: [Q4_K_M, Q8_0]
    requires_ram_gb: 8
  - id: org/big
    family: big
    formats: [Q4_K_M]
    requires_ram_gb: 64
""",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def tmp_profile(tmp_path: Path) -> Path:
    p = tmp_path / "profile.yaml"
    p.write_text(
        """host_name: tiny
os: linux
arch: x86_64
cpu_cores: 4
ram_total_gb: 16
ram_available_gb: 12
gpu_type: none
vram_gb: null
disk_free_gb: 100
notes: "small test profile"
""",
        encoding="utf-8",
    )
    return p


def test_load_matrix(tmp_matrix: Path) -> None:
    cfg = load_matrix(tmp_matrix)
    assert "models" in cfg
    assert len(cfg["models"]) == 2


def test_load_profile(tmp_profile: Path) -> None:
    cfg = load_profile(tmp_profile)
    assert cfg["host_name"] == "tiny"


def test_resolve_matrix_filters_by_ram(tmp_matrix: Path, tmp_profile: Path) -> None:
    matrix_cfg = load_matrix(tmp_matrix)
    profile_cfg = load_profile(tmp_profile)
    resolved = resolve_matrix(matrix_cfg, profile_cfg, profile_cfg["host_name"])
    # 2 models × formats → 3 rows total; only those runnable on 12 GB
    assert len(resolved) == 3
    runnable_ids = [c["config_id"] for c in resolved if c["runnable"]]
    assert "org/small|Q4_K_M" in runnable_ids
    assert "org/small|Q8_0" in runnable_ids
    assert "org/big|Q4_K_M" not in runnable_ids  # 64 GB requirement


def test_resolve_matrix_records_skip_reason(tmp_matrix: Path, tmp_profile: Path) -> None:
    matrix_cfg = load_matrix(tmp_matrix)
    profile_cfg = load_profile(tmp_profile)
    resolved = resolve_matrix(matrix_cfg, profile_cfg, profile_cfg["host_name"])
    big = [c for c in resolved if c["model_id"] == "org/big"][0]
    assert big["runnable"] is False
    assert big["skip_reason"] is not None
    assert "needs 64" in big["skip_reason"]


def test_parser_has_expected_flags() -> None:
    parser = build_parser()
    # Argparse should accept --dry-run + --resume + --max-examples etc.
    ns = parser.parse_args(
        ["--profile", "p.yaml", "--output-dir", "results/foo", "--dry-run", "--max-examples", "5"]
    )
    assert ns.dry_run is True
    assert ns.max_examples == 5
    assert ns.resume is False
