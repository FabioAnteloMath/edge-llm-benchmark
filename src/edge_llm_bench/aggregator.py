"""Result aggregator — atomic CSV writes + canonical schema."""

from __future__ import annotations

import csv
import hashlib
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .utils.logging import get_logger

_log = get_logger("aggregator")


# Canonical CSV schema — one row per (model, format, backend, ctx_size) config.
CSV_FIELDS: list[str] = [
    "run_id",
    "timestamp",
    "host_profile",
    "model_id",
    "format",
    "backend",
    "ctx_size",
    "status",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "tokens_per_s_p50",
    "tokens_per_s_p95",
    "ram_peak_gb",
    "vram_peak_gb",
    "disk_size_gb",
    "mmlu_acc",
    "ifeval_strict_acc",
    "humaneval_pass1",
    "perplexity",
    "git_commit",
    "notes",
]


@dataclass
class ResultRow:
    """One row of the canonical CSV."""

    run_id: str
    timestamp: str
    host_profile: str
    model_id: str
    format: str
    backend: str
    ctx_size: int
    status: str
    ttft_ms_p50: float | None = None
    ttft_ms_p95: float | None = None
    tokens_per_s_p50: float | None = None
    tokens_per_s_p95: float | None = None
    ram_peak_gb: float | None = None
    vram_peak_gb: float | None = None
    disk_size_gb: float | None = None
    mmlu_acc: float | None = None
    ifeval_strict_acc: float | None = None
    humaneval_pass1: float | None = None
    perplexity: float | None = None
    git_commit: str | None = None
    notes: str | None = None

    def to_csv_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: ("" if v is None else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def make_run_id() -> str:
    """ISO date + 6-char short hash, e.g. ``2026-06-23-a1b2c3``."""
    iso = datetime.now(UTC).strftime("%Y-%m-%d")
    seed = f"{socket.gethostname()}-{iso}-{datetime.now(UTC).isoformat()}"
    short = hashlib.sha256(seed.encode()).hexdigest()[:6]
    return f"{iso}-{short}"


def get_git_commit() -> str:
    """Return current HEAD commit SHA, or ``"unknown"`` outside a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return sha[:12]
    except Exception:  # noqa: BLE001 — no git, or not a repo
        return "unknown"


# ---------------------------------------------------------------------------
# Atomic append
# ---------------------------------------------------------------------------


def append_result(run_dir: Path, row: ResultRow) -> Path:
    """Append one row to ``results.csv`` atomically. Creates file if missing.

    We accumulate the full CSV in-memory (it is small: hundreds of rows max)
    and rewrite via a temp + rename on every call. This keeps the operation
    atomic and avoids the open-in-``a`` append race that loses rows on Windows
    when the file is read concurrently.
    """
    csv_path = run_dir / "results.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    tmp_path = csv_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)
        writer.writerow(row.to_csv_dict())
        f.flush()
    tmp_path.replace(csv_path)
    _log.info("row_written", path=str(csv_path), model=row.model_id, format=row.format)
    return csv_path
