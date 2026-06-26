"""CLI runner — top-level orchestrator. Wires hardware + matrix + bench + report."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from .aggregator import (
    ResultRow,
    append_result,
    get_git_commit,
    make_run_id,
)
from .inference import BackendRegistry
from .perf_bench import run_perf
from .quality_bench import run_quality
from .utils.logging import get_logger, setup_logging
from .utils.state import RunState

_log = get_logger("runner")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_matrix(path: Path) -> dict:
    cfg = load_yaml(path)
    if "models" not in cfg:
        raise ValueError(f"matrix config missing 'models' key: {path}")
    return cfg


def load_profile(path: Path) -> dict:
    cfg = load_yaml(path)
    if "host_name" not in cfg:
        raise ValueError(f"profile config missing 'host_name' key: {path}")
    return cfg


# ---------------------------------------------------------------------------
# Matrix resolution
# ---------------------------------------------------------------------------


def resolve_matrix(matrix_cfg: dict, profile: dict, profile_name: str) -> list[dict]:
    """Expand (model × format) and filter by hardware capability."""
    defaults = matrix_cfg.get("defaults", {})
    resolved: list[dict] = []

    ram_avail = float(profile.get("ram_available_gb") or profile.get("ram_total_gb") or 0)

    for model in matrix_cfg["models"]:
        for fmt in model["formats"]:
            requires = float(model.get("requires_ram_gb", 0))
            # Use can_run-style check against available RAM
            buffer = 1.0
            runnable = (ram_avail - buffer) >= requires
            resolved.append(
                {
                    "config_id": f"{model['id']}|{fmt}",
                    "model_id": model["id"],
                    "family": model.get("family", "unknown"),
                    "format": fmt,
                    "requires_ram_gb": requires,
                    "ctx_size": int(model.get("ctx_size", defaults.get("ctx_size", 4096))),
                    "runnable": runnable,
                    "skip_reason": None
                    if runnable
                    else f"needs {requires} GB, have {ram_avail} GB",
                }
            )
    return resolved


# ---------------------------------------------------------------------------
# Per-config execution
# ---------------------------------------------------------------------------


def _run_one_config(
    cfg: dict,
    matrix_cfg: dict,
    profile_name: str,
    backend_name: str,
    output_dir: Path,
    skip_quality: bool,
    skip_quant: bool,
    max_examples: int | None,
) -> ResultRow:
    """Execute a single (model, format) config and return one CSV row."""
    run_id = output_dir.name
    timestamp = datetime.now(UTC).isoformat()

    if not cfg["runnable"]:
        return ResultRow(
            run_id=run_id,
            timestamp=timestamp,
            host_profile=profile_name,
            model_id=cfg["model_id"],
            format=cfg["format"],
            backend=backend_name,
            ctx_size=cfg["ctx_size"],
            status="skipped",
            notes=cfg.get("skip_reason"),
        )

    backend_cls = BackendRegistry.get(backend_name)
    backend = backend_cls()
    tag = cfg["model_id"]  # for Ollama, use the repo id as the tag

    try:
        backend.load(tag, format=cfg["format"], ctx_size=cfg["ctx_size"])
    except Exception as exc:  # noqa: BLE001
        return ResultRow(
            run_id=run_id,
            timestamp=timestamp,
            host_profile=profile_name,
            model_id=cfg["model_id"],
            format=cfg["format"],
            backend=backend_name,
            ctx_size=cfg["ctx_size"],
            status="error",
            notes=f"load failed: {exc}",
        )

    try:
        perf = run_perf(backend, progress=False)
        quality = (
            run_quality(backend, suite="all", max_examples=max_examples)
            if not skip_quality
            else None
        )
        status = "ok" if perf.error is None else "partial"
        return ResultRow(
            run_id=run_id,
            timestamp=timestamp,
            host_profile=profile_name,
            model_id=cfg["model_id"],
            format=cfg["format"],
            backend=backend_name,
            ctx_size=cfg["ctx_size"],
            status=status,
            ttft_ms_p50=perf.ttft_ms_p50,
            ttft_ms_p95=perf.ttft_ms_p95,
            tokens_per_s_p50=perf.tokens_per_s_p50,
            tokens_per_s_p95=perf.tokens_per_s_p95,
            ram_peak_gb=perf.ram_peak_gb,
            vram_peak_gb=perf.vram_peak_gb,
            mmlu_acc=quality.mmlu_acc if quality else None,
            ifeval_strict_acc=quality.ifeval_strict_acc if quality else None,
            humaneval_pass1=quality.humaneval_pass1 if quality else None,
            perplexity=quality.perplexity if quality else None,
            git_commit=get_git_commit(),
        )
    except Exception as exc:  # noqa: BLE001
        return ResultRow(
            run_id=run_id,
            timestamp=timestamp,
            host_profile=profile_name,
            model_id=cfg["model_id"],
            format=cfg["format"],
            backend=backend_name,
            ctx_size=cfg["ctx_size"],
            status="error",
            notes=f"benchmark failed: {exc}",
        )
    finally:
        try:
            backend.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _sigint_handler(state: RunState, state_path: Path):
    def handler(signum, frame):  # noqa: ARG001
        _log.warning("sigint_received", checkpoint=str(state_path))
        state.save(state_path)
        sys.exit(130)

    return handler


def run(args: argparse.Namespace) -> int:
    """Main entry. Returns process exit code (0 = ≥1 success, 1 = all failed)."""
    matrix_cfg = load_matrix(args.config)
    profile_cfg = load_profile(args.profile)
    profile_name = profile_cfg.get("host_name") or args.profile.stem

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    setup_logging(log_path, level=args.log_level)
    _log.info(
        "run_started",
        profile=profile_name,
        backend=args.backend,
        output_dir=str(output_dir),
        skip_quality=args.skip_quality,
        dry_run=args.dry_run,
    )

    # Resume state
    run_id = output_dir.name if output_dir.name != "latest" else make_run_id()
    state_path = output_dir / "state.json"
    state = RunState.load_or_new(
        state_path, run_id=run_id, started_at=datetime.now(UTC).isoformat()
    )

    # SIGINT checkpoint
    signal.signal(signal.SIGINT, _sigint_handler(state, state_path))

    # Resolve matrix
    matrix = resolve_matrix(matrix_cfg, profile_cfg, profile_name)
    _log.info("matrix_resolved", n=len(matrix), runnable=sum(1 for c in matrix if c["runnable"]))

    if args.dry_run:
        print(json.dumps(matrix, indent=2))
        return 0

    # Pre-flight disk check
    disk_free_gb = profile_cfg.get("disk_free_gb")
    if disk_free_gb is not None and disk_free_gb < 10:
        _log.error("aborting_low_disk", free_gb=disk_free_gb)
        return 2

    # Iterate
    successes = 0
    failures = 0
    bar = tqdm(matrix, desc="benchmark", disable=not sys.stderr.isatty())
    for cfg in bar:
        bar.set_description(f"→ {cfg['model_id']}|{cfg['format']}")
        if args.resume and state.is_done(cfg["config_id"]):
            _log.info("skipping_resume", config_id=cfg["config_id"])
            continue

        t0 = time.perf_counter()
        row = _run_one_config(
            cfg,
            matrix_cfg,
            profile_name,
            args.backend,
            output_dir,
            args.skip_quality,
            args.skip_quant,
            args.max_examples,
        )
        elapsed = time.perf_counter() - t0
        append_result(output_dir, row)
        state.mark(cfg["config_id"], row.status, note=row.notes or "")
        state.save(state_path)

        if row.status in ("ok", "partial", "skipped"):
            successes += 1
        else:
            failures += 1
        _log.info(
            "config_done",
            config_id=cfg["config_id"],
            status=row.status,
            elapsed_s=round(elapsed, 1),
        )

    exit_code = 0 if successes > 0 else 1
    _log.info("run_complete", successes=successes, failures=failures, exit_code=exit_code)
    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edge-llm-bench", description=__doc__)
    p.add_argument("--profile", type=Path, required=True, help="Hardware profile YAML")
    p.add_argument(
        "--config", type=Path, default=Path("configs/matrix.yaml"), help="Model/format matrix YAML"
    )
    p.add_argument("--backend", default="ollama", choices=BackendRegistry.available())
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dry-run"),
        help="Directory for results.csv + state.json. Auto-created.",
    )
    p.add_argument("--skip-quant", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--max-examples", type=int, default=None, help="Cap eval examples per suite (smoke testing)"
    )
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--track-ram", action="store_true", default=True)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
