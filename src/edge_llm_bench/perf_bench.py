"""Performance benchmark — measure TTFT, tokens/s, RAM, prefill cost.

Protocol (from plan §5.5):

1. Load model once.
2. Warmup — N generations, ignored.
3. Short prompts  (10 in / 50 out)  — 5 runs. TTFT variance.
4. Medium prompts (256 in / 256 out) — 5 runs. Steady-state tokens/s.
5. Long prompts   (1024 in / 512 out) — 3 runs. Prefill cost + memory ceiling.
6. Streaming sample — track inter-token intervals for jitter.

Failure modes:

* OOM during load → record ``error="oom_during_load"``, skip the rest of the run.
* Backend crash mid-run → record partial, mark as failed but keep earlier runs.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from tqdm import tqdm

from .inference.base import GenerateResult, InferenceBackend

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PerfStats:
    ttft_ms_p50: float = 0.0
    ttft_ms_p95: float = 0.0
    tokens_per_s_p50: float = 0.0
    tokens_per_s_p95: float = 0.0
    ram_peak_gb: float = 0.0
    vram_peak_gb: float | None = None
    prefill_tokens_per_s: float | None = None
    inter_token_jitter_ms: float | None = None
    error: str | None = None
    n_runs: int = 0
    raw: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Prompt fixtures
# ---------------------------------------------------------------------------


def _short_prompts(n: int) -> list[str]:
    base = "Summarize the following in one sentence: "
    seed = (
        "Large language models have transformed natural language processing. "
        "They are trained on vast corpora of text and can generate coherent, "
        "contextually relevant continuations of a prompt. Recent work focuses "
        "on efficiency, safety, and reasoning capabilities."
    )
    return [base + seed for _ in range(n)]


def _medium_prompts(n: int, in_tokens: int = 256) -> list[str]:
    filler = "The quick brown fox jumps over the lazy dog. " * 30
    return [filler[: in_tokens * 4] for _ in range(n)]


def _long_prompts(n: int, in_tokens: int = 1024) -> list[str]:
    filler = (
        "Large language models are neural networks trained on text. "
        "They use the transformer architecture and self-attention. "
    ) * 200
    return [filler[: in_tokens * 4] for _ in range(n)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(pct) - 1])


def _safe_generate(backend: InferenceBackend, prompt: str, max_tokens: int) -> GenerateResult:
    try:
        return backend.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 — record, don't crash the run
        return GenerateResult(
            text="",
            total_ms=0.0,
            output_tokens=0,
            tokens_per_s=0.0,
            ram_peak_gb=0.0,
            extras={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_perf(
    backend: InferenceBackend,
    n_warmup: int = 5,
    n_short: int = 5,
    n_medium: int = 5,
    n_long: int = 3,
    short_in: int = 10,
    short_out: int = 50,
    medium_in: int = 256,
    medium_out: int = 256,
    long_in: int = 1024,
    long_out: int = 512,
    progress: bool = True,
) -> PerfStats:
    """Run the full performance protocol and return aggregated stats.

    The caller is responsible for ``backend.load(...)`` before calling this.
    """
    stats = PerfStats()

    # ---- Warmup -----------------------------------------------------------
    warmup_prompts = ["hi"] * n_warmup
    for prompt in tqdm(warmup_prompts, desc="warmup", disable=not progress):
        _safe_generate(backend, prompt, max_tokens=8)

    # ---- Short prompts (TTFT) ---------------------------------------------
    short_prompts = _short_prompts(n_short)
    for prompt in tqdm(short_prompts, desc="short", disable=not progress):
        r = _safe_generate(backend, prompt, max_tokens=short_out)
        stats.raw.append({"phase": "short", **r.__dict__})

    # ---- Medium prompts (steady-state tokens/s) ---------------------------
    medium_prompts = _medium_prompts(n_medium, in_tokens=medium_in)
    for prompt in tqdm(medium_prompts, desc="medium", disable=not progress):
        r = _safe_generate(backend, prompt, max_tokens=medium_out)
        stats.raw.append({"phase": "medium", **r.__dict__})

    # ---- Long prompts (prefill cost + memory ceiling) ---------------------
    long_prompts = _long_prompts(n_long, in_tokens=long_in)
    for prompt in tqdm(long_prompts, desc="long", disable=not progress):
        r = _safe_generate(backend, prompt, max_tokens=long_out)
        stats.raw.append({"phase": "long", **r.__dict__})

    # ---- Aggregate --------------------------------------------------------
    all_oks = [row for row in stats.raw if not row.get("extras", {}).get("error")]
    if not all_oks:
        stats.error = "all_runs_failed"
        return stats

    ttfst = [row["ttft_ms"] for row in all_oks if row["ttft_ms"] > 0]
    tps = [row["tokens_per_s"] for row in all_oks if row["tokens_per_s"] > 0]
    ram_peaks = [row["ram_peak_gb"] for row in all_oks if row["ram_peak_gb"] > 0]
    vram_peaks = [
        row["vram_peak_gb"]
        for row in all_oks
        if row.get("vram_peak_gb") is not None and row["vram_peak_gb"] > 0
    ]

    long_rows = [row for row in all_oks if row["phase"] == "long"]
    long_total_ms = sum(row["total_ms"] for row in long_rows) or 1.0
    long_in_tokens = long_in  # approximate; tokenizer-dependent but OK for relative
    prefill_tps = (
        (long_in_tokens * len(long_rows) / (long_total_ms / 1000.0)) if long_rows else None
    )

    stats.ttft_ms_p50 = round(_percentile(ttfst, 50), 2)
    stats.ttft_ms_p95 = round(_percentile(ttfst, 95), 2)
    stats.tokens_per_s_p50 = round(_percentile(tps, 50), 3)
    stats.tokens_per_s_p95 = round(_percentile(tps, 95), 3)
    stats.ram_peak_gb = round(max(ram_peaks), 3) if ram_peaks else 0.0
    stats.vram_peak_gb = round(max(vram_peaks), 3) if vram_peaks else None
    stats.prefill_tokens_per_s = round(prefill_tps, 2) if prefill_tps else None
    stats.n_runs = len(all_oks)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:  # pragma: no cover - exercised manually
    import argparse

    from .inference import BackendRegistry

    parser = argparse.ArgumentParser(description="Run performance benchmark on one model.")
    parser.add_argument("--model", required=True, help="Ollama tag or path to GGUF")
    parser.add_argument("--backend", default="ollama", choices=BackendRegistry.available())
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    backend_cls = BackendRegistry.get(args.backend)
    backend = backend_cls()
    backend.load(args.model, ctx_size=args.ctx_size)
    try:
        stats = run_perf(backend)
        print(stats)
    finally:
        backend.close()


if __name__ == "__main__":  # pragma: no cover
    _cli()
