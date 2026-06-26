"""Decision tree — given a hardware profile + results, return ordered recommendations."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

VALID_STRATEGIES = ("max-quality", "max-speed", "min-ram", "balanced", "code")


@dataclass
class Recommendation:
    rank: int
    model_id: str
    format: str
    rationale: str
    score_quality: float
    score_speed: float
    score_ram: float
    pareto_optimal: bool = False
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _pareto_mask(df: pd.DataFrame, x: str, y: str, higher_better_x: bool = True) -> list[bool]:
    """Return a boolean column where True = Pareto-optimal point."""
    if x not in df.columns or y not in df.columns:
        return [False] * len(df)
    rows = list(zip(df[x], df[y], strict=False))
    mask: list[bool] = []
    for i, (xi, yi) in enumerate(rows):
        if pd.isna(xi) or pd.isna(yi):
            mask.append(False)
            continue
        dominated = False
        for j, (xj, yj) in enumerate(rows):
            if i == j or pd.isna(xj) or pd.isna(yj):
                continue
            ge_x = xj >= xi if higher_better_x else xj <= xi
            if ge_x and yj >= yi and (xj != xi or yj != yi):
                # j dominates i in at least one strictly
                if (xj > xi if higher_better_x else xj < xi) or yj > yi:
                    dominated = True
                    break
        mask.append(not dominated)
    return mask


def _filter_runnable(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Drop configs that this profile cannot run (RAM gate)."""
    if "ram_peak_gb" not in df.columns:
        return df.copy()
    # ruff doesn't understand pandas query()'s @variable syntax — the
    # variable IS used at runtime via the query string, but ruff only
    # sees it as "assigned but never read in Python code".
    avail_gb = float(profile.get("ram_available_gb") or profile.get("ram_total_gb") or 0)  # noqa: F841
    # Reserve 1 GB for the OS + the runner process itself.
    return df.query("`ram_peak_gb` <= @avail_gb - 1.0").copy()


def _strategy_max_quality(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["mmlu_acc"]).sort_values("mmlu_acc", ascending=False)
    return sub.head(5)


def _strategy_max_speed(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["tokens_per_s_p50", "ttft_ms_p50"]).copy()
    return (
        sub.query("`ttft_ms_p50` < 500.0").sort_values("tokens_per_s_p50", ascending=False).head(5)
    )


def _strategy_min_ram(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["ram_peak_gb", "mmlu_acc"]).copy()
    return sub.query("`mmlu_acc` >= 0.70").sort_values("ram_peak_gb", ascending=True).head(5)


def _strategy_balanced(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["mmlu_acc", "tokens_per_s_p50"]).copy()
    if sub.empty:
        return sub
    sub["pareto"] = _pareto_mask(sub, x="tokens_per_s_p50", y="mmlu_acc", higher_better_x=True)
    return sub[sub["pareto"]].sort_values("mmlu_acc", ascending=False).head(5)


def _strategy_code(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["humaneval_pass1"]).sort_values("humaneval_pass1", ascending=False)
    return sub.head(5)


_STRATEGIES = {
    "max-quality": _strategy_max_quality,
    "max-speed": _strategy_max_speed,
    "min-ram": _strategy_min_ram,
    "balanced": _strategy_balanced,
    "code": _strategy_code,
}


def _rationale(strategy: str, row: pd.Series) -> str:
    mmlu = row.get("mmlu_acc")
    tps = row.get("tokens_per_s_p50")
    ram = row.get("ram_peak_gb")
    ttft = row.get("ttft_ms_p50")
    he = row.get("humaneval_pass1")

    def fmt(v, places=2) -> str:
        return f"{v:.{places}f}" if pd.notna(v) else "—"

    if strategy == "max-quality":
        return f"Highest MMLU ({fmt(mmlu, 3)}) in your runnable set; {fmt(tps, 1)} tok/s, {fmt(ram)} GB."
    if strategy == "max-speed":
        return f"Fastest TTFT×speed combo under 500ms TTFT: {fmt(ttft)} ms, {fmt(tps, 1)} tok/s."
    if strategy == "min-ram":
        return f"Lowest RAM that still clears MMLU ≥ 0.70: {fmt(ram)} GB, MMLU={fmt(mmlu, 3)}."
    if strategy == "code":
        return f"Best HumanEval pass@1: {fmt(he, 3)}; model={row.get('model_id')}."
    # balanced
    return f"Pareto-optimal on (MMLU={fmt(mmlu, 3)}, tok/s={fmt(tps, 1)}); RAM={fmt(ram)} GB."


def recommend(
    profile: dict,
    results: pd.DataFrame,
    strategy: str = "balanced",
) -> list[Recommendation]:
    """Return ranked :class:`Recommendation` list for this profile + strategy."""
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {VALID_STRATEGIES}")

    runnable = _filter_runnable(results, profile)
    if runnable.empty:
        return []

    selected = _STRATEGIES[strategy](runnable)
    recs: list[Recommendation] = []

    # Compute Pareto mask only over rows that have BOTH quality and speed data.
    # Pad to the full runnable index so .assign() matches lengths.
    with_data = runnable.dropna(subset=["mmlu_acc", "tokens_per_s_p50"])
    if not with_data.empty:
        pareto_short = _pareto_mask(
            with_data, x="tokens_per_s_p50", y="mmlu_acc", higher_better_x=True
        )
        full_pareto = pd.Series(False, index=runnable.index)
        full_pareto.loc[with_data.index] = pareto_short
        runnable = runnable.assign(_pareto=full_pareto)
    else:
        runnable = runnable.assign(_pareto=False)

    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        is_pareto = bool(row.get("_pareto", False))
        recs.append(
            Recommendation(
                rank=rank,
                model_id=str(row.get("model_id", "")),
                format=str(row.get("format", "")),
                rationale=_rationale(strategy, row),
                score_quality=float(row.get("mmlu_acc", 0.0) or 0.0),
                score_speed=float(row.get("tokens_per_s_p50", 0.0) or 0.0),
                score_ram=float(row.get("ram_peak_gb", 0.0) or 0.0),
                pareto_optimal=is_pareto,
            )
        )
    return recs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_profile(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="edge-llm-bench-decide", description="Recommend models for a hardware profile."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--csv", type=Path, default=None, help="results CSV (default: newest in results/)"
    )
    parser.add_argument("--strategy", default="balanced", choices=VALID_STRATEGIES)
    args = parser.parse_args(argv)

    profile = _load_profile(args.profile)

    csv_path = args.csv
    if csv_path is None:
        from .report import latest_csv

        csv_path = latest_csv()
    if csv_path is None or not csv_path.exists():
        print("No results CSV. Run the benchmark first.", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    recs = recommend(profile, df, strategy=args.strategy)
    if not recs:
        print("No runnable configs for this profile + strategy.")
        return 0

    print(f"\nRecommendations (strategy={args.strategy}, profile={profile.get('host_name')}):\n")
    for r in recs:
        flag = " ⭐ Pareto" if r.pareto_optimal else ""
        print(f"  #{r.rank}  {r.model_id}  [{r.format}]{flag}")
        print(f"      {r.rationale}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
