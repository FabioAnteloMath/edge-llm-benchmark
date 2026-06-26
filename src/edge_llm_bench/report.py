"""Report generator — CSV → self-contained HTML with 5 Plotly charts + sortable table.

The output is a single static HTML file that opens offline (Plotly loads from
CDN by default; flip ``--cdn plotly`` to bundle the JS). All CSV data is
inlined as JSON in the page; no server, no build step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that knows how to serialize numpy scalars and arrays."""

    def default(self, obj):  # noqa: D401
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            f = float(obj)
            return f if np.isfinite(f) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    # Drop rows that did not run at all.
    df: pd.DataFrame = raw.query("status == 'ok' or status == 'partial'").copy()
    # Coerce numerics, leaving NaN where the test did not produce a value.
    for col in (
        "ttft_ms_p50",
        "tokens_per_s_p50",
        "ram_peak_gb",
        "mmlu_acc",
        "ifeval_strict_acc",
        "humaneval_pass1",
        "perplexity",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def latest_csv(root: Path = Path("results")) -> Path | None:
    """Return the newest ``results.csv`` under ``root``, or None."""
    candidates = sorted(root.glob("*/results.csv"), reverse=True)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _empty_figure(title: str, message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[dict(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)],
        template="plotly_white",
    )
    return fig


def chart_quality_vs_speed(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["mmlu_acc", "ttft_ms_p50"])
    if sub.empty:
        return _empty_figure("Quality vs Time-to-First-Token")
    fig = px.scatter(
        sub,
        x="ttft_ms_p50",
        y="mmlu_acc",
        size="ram_peak_gb",
        color="family" if "family" in sub.columns else "model_id",
        hover_data=["model_id", "format", "tokens_per_s_p50"],
        title="Quality (MMLU) vs Time-to-First-Token<br>"
        "<sub>Bubble size = RAM peak. Lower-left is better.</sub>",
        labels={
            "ttft_ms_p50": "TTFT (ms, lower is better)",
            "mmlu_acc": "MMLU accuracy (higher is better)",
        },
    )
    return fig


def chart_quality_vs_memory(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["mmlu_acc", "ram_peak_gb"])
    if sub.empty:
        return _empty_figure("Quality vs Memory")
    fig = px.scatter(
        sub,
        x="ram_peak_gb",
        y="mmlu_acc",
        color="format",
        symbol="model_id",
        hover_data=["model_id", "ttft_ms_p50", "tokens_per_s_p50"],
        title="Quality (MMLU) vs Memory<br>"
        "<sub>Each symbol = one model. Color = quantization format.</sub>",
        labels={
            "ram_peak_gb": "RAM peak (GB, lower is better)",
            "mmlu_acc": "MMLU accuracy (higher is better)",
        },
    )
    return fig


def chart_per_model_bars(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("Per-model Metrics")
    metric_cols = [c for c in ("ttft_ms_p50", "tokens_per_s_p50", "mmlu_acc") if c in df.columns]
    melted = df.melt(
        id_vars=["model_id", "format"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    ).dropna()
    if melted.empty:
        return _empty_figure("Per-model Metrics")
    fig = px.bar(
        melted,
        x="model_id",
        y="value",
        color="metric",
        barmode="group",
        facet_col="format",
        title="Per-model Metrics by Quantization Format",
    )
    return fig


def chart_pareto(df: pd.DataFrame) -> go.Figure:
    """Pareto frontier on (MMLU, tokens_per_s_p50) — higher is better in both."""
    sub = df.dropna(subset=["mmlu_acc", "tokens_per_s_p50"]).copy()
    if sub.empty:
        return _empty_figure("Pareto Frontier")
    # Compute Pareto front manually.
    pts = list(
        zip(
            sub["mmlu_acc"],
            sub["tokens_per_s_p50"],
            sub["model_id"],
            sub["format"],
            strict=False,
        )
    )
    pareto_mask: list[bool] = []
    for i, (x_i, y_i, _m_i, _f_i) in enumerate(pts):
        dominated = False
        for j, (x_j, y_j, _, _) in enumerate(pts):
            if i == j:
                continue
            if x_j >= x_i and y_j >= y_i and (x_j > x_i or y_j > y_i):
                dominated = True
                break
        pareto_mask.append(not dominated)
    sub["pareto"] = pareto_mask

    fig = px.scatter(
        sub,
        x="tokens_per_s_p50",
        y="mmlu_acc",
        color="pareto",
        color_discrete_map={True: "#d62728", False: "#9ca3af"},
        symbol="model_id",
        hover_data=["format", "ram_peak_gb"],
        title="Pareto Frontier (Quality × Speed)<br>"
        "<sub>Red points are Pareto-optimal — you cannot improve one "
        "without sacrificing the other.</sub>",
        labels={
            "tokens_per_s_p50": "Tokens/s (higher is better)",
            "mmlu_acc": "MMLU (higher is better)",
        },
    )
    return fig


def chart_quality_degradation(df: pd.DataFrame) -> go.Figure:
    """MMLU + perplexity across quantization levels, per model family."""
    if df.empty or "family" not in df.columns:
        return _empty_figure("Quality Degradation by Quantization")
    fig = go.Figure()
    families: list[str] = sorted([str(f) for f in df["family"].dropna().unique().tolist()])
    plotted = False
    for fam in families:
        sub = df.query("`family` == @fam").dropna(subset=["mmlu_acc"])
        if sub.empty:
            continue
        sub = sub.sort_values("format")
        fig.add_trace(
            go.Scatter(
                x=sub["format"],
                y=sub["mmlu_acc"],
                mode="lines+markers",
                name=f"{fam} (MMLU)",
            )
        )
        plotted = True
    if not plotted:
        return _empty_figure("Quality Degradation by Quantization")
    fig.update_layout(
        title="Quality Degradation Across Quantization Levels",
        xaxis_title="Quantization format",
        yaxis_title="MMLU accuracy",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# HTML templating
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>edge-llm-benchmark report — {run_id}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 24px; background: #fafafa; color: #222; }}
  h1 {{ margin-top: 0; }}
  .meta {{ color: #555; font-size: 14px; margin-bottom: 24px; }}
  .chart {{ background: white; border-radius: 8px; padding: 16px;
            margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  table {{ width: 100%; border-collapse: collapse; background: white; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f3f4f6; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 12px; background: #e5e7eb; }}
  .badge.ok {{ background: #d1fae5; color: #065f46; }}
  .badge.partial {{ background: #fef3c7; color: #92400e; }}
  .badge.error {{ background: #fee2e2; color: #991b1b; }}
</style>
</head>
<body>
<h1>edge-llm-benchmark report</h1>
<div class="meta">
  Run ID: <strong>{run_id}</strong> &middot;
  Host: <span class="badge">{host}</span> &middot;
  Backend: <span class="badge">{backend}</span> &middot;
  Rows: <span class="badge">{n_rows}</span> &middot;
  Generated: {generated_at}
</div>

<div class="chart" id="chart-speed"></div>
<div class="chart" id="chart-mem"></div>
<div class="chart" id="chart-bars"></div>
<div class="chart" id="chart-pareto"></div>
<div class="chart" id="chart-degrade"></div>

<h2>All results</h2>
<table>
  <thead>
    <tr>
      <th>Model</th><th>Format</th><th>Status</th>
      <th>TTFT (ms)</th><th>Tok/s</th><th>RAM (GB)</th>
      <th>MMLU</th><th>IFEval</th><th>HumanEval</th><th>PPL</th>
    </tr>
  </thead>
  <tbody>
{table_rows}
  </tbody>
</table>

<script>
  const figs = {figures_json};
  for (const [divId, fig] of Object.entries(figs)) {{
    Plotly.newPlot(divId, fig.data, fig.layout, {{responsive: true}});
  }}
</script>
</body>
</html>
"""


def _table_rows(df: pd.DataFrame) -> str:
    parts: list[str] = []
    cols = [
        "model_id",
        "format",
        "status",
        "ttft_ms_p50",
        "tokens_per_s_p50",
        "ram_peak_gb",
        "mmlu_acc",
        "ifeval_strict_acc",
        "humaneval_pass1",
        "perplexity",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            v = row.get(col)
            if pd.isna(v):
                cells.append("<td>&mdash;</td>")
            elif col == "status":
                cells.append(f'<td><span class="badge {v}">{v}</span></td>')
            elif isinstance(v, float):
                cells.append(f"<td>{v:.3f}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        parts.append(f"<tr>{''.join(cells)}</tr>")
    return "\n".join(parts)


def generate_report(csv_path: Path, output_html: Path) -> None:
    df = load_results(csv_path)
    run_id = csv_path.parent.name
    host = (
        df["host_profile"].iloc[0] if "host_profile" in df.columns and not df.empty else "unknown"
    )
    backend = df["backend"].iloc[0] if "backend" in df.columns and not df.empty else "unknown"
    n_rows = len(df)

    charts = [
        chart_quality_vs_speed(df),
        chart_quality_vs_memory(df),
        chart_per_model_bars(df),
        chart_pareto(df),
        chart_quality_degradation(df),
    ]
    figures_json = json.dumps(
        {
            f"chart-{name}": fig.to_plotly_json()
            for name, fig in zip(
                ("speed", "mem", "bars", "pareto", "degrade"), charts, strict=False
            )
        },
        cls=_NumpyJSONEncoder,
    )

    html = _HTML_TEMPLATE.format(
        run_id=run_id,
        host=host,
        backend=backend,
        n_rows=n_rows,
        generated_at=pd.Timestamp.utcnow().isoformat(),
        table_rows=_table_rows(df),
        figures_json=figures_json,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="edge-llm-bench-report", description="Render a benchmark CSV to HTML."
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/report.html"))
    parser.add_argument("--latest", action="store_true", help="Use the newest results/*.csv")
    args = parser.parse_args(argv)

    csv_path = args.csv
    if args.latest or csv_path is None:
        csv_path = latest_csv()
    if csv_path is None or not csv_path.exists():
        print("No CSV found. Pass --csv <path> or --latest (after a runner run).", file=sys.stderr)
        return 1

    out = args.out
    if str(out) == "docs/report.html" and csv_path:
        out = csv_path.parent.parent.parent / "docs" / "report.html"
    generate_report(csv_path, out)
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
