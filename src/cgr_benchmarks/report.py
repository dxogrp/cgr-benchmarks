from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import BenchmarkConfig
from .plots import write_plots


def _table_or_message(path: Path, group_columns: list[str], value_column: str) -> str:
    if not path.exists():
        return f"No `{path}` file found.\n"
    frame = pd.read_csv(path)
    if frame.empty:
        return f"`{path}` is empty.\n"
    if value_column == "seconds" and "error" in frame.columns:
        frame = frame[frame["error"].isna()]
        if frame.empty:
            return f"`{path}` has no successful solve rows.\n"
    if value_column not in frame.columns:
        return frame.to_markdown(index=False) + "\n"
    summary = frame.groupby(group_columns, dropna=False)[value_column].agg(["count", "median", "mean", "min", "max"]).reset_index()
    return summary.to_markdown(index=False) + "\n"


def write_report(config: BenchmarkConfig) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    plots = write_plots(config)
    plot_lines = [f"- `{path.relative_to(config.results_dir)}`" for path in plots]
    if not plot_lines:
        plot_lines = ["No plots generated; run benchmarks first."]
    parts = [
        "# CVXGenRust Benchmark Summary",
        "",
        "## Solve Time Seconds",
        "",
        _table_or_message(
            config.results_dir / "solve_times.csv",
            ["problem_family", "size_label", "size", "problem", "tool"],
            "seconds",
        ),
        "",
        "## Plots",
        "",
        "\n".join(plot_lines),
        "",
    ]
    (config.results_dir / "summary.md").write_text("\n".join(parts), encoding="utf-8")
