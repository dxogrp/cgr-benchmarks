from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import BenchmarkConfig


def _save_bar_plot(frame: pd.DataFrame, x: str, y: str, hue: str, title: str, ylabel: str, path: Path) -> None:
    if frame.empty:
        return
    pivot = frame.pivot(index=x, columns=hue, values=y)
    ax = pivot.plot(kind="bar", figsize=(10, 5), width=0.82)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def _save_size_plot(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    families = list(frame["problem_family"].drop_duplicates())
    column_count = min(3, len(families))
    row_count = (len(families) + column_count - 1) // column_count
    fig, axes = plt.subplots(row_count, column_count, figsize=(4.2 * column_count, 3.2 * row_count), squeeze=False)
    for axis in axes.flat[len(families) :]:
        axis.axis("off")
    for axis, family in zip(axes.flat, families):
        family_frame = frame[frame["problem_family"] == family]
        pivot = family_frame.pivot(index="size", columns="tool", values="seconds").sort_index()
        pivot.plot(ax=axis, marker="o")
        axis.set_title(family)
        axis.set_xlabel("problem size")
        axis.set_ylabel("median seconds")
        axis.grid(alpha=0.25)
        axis.legend(title="", fontsize="small")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close(fig)


def _save_solution_error_plot(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    plot_frame = frame.copy()
    plot_frame["plot_error"] = plot_frame["rel_solution_error"].clip(lower=1e-16)
    pivot = plot_frame.pivot(index="problem", columns="tool", values="plot_error")
    ax = pivot.plot(kind="bar", figsize=(10, 5), width=0.82)
    ax.set_title("Relative Solution Error vs CVXPY")
    ax.set_xlabel("")
    ax.set_ylabel("median relative solution error")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def _save_objective_error_plot(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    plot_frame = frame.copy()
    plot_frame["plot_error"] = plot_frame["rel_objective_error"].clip(lower=1e-16)
    pivot = plot_frame.pivot(index="problem", columns="tool", values="plot_error")
    ax = pivot.plot(kind="bar", figsize=(10, 5), width=0.82)
    ax.set_title("Relative Objective Error vs CVXPY")
    ax.set_xlabel("")
    ax.set_ylabel("median relative objective error")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def write_plots(config: BenchmarkConfig) -> list[Path]:
    plot_dir = config.results_dir / "plots"
    written: list[Path] = []

    solve_path = config.results_dir / "solve_times.csv"
    if solve_path.exists():
        solve = pd.read_csv(solve_path)
        solve = solve[solve["error"].isna()] if "error" in solve.columns else solve
        if not solve.empty:
            summary = solve.groupby(["problem", "tool"], dropna=False)["seconds"].median().reset_index()
            path = plot_dir / "solve_time_median_seconds.png"
            _save_bar_plot(summary, "problem", "seconds", "tool", "Median Solve Time", "seconds", path)
            written.append(path)
            if {"problem_family", "size", "tool"}.issubset(solve.columns):
                size_summary = (
                    solve.groupby(["problem_family", "size", "tool"], dropna=False)["seconds"].median().reset_index()
                )
                path = plot_dir / "solve_time_by_size_seconds.png"
                _save_size_plot(size_summary, path)
                written.append(path)
            if {"problem", "tool", "rel_solution_error"}.issubset(solve.columns):
                solution_error = solve.dropna(subset=["rel_solution_error"])
                if not solution_error.empty:
                    solution_error_summary = (
                        solution_error.groupby(["problem", "tool"], dropna=False)["rel_solution_error"]
                        .median()
                        .reset_index()
                    )
                    path = plot_dir / "solution_error_median_relative.png"
                    _save_solution_error_plot(solution_error_summary, path)
                    written.append(path)
            if {"problem", "tool", "rel_objective_error"}.issubset(solve.columns):
                objective_error = solve.dropna(subset=["rel_objective_error"])
                if not objective_error.empty:
                    objective_error_summary = (
                        objective_error.groupby(["problem", "tool"], dropna=False)["rel_objective_error"]
                        .median()
                        .reset_index()
                    )
                    path = plot_dir / "objective_error_median_relative.png"
                    _save_objective_error_plot(objective_error_summary, path)
                    written.append(path)

    return written
