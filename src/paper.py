from pathlib import Path
from shutil import which

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cgr_benchmarks.problems import size_axis_label

mpl.use("Agg")

mpl.rcParams["text.usetex"] = which("latex") is not None
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = ["serif"]
mpl.rcParams["font.size"] = 15
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["xtick.top"] = True
mpl.rcParams["ytick.right"] = True

PROBLEM_FAMILIES = [
    "nnls",
    "box_qp",
    "socp",
    "parametric_quad_form",
    "trace_sdp",
    "covariance_sdp",
    "mpc",
    "portfolio",
    "flow_lp",
    "logistic_regression",
    "lasso",
    "svm",
]

METHODS = ["cvxpy_direct", "cgr_python", "cvxpygen", "cgr_rust", "cvxpygen_c"]

METHOD_LABELS = {
    "cvxpy_direct": "CVXPY",
    "cgr_python": "CvxGenRust (Python)",
    "cvxpygen": "CVXPYgen (Python)",
    "cgr_rust": "CvxGenRust (Rust)",
    "cvxpygen_c": "CVXPYgen (C)",
}

METHOD_STYLES = {
    "cvxpy_direct": {"marker": "o", "color": "#4C78A8", "zorder": 2},
    "cgr_python": {"marker": "s", "color": "#F58518", "zorder": 5},
    "cvxpygen": {"marker": "^", "color": "#54A24B", "zorder": 2},
    "cgr_rust": {"marker": "D", "color": "#B279A2", "zorder": 5},
    "cvxpygen_c": {"marker": "v", "color": "#E45756", "zorder": 2},
}

SUCCESS_STATUSES = {"optimal", "optimal_inaccurate", "solved", "almostsolved", "almost solved"}


def _is_success_status(status: object) -> bool:
    if status is None or pd.isna(status):
        return False
    text = str(status).strip()
    lowered = text.lower()
    if lowered in SUCCESS_STATUSES:
        return True
    return text.split(maxsplit=1)[0] in {"1", "4"}


def _successful_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame
    if "error" in rows.columns:
        rows = rows[rows["error"].isna()]
    if "status" in rows.columns:
        rows = rows[rows["status"].map(_is_success_status)]
    return rows.copy()


def _summarize_solve_times(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["problem_family", "size", "size_label", "tool"], dropna=False)["seconds"]
        .agg(mean_seconds="mean")
        .reset_index()
    )
    summary["mean_ms"] = 1_000.0 * summary["mean_seconds"]
    return summary


def _format_family_title(family: str) -> str:
    if mpl.rcParams["text.usetex"]:
        return family.replace("_", r"\_")
    return family.replace("_", " ")


def _plot_solve_times(result_df: pd.DataFrame, path: Path) -> None:
    solve_df = _successful_rows(result_df)
    summary = _summarize_solve_times(solve_df)
    available_families = set(summary["problem_family"])
    families = [family for family in PROBLEM_FAMILIES if family in available_families]

    fig, axs = plt.subplots(4, 3, figsize=(12, 13.5), sharey=True)
    legend_handles = {}

    for index, family in enumerate(families):
        ax = axs[index // 3, index % 3]
        family_df = summary[summary["problem_family"] == family]
        sizes = sorted(family_df["size"].unique())

        for method in METHODS:
            method_df = family_df[family_df["tool"] == method].sort_values("size")
            if method_df.empty:
                continue

            handle = ax.plot(
                method_df["size"],
                method_df["mean_ms"],
                linewidth=1.5,
                markersize=4,
                label=METHOD_LABELS[method],
                **METHOD_STYLES[method],
            )[0]
            legend_handles.setdefault(method, handle)

        ax.set_title(_format_family_title(family), pad=6)
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.set_xlabel(size_axis_label(family))
        ax.grid(True, which="major", axis="x", alpha=0.18)
        ax.grid(True, which="major", axis="y", alpha=0.25)
        ax.grid(True, which="minor", axis="y", alpha=0.12)
        if index % 3 == 0:
            ax.set_ylabel("Solve time (ms)")

    for ax in axs.flat[len(families) :]:
        ax.axis("off")

    handles = [legend_handles[method] for method in METHODS if method in legend_handles]
    labels = [METHOD_LABELS[method] for method in METHODS if method in legend_handles]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    result_df = pd.read_csv("./results/solve_times.csv")
    _plot_solve_times(result_df, Path("./results/plots/solve_times.pdf"))
