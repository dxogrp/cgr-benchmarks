from pathlib import Path

import pandas as pd
import pytest

from cgr_benchmarks.config import BenchmarkConfig
from cgr_benchmarks.report import write_report
from cgr_benchmarks.runner import SolveRecord, _write_records


@pytest.mark.reproducibility
def test_experiment_result_schema_and_report_are_stable(tmp_path):
    config = BenchmarkConfig(
        path=Path("benchmarks.toml"),
        warmups=1,
        samples=1,
        seed=1,
        enabled_problems=("nnls",),
        enabled_tools=("cvxpy_direct",),
        artifacts_dir=tmp_path / "artifacts",
        generated_dir=tmp_path / "generated",
        results_dir=tmp_path / "results",
        cvxpy_solver="CLARABEL",
        cvxpygen_solver="CLARABEL",
        rust_release=True,
    )
    _write_records(
        config,
        [
            SolveRecord(
                problem="nnls",
                problem_family="nnls",
                size_label="small",
                size=4,
                tool="cvxpy_direct",
                sample=0,
                seconds=0.001,
                status="optimal",
                objective=1.0,
                iterations=5,
                rel_objective_error=0.0,
                rel_solution_error=0.0,
                error=None,
            )
        ],
    )
    write_report(config)
    frame = pd.read_csv(config.results_dir / "solve_times.csv")
    assert list(frame.columns) == [
        "problem",
        "problem_family",
        "size_label",
        "size",
        "tool",
        "sample",
        "seconds",
        "status",
        "objective",
        "iterations",
        "rel_objective_error",
        "rel_solution_error",
        "error",
    ]
    assert (config.results_dir / "summary.md").exists()
