import os

import pytest

from cgr_benchmarks.config import BenchmarkConfig
from cgr_benchmarks.generation import (
    cgr_dir,
    cgr_python_package,
    cgr_python_source_dir,
    cpg_dir,
    generate_all,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("CGR_BENCH_RUN_GENERATION_SMOKE") != "1",
    reason="set CGR_BENCH_RUN_GENERATION_SMOKE=1 to run heavyweight code-generation smoke tests",
)
pytestmark = [pytestmark, pytest.mark.toolchain]


def _config(tmp_path, problem):
    return BenchmarkConfig(
        path=tmp_path / "benchmarks.toml",
        warmups=1,
        samples=1,
        seed=1,
        enabled_problems=(problem,),
        enabled_tools=("cgr_python", "cvxpygen"),
        artifacts_dir=tmp_path / "artifacts",
        generated_dir=tmp_path / "artifacts" / "generated",
        results_dir=tmp_path / "results",
        cvxpy_solver="CLARABEL",
        cvxpygen_solver="CLARABEL",
        rust_release=True,
        strip_artifacts=True,
    )


@pytest.mark.parametrize("problem", ["nnls", "socp"])
def test_codegen_toolchain_generates_reference_problem(tmp_path, problem):
    config = _config(tmp_path, problem)
    generate_all(config, clean=True)
    assert (cgr_dir(config, problem) / "Cargo.toml").exists()
    assert (
        cgr_python_source_dir(config, problem)
        / cgr_python_package(config, problem)
        / "cgr_solver.py"
    ).exists()
    assert (cpg_dir(config, problem) / "cpg_solver.py").exists()
