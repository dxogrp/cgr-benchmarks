from pathlib import Path

import pytest

from cgr_benchmarks.config import BenchmarkConfig


@pytest.mark.reproducibility
def test_default_experiment_config_loads():
    config = BenchmarkConfig.load(Path("benchmarks.toml"))
    assert config.samples > 0
    assert "nnls_small" in config.enabled_problems
    assert config.cvxpy_solver == "CLARABEL"
