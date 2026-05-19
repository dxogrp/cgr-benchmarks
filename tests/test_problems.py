import cvxpy as cp
import pytest

from cgr_benchmarks.problems import CASES


@pytest.mark.reproducibility
def test_experiment_problem_families_are_dpp_and_solve_directly():
    for case in CASES.values():
        fixture = case.factory()
        assert fixture.problem.is_dpp(quad_form_dpp="qp"), case.name
        value = fixture.problem.solve(solver=cp.CLARABEL)
        assert value is not None, case.name
        assert fixture.problem.status in {"optimal", "optimal_inaccurate"}, case.name


@pytest.mark.reproducibility
def test_sample_generators_match_declared_parameters():
    for case in CASES.values():
        fixture = case.factory()
        values = fixture.sample_parameters(__import__("numpy").random.default_rng(123), 0)
        assert set(values) == set(fixture.parameters), case.name
