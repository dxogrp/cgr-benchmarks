from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cvxpy as cp
import numpy as np


@dataclass
class ProblemFixture:
    problem: cp.Problem
    variables: dict[str, cp.Variable]
    parameters: dict[str, cp.Parameter]
    sample_parameters: Callable[[np.random.Generator, int], dict[str, np.ndarray | float]]


@dataclass(frozen=True)
class ProblemCase:
    name: str
    description: str
    factory: Callable[[], ProblemFixture]
    family: str
    size_label: str
    size: int


def _assign(parameters: dict[str, cp.Parameter], values: dict[str, np.ndarray | float]) -> None:
    for name, value in values.items():
        parameters[name].value = value


def _nnls(m: int = 8, n: int = 4) -> ProblemFixture:
    x = cp.Variable(n, name="x")
    a = cp.Parameter((m, n), name="A")
    b = cp.Parameter(m, name="b")
    problem = cp.Problem(cp.Minimize(cp.sum_squares(a @ x - b)), [x >= 0])

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray]:
        return {"A": rng.normal(size=(m, n)), "b": rng.normal(size=m)}

    parameters = {"A": a, "b": b}
    _assign(parameters, sample(np.random.default_rng(1), 0))
    return ProblemFixture(problem, {"x": x}, parameters, sample)


def _box_qp(n: int = 6) -> ProblemFixture:
    x = cp.Variable(n, name="x")
    x_ref = cp.Parameter(n, name="x_ref")
    q = cp.Parameter(n, name="q")
    a = cp.Parameter((1, n), name="A")
    b = cp.Parameter(1, name="b")
    objective = 0.5 * cp.sum_squares(x - x_ref) + q @ x
    constraints = [a @ x == b, x >= 0, x <= 1]
    problem = cp.Problem(cp.Minimize(objective), constraints)

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray]:
        weights = rng.uniform(0.2, 1.2, size=n)
        return {
            "x_ref": rng.uniform(0.0, 1.0, size=n),
            "q": rng.normal(size=n),
            "A": weights.reshape(1, n),
            "b": np.array([0.5 * float(weights.sum())]),
        }

    parameters = {"x_ref": x_ref, "q": q, "A": a, "b": b}
    _assign(parameters, sample(np.random.default_rng(2), 0))
    return ProblemFixture(problem, {"x": x}, parameters, sample)


def _parametric_quad_form(n: int = 4) -> ProblemFixture:
    x = cp.Variable(n, name="x")
    p = cp.Parameter((n, n), PSD=True, name="P")
    q = cp.Parameter(n, name="q")
    problem = cp.Problem(cp.Minimize(cp.quad_form(x, p) + q @ x), [cp.sum(x) == 1, x >= 0])

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray]:
        raw = rng.normal(size=(n, n))
        matrix = raw.T @ raw + 0.25 * np.eye(n)
        return {"P": matrix, "q": rng.normal(size=n)}

    parameters = {"P": p, "q": q}
    _assign(parameters, sample(np.random.default_rng(3), 0))
    return ProblemFixture(problem, {"x": x}, parameters, sample)


def _socp(m: int = 8, n: int = 4) -> ProblemFixture:
    x = cp.Variable(n, name="x")
    a = cp.Parameter((m, n), name="A")
    b = cp.Parameter(m, name="b")
    rho = cp.Parameter(nonneg=True, name="rho")
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(a @ x - b) + 0.1 * cp.sum_squares(x)),
        [cp.norm(x, 2) <= rho, x >= 0, cp.sum(x) <= 1],
    )

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray | float]:
        return {"A": rng.normal(size=(m, n)), "b": rng.normal(size=m), "rho": float(rng.uniform(0.5, 1.5))}

    parameters = {"A": a, "b": b, "rho": rho}
    _assign(parameters, sample(np.random.default_rng(4), 0))
    return ProblemFixture(problem, {"x": x}, parameters, sample)


def _flow_lp(commodities: int = 2) -> ProblemFixture:
    edge_count = 2 * commodities
    f = cp.Variable(edge_count, name="f")
    c = cp.Parameter(edge_count, name="c")
    d = cp.Parameter(commodities, name="d")
    constraints = [f >= 0, f <= 3]
    for index in range(commodities):
        constraints.append(f[index] + f[index + commodities] == d[index])
    constraints.append(cp.sum(f[:commodities]) <= float(commodities))
    problem = cp.Problem(
        cp.Minimize(c @ f),
        constraints,
    )

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray]:
        return {"c": rng.uniform(0.1, 3.0, size=edge_count), "d": rng.uniform(0.2, 1.8, size=commodities)}

    parameters = {"c": c, "d": d}
    _assign(parameters, sample(np.random.default_rng(5), 0))
    return ProblemFixture(problem, {"f": f}, parameters, sample)


def _mpc(horizon: int = 6) -> ProblemFixture:
    x = cp.Variable(horizon + 1, name="x")
    u = cp.Variable(horizon, name="u")
    x0 = cp.Parameter(name="x0")
    reference = cp.Parameter(horizon + 1, name="reference")
    constraints = [x[0] == x0, cp.abs(u) <= 1.0]
    for t in range(horizon):
        constraints.append(x[t + 1] == x[t] + u[t])
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(x - reference) + 0.05 * cp.sum_squares(u)),
        constraints,
    )

    def sample(rng: np.random.Generator, index: int) -> dict[str, np.ndarray | float]:
        grid = np.linspace(0.0, 1.0, horizon + 1)
        return {
            "x0": float(rng.normal(scale=0.2)),
            "reference": np.sin(grid * np.pi + 0.1 * index),
        }

    parameters = {"x0": x0, "reference": reference}
    _assign(parameters, sample(np.random.default_rng(6), 0))
    return ProblemFixture(problem, {"x": x, "u": u}, parameters, sample)


def _portfolio(n: int = 8) -> ProblemFixture:
    w = cp.Variable(n, name="w")
    mu = cp.Parameter(n, name="mu")
    target = cp.Parameter(name="target")
    risk = cp.Parameter(nonneg=True, name="risk")
    covariance = np.diag(np.linspace(0.5, 2.0, n))
    problem = cp.Problem(
        cp.Minimize(risk * cp.quad_form(w, covariance) - mu @ w),
        [cp.sum(w) == 1, w >= 0, mu @ w >= target],
    )

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray | float]:
        returns = rng.uniform(0.01, 0.20, size=n)
        return {"mu": returns, "target": float(np.quantile(returns, 0.35)), "risk": float(rng.uniform(0.1, 1.0))}

    parameters = {"mu": mu, "target": target, "risk": risk}
    _assign(parameters, sample(np.random.default_rng(7), 0))
    return ProblemFixture(problem, {"w": w}, parameters, sample)


def _case(
    name: str,
    description: str,
    factory: Callable[[], ProblemFixture],
    family: str,
    size_label: str,
    size: int,
) -> ProblemCase:
    return ProblemCase(name, description, factory, family, size_label, size)


CASES: dict[str, ProblemCase] = {
    "nnls": _case("nnls", "Nonnegative least squares QP", _nnls, "nnls", "small", 4),
    "nnls_small": _case("nnls_small", "Nonnegative least squares QP, n=4", lambda: _nnls(8, 4), "nnls", "small", 4),
    "nnls_medium": _case("nnls_medium", "Nonnegative least squares QP, n=8", lambda: _nnls(16, 8), "nnls", "medium", 8),
    "nnls_large": _case("nnls_large", "Nonnegative least squares QP, n=16", lambda: _nnls(32, 16), "nnls", "large", 16),
    "box_qp": _case("box_qp", "Box-constrained equality QP", _box_qp, "box_qp", "small", 6),
    "box_qp_small": _case("box_qp_small", "Box-constrained equality QP, n=6", lambda: _box_qp(6), "box_qp", "small", 6),
    "box_qp_medium": _case("box_qp_medium", "Box-constrained equality QP, n=12", lambda: _box_qp(12), "box_qp", "medium", 12),
    "box_qp_large": _case("box_qp_large", "Box-constrained equality QP, n=24", lambda: _box_qp(24), "box_qp", "large", 24),
    "parametric_quad_form": _case(
        "parametric_quad_form",
        "QP with PSD matrix parameter",
        _parametric_quad_form,
        "parametric_quad_form",
        "small",
        4,
    ),
    "parametric_quad_form_small": _case(
        "parametric_quad_form_small",
        "QP with PSD matrix parameter, n=4",
        lambda: _parametric_quad_form(4),
        "parametric_quad_form",
        "small",
        4,
    ),
    "parametric_quad_form_medium": _case(
        "parametric_quad_form_medium",
        "QP with PSD matrix parameter, n=8",
        lambda: _parametric_quad_form(8),
        "parametric_quad_form",
        "medium",
        8,
    ),
    "parametric_quad_form_large": _case(
        "parametric_quad_form_large",
        "QP with PSD matrix parameter, n=12",
        lambda: _parametric_quad_form(12),
        "parametric_quad_form",
        "large",
        12,
    ),
    "socp": _case("socp", "Second-order cone least-squares problem", _socp, "socp", "small", 4),
    "socp_small": _case("socp_small", "Second-order cone least-squares problem, n=4", lambda: _socp(8, 4), "socp", "small", 4),
    "socp_medium": _case(
        "socp_medium", "Second-order cone least-squares problem, n=8", lambda: _socp(16, 8), "socp", "medium", 8
    ),
    "socp_large": _case(
        "socp_large", "Second-order cone least-squares problem, n=16", lambda: _socp(32, 16), "socp", "large", 16
    ),
    "flow_lp": _case("flow_lp", "Small network-flow LP", _flow_lp, "flow_lp", "small", 2),
    "flow_lp_small": _case("flow_lp_small", "Network-flow LP, commodities=2", lambda: _flow_lp(2), "flow_lp", "small", 2),
    "flow_lp_medium": _case(
        "flow_lp_medium", "Network-flow LP, commodities=4", lambda: _flow_lp(4), "flow_lp", "medium", 4
    ),
    "flow_lp_large": _case("flow_lp_large", "Network-flow LP, commodities=8", lambda: _flow_lp(8), "flow_lp", "large", 8),
    "mpc": _case("mpc", "Paper-inspired scalar MPC QP", _mpc, "mpc", "small", 6),
    "mpc_small": _case("mpc_small", "Scalar MPC QP, horizon=6", lambda: _mpc(6), "mpc", "small", 6),
    "mpc_medium": _case("mpc_medium", "Scalar MPC QP, horizon=12", lambda: _mpc(12), "mpc", "medium", 12),
    "mpc_large": _case("mpc_large", "Scalar MPC QP, horizon=24", lambda: _mpc(24), "mpc", "large", 24),
    "portfolio": _case("portfolio", "Paper-inspired long-only portfolio QP", _portfolio, "portfolio", "small", 8),
    "portfolio_small": _case(
        "portfolio_small", "Long-only portfolio QP, n=8", lambda: _portfolio(8), "portfolio", "small", 8
    ),
    "portfolio_medium": _case(
        "portfolio_medium", "Long-only portfolio QP, n=16", lambda: _portfolio(16), "portfolio", "medium", 16
    ),
    "portfolio_large": _case(
        "portfolio_large", "Long-only portfolio QP, n=32", lambda: _portfolio(32), "portfolio", "large", 32
    ),
}


def selected_cases(names: tuple[str, ...] | list[str]) -> list[ProblemCase]:
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise KeyError(f"unknown benchmark problems: {', '.join(unknown)}")
    return [CASES[name] for name in names]
