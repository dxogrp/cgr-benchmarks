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


FAMILY_SIZE_AXIS_LABELS = {
    "nnls": "features n (samples = 2n)",
    "box_qp": "decision variables n",
    "parametric_quad_form": "decision variables n",
    "socp": "decision variables n (constraints = 2n)",
    "trace_sdp": "matrix dimension n",
    "covariance_sdp": "matrix dimension n",
    "flow_lp": "commodities",
    "mpc": "prediction horizon",
    "portfolio": "assets n",
    "lasso": "features n (samples = 2n)",
    "svm": "features n (samples = 3n)",
    "logistic_regression": "features n (samples = 3n)",
}


def size_axis_label(family: str) -> str:
    return FAMILY_SIZE_AXIS_LABELS.get(family, "problem size")


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


def _flow_edges(node_count: int) -> list[tuple[int, int]]:
    edges = [(node, node + 1) for node in range(node_count - 1)]
    edges.extend((node, node + 2) for node in range(node_count - 2))
    edges.append((0, node_count - 1))
    return edges


def _flow_incidence(node_count: int, edges: list[tuple[int, int]]) -> np.ndarray:
    incidence = np.zeros((node_count, len(edges)))
    for edge_index, (tail, head) in enumerate(edges):
        incidence[tail, edge_index] = 1.0
        incidence[head, edge_index] = -1.0
    return incidence


def _flow_lp(commodities: int = 2) -> ProblemFixture:
    node_count = max(4, commodities + 3)
    edges = _flow_edges(node_count)
    edge_count = len(edges)
    incidence = _flow_incidence(node_count, edges)
    balances = np.zeros((commodities, node_count))
    for commodity in range(commodities):
        source = 0
        sink = node_count - 1 - (commodity % min(2, node_count - 1))
        balances[commodity, source] = 1.0
        balances[commodity, sink] = -1.0

    f = cp.Variable((commodities, edge_count), name="f")
    c = cp.Parameter((commodities, edge_count), name="c")
    d = cp.Parameter(commodities, nonneg=True, name="d")
    capacity = cp.Parameter(edge_count, nonneg=True, name="capacity")
    constraints = [f >= 0, cp.sum(f, axis=0) <= capacity]
    for commodity in range(commodities):
        constraints.append(incidence @ f[commodity, :] == d[commodity] * balances[commodity])
    problem = cp.Problem(cp.Minimize(cp.sum(cp.multiply(c, f))), constraints)

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray]:
        demand = rng.uniform(0.15, 0.55, size=commodities)
        base_capacity = 0.75 * float(demand.sum()) + 0.4
        return {
            "c": rng.uniform(0.1, 3.0, size=(commodities, edge_count)),
            "d": demand,
            "capacity": rng.uniform(base_capacity, base_capacity + 1.0, size=edge_count),
        }

    parameters = {"c": c, "d": d, "capacity": capacity}
    _assign(parameters, sample(np.random.default_rng(5), 0))
    return ProblemFixture(problem, {"f": f}, parameters, sample)


def _mpc(horizon: int = 6) -> ProblemFixture:
    state_dim = 3
    control_dim = 2
    a_matrix = np.array(
        [
            [0.92, 0.12, 0.00],
            [0.00, 0.85, 0.10],
            [0.05, 0.00, 0.88],
        ]
    )
    b_matrix = np.array(
        [
            [0.18, 0.02],
            [0.04, 0.15],
            [0.10, -0.08],
        ]
    )
    state_weight_sqrt = np.sqrt(np.array([[1.8], [1.0], [0.6]]))
    control_weight_sqrt = np.sqrt(np.array([[0.08], [0.05]]))

    x = cp.Variable((state_dim, horizon + 1), name="x")
    u = cp.Variable((control_dim, horizon), name="u")
    x0 = cp.Parameter(state_dim, name="x0")
    reference = cp.Parameter((state_dim, horizon + 1), name="reference")
    constraints = [
        x[:, 0] == x0,
        cp.abs(u) <= 1.0,
        cp.abs(x[0:2, :]) <= 2.5,
    ]
    for t in range(horizon):
        constraints.append(x[:, t + 1] == a_matrix @ x[:, t] + b_matrix @ u[:, t])
    if horizon > 1:
        constraints.append(cp.abs(u[:, 1:] - u[:, :-1]) <= 0.35)
    objective = (
        cp.sum_squares(cp.multiply(state_weight_sqrt, x - reference))
        + cp.sum_squares(cp.multiply(control_weight_sqrt, u))
        + 0.03 * cp.sum_squares(x[:, horizon])
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)

    def sample(rng: np.random.Generator, index: int) -> dict[str, np.ndarray]:
        grid = np.linspace(0.0, 1.0, horizon + 1)
        phase = 0.15 * index
        path = np.vstack(
            [
                0.7 * np.sin(np.pi * grid + phase),
                0.5 * np.cos(0.5 * np.pi * grid + phase),
                0.25 * np.sin(2.0 * np.pi * grid + 0.5 * phase),
            ]
        )
        return {
            "x0": rng.normal(scale=0.15, size=state_dim),
            "reference": path,
        }

    parameters = {"x0": x0, "reference": reference}
    _assign(parameters, sample(np.random.default_rng(6), 0))
    return ProblemFixture(problem, {"x": x, "u": u}, parameters, sample)


def _sector_matrix(n: int, sector_count: int) -> np.ndarray:
    matrix = np.zeros((sector_count, n))
    for asset in range(n):
        matrix[asset % sector_count, asset] = 1.0
    return matrix


def _portfolio(n: int = 8) -> ProblemFixture:
    factor_count = max(1, min(4, n // 2))
    sector_count = max(1, min(4, n))
    asset_grid = np.linspace(0.0, 1.0, n)
    factor_columns = [
        np.ones(n),
        np.sin(np.pi * asset_grid),
        np.cos(2.0 * np.pi * asset_grid),
        np.linspace(-1.0, 1.0, n),
    ]
    factor_exposure = np.vstack(factor_columns[:factor_count]).T
    idiosyncratic_sqrt = np.sqrt(np.linspace(0.2, 1.1, n))
    sector_matrix = _sector_matrix(n, sector_count)
    max_position = min(0.7, max(0.12, 2.5 / n))

    w = cp.Variable(n, name="w")
    mu = cp.Parameter(n, name="mu")
    target = cp.Parameter(name="target")
    previous = cp.Parameter(n, nonneg=True, name="previous")
    risk = cp.Parameter(nonneg=True, name="risk")
    factor_risk = cp.sum_squares(factor_exposure.T @ w)
    specific_risk = cp.sum_squares(cp.multiply(idiosyncratic_sqrt, w))
    transaction_penalty = 0.02 * cp.norm1(w - previous)
    objective = risk * (factor_risk + specific_risk) - mu @ w + transaction_penalty
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_position,
        cp.norm1(w - previous) <= 0.75,
        sector_matrix @ w <= 0.65,
        mu @ w >= target,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray | float]:
        returns = rng.uniform(0.01, 0.20, size=n)
        holdings = rng.uniform(0.1, 1.0, size=n)
        holdings /= float(holdings.sum())
        return {
            "mu": returns,
            "target": float(np.quantile(returns, 0.30)),
            "previous": holdings,
            "risk": float(rng.uniform(0.2, 1.2)),
        }

    parameters = {"mu": mu, "target": target, "previous": previous, "risk": risk}
    _assign(parameters, sample(np.random.default_rng(7), 0))
    return ProblemFixture(problem, {"w": w}, parameters, sample)


def _lasso(m: int = 12, n: int = 6) -> ProblemFixture:
    x = cp.Variable(n, name="x")
    a = cp.Parameter((m, n), name="A")
    b = cp.Parameter(m, name="b")
    lambda_reg = cp.Parameter(nonneg=True, name="lambda_reg")
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(a @ x - b) + lambda_reg * cp.norm1(x)))

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray | float]:
        signal = rng.normal(size=n)
        design = rng.normal(size=(m, n)) / np.sqrt(float(n))
        response = design @ signal + rng.normal(scale=0.15, size=m)
        return {"A": design, "b": response, "lambda_reg": float(rng.uniform(0.03, 0.3))}

    parameters = {"A": a, "b": b, "lambda_reg": lambda_reg}
    _assign(parameters, sample(np.random.default_rng(10), 0))
    return ProblemFixture(problem, {"x": x}, parameters, sample)


def _classification_labels(m: int) -> np.ndarray:
    labels = np.ones(m)
    labels[1::2] = -1.0
    return labels


def _svm(m: int = 12, n: int = 4) -> ProblemFixture:
    w = cp.Variable(n, name="w")
    beta = cp.Variable(name="beta")
    slack = cp.Variable(m, name="slack")
    x_data = cp.Parameter((m, n), name="X")
    labels = _classification_labels(m)
    margin = cp.multiply(labels, x_data @ w + beta)
    penalty = 1.0
    constraints = [slack >= 1.0 - margin, slack >= 0]
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(w) + penalty * cp.sum(slack)), constraints)

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray]:
        centers = 0.6 * labels.reshape(m, 1) * np.linspace(0.5, 1.2, n)
        return {"X": centers + rng.normal(scale=0.8, size=(m, n))}

    parameters = {"X": x_data}
    _assign(parameters, sample(np.random.default_rng(11), 0))
    return ProblemFixture(problem, {"w": w, "beta": beta, "slack": slack}, parameters, sample)


def _logistic_regression(m: int = 12, n: int = 4) -> ProblemFixture:
    w = cp.Variable(n, name="w")
    beta = cp.Variable(name="beta")
    x_data = cp.Parameter((m, n), name="X")
    lambda_reg = cp.Parameter(nonneg=True, name="lambda_reg")
    labels = _classification_labels(m)
    margin = cp.multiply(labels, x_data @ w + beta)
    # QP-representable second-order surrogate of log(1 + exp(-margin)) around zero.
    surrogate_loss = cp.sum(0.125 * cp.square(margin) - 0.5 * margin + np.log(2.0)) / float(m)
    objective = surrogate_loss + 0.5 * lambda_reg * cp.sum_squares(w)
    problem = cp.Problem(cp.Minimize(objective))

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray | float]:
        centers = 0.5 * labels.reshape(m, 1) * np.linspace(0.4, 1.0, n)
        return {
            "X": centers + rng.normal(scale=0.9, size=(m, n)),
            "lambda_reg": float(rng.uniform(0.03, 0.3)),
        }

    parameters = {"X": x_data, "lambda_reg": lambda_reg}
    _assign(parameters, sample(np.random.default_rng(12), 0))
    return ProblemFixture(problem, {"w": w, "beta": beta}, parameters, sample)


def _trace_sdp(n: int = 3) -> ProblemFixture:
    x = cp.Variable((n, n), symmetric=True, name="X")
    c = cp.Parameter((n, n), symmetric=True, name="C")
    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(c, x))),
        [x >> 0, cp.trace(x) == 1],
    )

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray]:
        raw = rng.normal(size=(n, n))
        matrix = 0.5 * (raw + raw.T)
        matrix += np.diag(np.linspace(0.0, 0.25, n))
        return {"C": matrix}

    parameters = {"C": c}
    _assign(parameters, sample(np.random.default_rng(8), 0))
    return ProblemFixture(problem, {"X": x}, parameters, sample)


def _covariance_sdp(n: int = 3) -> ProblemFixture:
    sigma = cp.Variable((n, n), symmetric=True, name="Sigma")
    sample_covariance = cp.Parameter((n, n), symmetric=True, name="S")
    diagonal_target = cp.diag(cp.diag(sample_covariance))
    problem = cp.Problem(
        cp.Minimize(
            cp.sum_squares(sigma - sample_covariance)
            + 0.15 * cp.sum_squares(sigma - diagonal_target)
        ),
        [sigma >> 0, cp.diag(sigma) == cp.diag(sample_covariance)],
    )

    def sample(rng: np.random.Generator, _: int) -> dict[str, np.ndarray]:
        factor_count = max(1, min(3, n))
        factors = rng.normal(size=(n, factor_count))
        base = factors @ factors.T / factor_count
        noise = rng.normal(scale=0.35, size=(n, n))
        matrix = base + 0.5 * (noise + noise.T)
        diagonal = np.diag(base) + rng.uniform(0.2, 0.8, size=n)
        np.fill_diagonal(matrix, diagonal)
        return {"S": matrix}

    parameters = {"S": sample_covariance}
    _assign(parameters, sample(np.random.default_rng(9), 0))
    return ProblemFixture(problem, {"Sigma": sigma}, parameters, sample)


def _case(
    name: str,
    description: str,
    factory: Callable[[], ProblemFixture],
    family: str,
    size_label: str,
    size: int,
) -> ProblemCase:
    return ProblemCase(name, description, factory, family, size_label, size)


def _size_case(
    cases: dict[str, ProblemCase],
    family: str,
    label: str,
    description_prefix: str,
    size_text: str,
    factory: Callable[[], ProblemFixture],
    size: int,
) -> None:
    name = f"{family}_{label}"
    cases[name] = _case(name, f"{description_prefix}, {size_text}", factory, family, label, size)


CASES: dict[str, ProblemCase] = {
    "nnls": _case("nnls", "Nonnegative least squares QP", _nnls, "nnls", "small", 4),
    "box_qp": _case("box_qp", "Box-constrained equality QP", _box_qp, "box_qp", "small", 6),
    "parametric_quad_form": _case(
        "parametric_quad_form",
        "QP with PSD matrix parameter",
        _parametric_quad_form,
        "parametric_quad_form",
        "small",
        4,
    ),
    "socp": _case("socp", "Second-order cone least-squares problem", _socp, "socp", "small", 4),
    "trace_sdp": _case("trace_sdp", "Trace-normalized semidefinite program", _trace_sdp, "trace_sdp", "small", 3),
    "covariance_sdp": _case(
        "covariance_sdp",
        "Shrinkage covariance estimation SDP",
        _covariance_sdp,
        "covariance_sdp",
        "small",
        3,
    ),
    "flow_lp": _case("flow_lp", "Multi-commodity network-flow LP", _flow_lp, "flow_lp", "small", 2),
    "mpc": _case("mpc", "Multi-state tracking MPC QP", _mpc, "mpc", "small", 6),
    "portfolio": _case("portfolio", "Factor-risk portfolio QP", _portfolio, "portfolio", "small", 8),
    "lasso": _case("lasso", "L1-regularized least-squares problem", _lasso, "lasso", "small", 6),
    "svm": _case("svm", "Linear soft-margin SVM QP", _svm, "svm", "small", 4),
    "logistic_regression": _case(
        "logistic_regression",
        "Quadratic-surrogate logistic regression",
        _logistic_regression,
        "logistic_regression",
        "small",
        4,
    ),
}

for label, n in [("tiny", 2), ("small", 4), ("medium", 8), ("large", 16), ("xlarge", 32)]:
    _size_case(CASES, "nnls", label, "Nonnegative least squares QP", f"n={n}", lambda n=n: _nnls(2 * n, n), n)
    _size_case(
        CASES,
        "socp",
        label,
        "Second-order cone least-squares problem",
        f"n={n}",
        lambda n=n: _socp(2 * n, n),
        n,
    )

for label, n in [("tiny", 3), ("small", 6), ("medium", 12), ("large", 24), ("xlarge", 48)]:
    _size_case(CASES, "box_qp", label, "Box-constrained equality QP", f"n={n}", lambda n=n: _box_qp(n), n)

for label, n in [("tiny", 2), ("small", 4), ("medium", 8), ("large", 12), ("xlarge", 16)]:
    _size_case(
        CASES,
        "parametric_quad_form",
        label,
        "QP with PSD matrix parameter",
        f"n={n}",
        lambda n=n: _parametric_quad_form(n),
        n,
    )

for label, n in [("tiny", 2), ("small", 3), ("medium", 4), ("large", 6), ("xlarge", 8)]:
    _size_case(
        CASES,
        "trace_sdp",
        label,
        "Trace-normalized semidefinite program",
        f"n={n}",
        lambda n=n: _trace_sdp(n),
        n,
    )

for label, n in [("tiny", 2), ("small", 3), ("medium", 4), ("large", 6), ("xlarge", 8)]:
    _size_case(
        CASES,
        "covariance_sdp",
        label,
        "Shrinkage covariance estimation SDP",
        f"n={n}",
        lambda n=n: _covariance_sdp(n),
        n,
    )

for label, commodities in [("tiny", 1), ("small", 2), ("medium", 4), ("large", 8), ("xlarge", 16)]:
    _size_case(
        CASES,
        "flow_lp",
        label,
        "Network-flow LP",
        f"commodities={commodities}",
        lambda commodities=commodities: _flow_lp(commodities),
        commodities,
    )

for label, horizon in [("tiny", 3), ("small", 6), ("medium", 12), ("large", 24), ("xlarge", 48)]:
    _size_case(
        CASES,
        "mpc",
        label,
        "Multi-state tracking MPC QP",
        f"horizon={horizon}",
        lambda horizon=horizon: _mpc(horizon),
        horizon,
    )

for label, n in [("tiny", 4), ("small", 8), ("medium", 16), ("large", 32), ("xlarge", 64)]:
    _size_case(
        CASES,
        "portfolio",
        label,
        "Factor-risk portfolio QP",
        f"n={n}",
        lambda n=n: _portfolio(n),
        n,
    )

for label, n in [("tiny", 4), ("small", 6), ("medium", 10), ("large", 16), ("xlarge", 24)]:
    _size_case(
        CASES,
        "lasso",
        label,
        "L1-regularized least-squares problem",
        f"n={n}",
        lambda n=n: _lasso(2 * n, n),
        n,
    )

for label, n in [("tiny", 3), ("small", 4), ("medium", 6), ("large", 8), ("xlarge", 12)]:
    _size_case(
        CASES,
        "svm",
        label,
        "Linear soft-margin SVM QP",
        f"n={n}",
        lambda n=n: _svm(3 * n, n),
        n,
    )

for label, n in [("tiny", 3), ("small", 4), ("medium", 6), ("large", 8), ("xlarge", 10)]:
    _size_case(
        CASES,
        "logistic_regression",
        label,
        "Quadratic-surrogate logistic regression",
        f"n={n}",
        lambda n=n: _logistic_regression(3 * n, n),
        n,
    )


def selected_cases(names: tuple[str, ...] | list[str]) -> list[ProblemCase]:
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise KeyError(f"unknown benchmark problems: {', '.join(unknown)}")
    return [CASES[name] for name in names]
