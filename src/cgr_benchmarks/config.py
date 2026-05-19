from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    warmups: int
    samples: int
    seed: int
    enabled_problems: tuple[str, ...]
    enabled_tools: tuple[str, ...]
    artifacts_dir: Path
    generated_dir: Path
    results_dir: Path
    cvxpy_solver: str
    cvxpygen_solver: str
    rust_release: bool
    strip_artifacts: bool

    @classmethod
    def load(cls, path: Path) -> "BenchmarkConfig":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        benchmark = data.get("benchmark", {})
        paths = data.get("paths", {})
        cvxpy = data.get("cvxpy", {})
        cvxpygen = data.get("cvxpygen", {})
        rust = data.get("rust", {})
        return cls(
            path=path,
            warmups=int(benchmark.get("warmups", 3)),
            samples=int(benchmark.get("samples", 30)),
            seed=int(benchmark.get("seed", 1729)),
            enabled_problems=tuple(benchmark.get("enabled_problems", [])),
            enabled_tools=tuple(benchmark.get("enabled_tools", [])),
            artifacts_dir=Path(paths.get("artifacts_dir", "artifacts")),
            generated_dir=Path(paths.get("generated_dir", "artifacts/generated")),
            results_dir=Path(paths.get("results_dir", "results")),
            cvxpy_solver=str(cvxpy.get("solver", "CLARABEL")),
            cvxpygen_solver=str(cvxpygen.get("solver", "CLARABEL")),
            rust_release=bool(rust.get("release", True)),
            strip_artifacts=bool(rust.get("strip", True)),
        )

