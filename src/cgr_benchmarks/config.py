from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    warmups: int
    samples: int
    seed: int
    enabled_problems: tuple[str, ...]
    enabled_tools: tuple[str, ...]
    tool_skips: dict[str, tuple[str, ...]]
    artifacts_dir: Path
    generated_dir: Path
    results_dir: Path
    cvxpy_solver: str
    cvxpygen_solver: str
    solver_settings: dict[str, Any]
    rust_release: bool

    @classmethod
    def load(cls, path: Path) -> "BenchmarkConfig":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        benchmark = data.get("benchmark", {})
        paths = data.get("paths", {})
        cvxpy = data.get("cvxpy", {})
        cvxpygen = data.get("cvxpygen", {})
        solver_settings = data.get("solver_settings", {})
        rust = data.get("rust", {})
        tool_skips = {
            str(name): tuple(str(tool) for tool in tools)
            for name, tools in data.get("tool_skips", {}).items()
        }
        return cls(
            path=path,
            warmups=int(benchmark.get("warmups", 3)),
            samples=int(benchmark.get("samples", 30)),
            seed=int(benchmark.get("seed", 1729)),
            enabled_problems=tuple(benchmark.get("enabled_problems", [])),
            enabled_tools=tuple(benchmark.get("enabled_tools", [])),
            tool_skips=tool_skips,
            artifacts_dir=Path(paths.get("artifacts_dir", "artifacts")),
            generated_dir=Path(paths.get("generated_dir", "artifacts/generated")),
            results_dir=Path(paths.get("results_dir", "results")),
            cvxpy_solver=str(cvxpy.get("solver", "CLARABEL")),
            cvxpygen_solver=str(cvxpygen.get("solver", "CLARABEL")),
            solver_settings=dict(solver_settings),
            rust_release=bool(rust.get("release", True)),
        )


    def enabled_tools_for(self, problem_name: str, problem_family: str) -> tuple[str, ...]:
        skipped = set(self.tool_skips.get(problem_family, ())) | set(self.tool_skips.get(problem_name, ()))
        return tuple(tool for tool in self.enabled_tools if tool not in skipped)
