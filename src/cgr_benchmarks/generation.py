from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import cvxgenrust as cgr

from .config import BenchmarkConfig
from .problems import ProblemCase, selected_cases


def cgr_dir(config: BenchmarkConfig, problem_name: str) -> Path:
    return config.generated_dir / problem_name / "cgr_code"


def cgr_python_source_dir(config: BenchmarkConfig, problem_name: str) -> Path:
    return cgr_dir(config, problem_name) / "python"


def cgr_python_package(config: BenchmarkConfig, problem_name: str) -> str:
    pyproject_path = cgr_dir(config, problem_name) / "pyproject.toml"
    if pyproject_path.exists():
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        module_name = data.get("tool", {}).get("maturin", {}).get("module-name")
        if isinstance(module_name, str) and "." in module_name:
            return module_name.split(".", 1)[0]
    return cgr_dir(config, problem_name).name


def cpg_dir(config: BenchmarkConfig, problem_name: str) -> Path:
    return config.generated_dir / problem_name / "cpg_code"


def generate_cvxgenrust(case: ProblemCase, output_dir: Path) -> None:
    fixture = case.factory()
    output_dir.mkdir(parents=True, exist_ok=True)
    cgr.generate_code(fixture.problem, code_dir=output_dir.resolve(), module_name=f"{case.name}_cgr")


def generate_cvxpygen(case: ProblemCase, output_dir: Path, solver: str) -> None:
    cpg = importlib.import_module("cvxpygen.cpg")
    fixture = case.factory()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    try:
        os.chdir(output_dir.parent)
        cpg.generate_code(fixture.problem, code_dir=output_dir.name, solver=solver, wrapper=True)
    finally:
        os.chdir(cwd)


def generate_all(config: BenchmarkConfig, clean: bool = False) -> None:
    if clean and config.generated_dir.exists():
        shutil.rmtree(config.generated_dir)
    for case in selected_cases(config.enabled_problems):
        tools = config.enabled_tools_for(case.name, case.family)
        if "cgr_python" in tools or "cgr_rust" in tools:
            generate_cvxgenrust(case, cgr_dir(config, case.name))
        if "cvxpygen" in tools or "cvxpygen_c" in tools:
            generate_cvxpygen(case, cpg_dir(config, case.name), config.cvxpygen_solver)


def build_cvxgenrust(config: BenchmarkConfig, case: ProblemCase) -> None:
    cmd = ["cargo", "build", "--lib", "--manifest-path", str(cgr_dir(config, case.name) / "Cargo.toml")]
    if config.rust_release:
        cmd.append("--release")
    subprocess.run(cmd, check=True)


def build_cvxpygen(config: BenchmarkConfig, case: ProblemCase) -> None:
    solver_module = cpg_dir(config, case.name) / "cpg_solver.py"
    if not solver_module.exists():
        return
    script = (
        "import importlib.util, pathlib, sys\n"
        f"path = pathlib.Path({str(solver_module)!r})\n"
        "sys.path.insert(0, str(path.parent.parent.resolve()))\n"
        "spec = importlib.util.spec_from_file_location('cpg_solver_build', path)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
    )
    subprocess.run(["python", "-c", script], check=True)
