from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import BenchmarkConfig
from .environment import write_metadata
from .generation import (
    build_cvxgenrust,
    build_cvxpygen,
    cgr_dir,
    cgr_python_package,
    cgr_python_source_dir,
    cpg_dir,
    generate_all,
)
from .problems import ProblemCase, ProblemFixture, selected_cases
from .report import write_report


@dataclass(frozen=True)
class SolveRecord:
    problem: str
    problem_family: str
    size_label: str
    size: int
    tool: str
    sample: int
    seconds: float
    status: str
    objective: float | None
    iterations: int | None
    rel_objective_error: float | None
    rel_solution_error: float | None
    error: str | None


def _assign_values(fixture: ProblemFixture, values: dict[str, np.ndarray | float]) -> None:
    for name, value in values.items():
        fixture.parameters[name].value = value


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _solution_vector(fixture: ProblemFixture) -> np.ndarray | None:
    values = []
    for variable in fixture.variables.values():
        if variable.value is None:
            return None
        values.append(np.asarray(variable.value).reshape(-1, order="F"))
    if not values:
        return None
    return np.concatenate(values)


def _relative_error(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    denom = max(1.0, abs(reference))
    return abs(value - reference) / denom


def _relative_vector_error(value: np.ndarray | None, reference: np.ndarray | None) -> float | None:
    if value is None or reference is None:
        return None
    denom = max(1.0, float(np.linalg.norm(reference)))
    return float(np.linalg.norm(value - reference) / denom)


def _stats_iterations(problem: cp.Problem) -> int | None:
    stats = getattr(problem, "solver_stats", None)
    return getattr(stats, "num_iters", None) if stats is not None else None


def _time_call(callable_) -> tuple[float, object]:
    start = time.perf_counter()
    value = callable_()
    return time.perf_counter() - start, value


def _run_cvxpy_direct(
    fixture: ProblemFixture,
    solver: str,
) -> tuple[float, float | None, str, int | None, np.ndarray | None]:
    elapsed, objective = _time_call(lambda: fixture.problem.solve(solver=getattr(cp, solver), warm_start=False))
    return (
        elapsed,
        float(objective) if objective is not None else None,
        str(fixture.problem.status),
        _stats_iterations(fixture.problem),
        _solution_vector(fixture),
    )


def _register_cgr(fixture: ProblemFixture, config: BenchmarkConfig, case: ProblemCase) -> None:
    package_name = cgr_python_package(config, case.name)
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    python_source_dir = str(cgr_python_source_dir(config, case.name).resolve())
    sys.path.insert(0, python_source_dir)
    try:
        module = importlib.import_module(f"{package_name}.cgr_solver")
    finally:
        if python_source_dir in sys.path:
            sys.path.remove(python_source_dir)
    fixture.problem.register_solve("CGR", module.cgr_solve)


def _register_cpg(fixture: ProblemFixture, config: BenchmarkConfig, case: ProblemCase) -> None:
    solver_path = cpg_dir(config, case.name) / "cpg_solver.py"
    for module_name in list(sys.modules):
        if module_name == "cpg_code" or module_name.startswith("cpg_code."):
            sys.modules.pop(module_name, None)
    sys.path.insert(0, str(solver_path.parent.parent.resolve()))
    module = _load_module(solver_path, f"{case.name}_cpg_solver")
    fixture.problem.register_solve("CPG", module.cpg_solve)


def _json_ready(value: np.ndarray | float) -> object:
    array = np.asarray(value)
    if array.shape == ():
        return float(array)
    return array.tolist()


def _run_cvxpygen_subprocess(
    config: BenchmarkConfig,
    case: ProblemCase,
    values_by_sample: list[dict[str, np.ndarray | float]],
) -> list[tuple[int, float, float | None, str, int | None, list[float] | None, str | None]]:
    sample_path = config.artifacts_dir / "cvxpygen_samples" / f"{case.name}.json"
    output_path = config.artifacts_dir / "cvxpygen_samples" / f"{case.name}.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [{name: _json_ready(value) for name, value in values.items()} for values in values_by_sample]
    sample_path.write_text(json.dumps(serializable), encoding="utf-8")
    if output_path.exists():
        output_path.unlink()

    solver_path = (cpg_dir(config, case.name) / "cpg_solver.py").resolve()
    generated_parent = solver_path.parent.parent.resolve()
    script = r"""
import contextlib
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

from cgr_benchmarks.problems import CASES

case_name, solver_path, generated_parent, sample_path, output_path = sys.argv[1:6]
sys.path.insert(0, generated_parent)
spec = importlib.util.spec_from_file_location(f"{case_name}_cpg_solver", solver_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fixture = CASES[case_name].factory()
fixture.problem.register_solve("CPG", module.cpg_solve)
parameter_names = list(fixture.parameters)
samples = json.loads(pathlib.Path(sample_path).read_text(encoding="utf-8"))

def assign(values):
    for name, value in values.items():
        parameter = fixture.parameters[name]
        parameter.value = np.asarray(value).reshape(parameter.shape, order="C") if parameter.shape else value

def solution_vector():
    values = []
    for variable in fixture.variables.values():
        if variable.value is None:
            return None
        values.append(np.asarray(variable.value).reshape(-1, order="F"))
    if not values:
        return None
    return np.concatenate(values).tolist()

with pathlib.Path(output_path).open("w", encoding="utf-8") as handle:
    for sample, values in enumerate(samples):
        try:
            assign(values)
            start = time.perf_counter()
            with open("/dev/null", "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
                objective = fixture.problem.solve(method="CPG", updated_params=parameter_names)
            seconds = time.perf_counter() - start
            stats = getattr(fixture.problem, "solver_stats", None)
            row = {
                "sample": sample,
                "seconds": seconds,
                "objective": None if objective is None else float(objective),
                "status": str(fixture.problem.status),
                "iterations": getattr(stats, "num_iters", None) if stats is not None else None,
                "solution": solution_vector(),
                "error": None,
            }
        except Exception as exc:
            row = {
                "sample": sample,
                "seconds": 0.0,
                "objective": None,
                "status": "error",
                "iterations": None,
                "solution": None,
                "error": repr(exc),
            }
        handle.write(json.dumps(row, sort_keys=True) + "\n")
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            case.name,
            str(solver_path),
            str(generated_parent),
            str(sample_path),
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    rows = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.append(
            (
                int(row["sample"]),
                float(row["seconds"]),
                row["objective"],
                str(row["status"]),
                row["iterations"],
                row["solution"],
                row["error"],
            )
        )
    return rows


def _run_registered(
    fixture: ProblemFixture,
    method: str,
    parameter_names: list[str],
) -> tuple[float, float | None, str, int | None, np.ndarray | None]:
    elapsed, objective = _time_call(lambda: fixture.problem.solve(method=method, updated_params=parameter_names))
    return (
        elapsed,
        float(objective) if objective is not None else None,
        str(fixture.problem.status),
        _stats_iterations(fixture.problem),
        _solution_vector(fixture),
    )


def _write_cgr_rust_driver(config: BenchmarkConfig, case: ProblemCase, fixture: ProblemFixture) -> Path:
    driver_dir = config.artifacts_dir / "rust_drivers" / case.name
    driver_dir.mkdir(parents=True, exist_ok=True)
    crate_dir = cgr_dir(config, case.name).resolve()
    solution_extract_lines = [
        f"        solution_values.extend(problem.extract_variable({json.dumps(name)}, &solution.x)?);"
        for name in fixture.variables
    ]
    (driver_dir / "Cargo.toml").write_text(
        "\n".join(
            [
                "[package]",
                f'name = "{case.name}_cgr_driver"',
                'version = "0.1.0"',
                'edition = "2024"',
                "",
                "[dependencies]",
                f'{case.name}_cgr = {{ path = "{crate_dir.as_posix()}" }}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (driver_dir / "src").mkdir(exist_ok=True)
    (driver_dir / "src" / "main.rs").write_text(
        "\n".join(
            [
                f"use {case.name}_cgr::CGRProblem;",
                "use std::env;",
                "use std::fs;",
                "use std::time::Instant;",
                "",
                "fn main() -> Result<(), Box<dyn std::error::Error>> {",
                "    let path = env::args().nth(1).ok_or(\"missing sample file path\")?;",
                "    let text = fs::read_to_string(path)?;",
                "    println!(\"sample,status,objective,seconds,solution\");",
                "    for (sample, line) in text.lines().enumerate() {",
                "        if line.trim().is_empty() {",
                "            continue;",
                "        }",
                "        let mut problem = CGRProblem::new();",
                "        let start = Instant::now();",
                "        for assignment in line.split(';') {",
                "            let Some((name, values_text)) = assignment.split_once('=') else { continue; };",
                "            let values: Result<Vec<f64>, _> = values_text",
                "                .split(',')",
                "                .filter(|value| !value.is_empty())",
                "                .map(str::parse::<f64>)",
                "                .collect();",
                "            problem.set_parameter(name, &values?)?;",
                "        }",
                "        let solution = problem.solve()?;",
                "        let seconds = start.elapsed().as_secs_f64();",
                "        let mut solution_values = Vec::new();",
                *solution_extract_lines,
                "        let solution_text = solution_values",
                "            .iter()",
                "            .map(|value| value.to_string())",
                "            .collect::<Vec<_>>()",
                "            .join(\"|\");",
                "        println!(",
                "            \"{},{},{},{},{}\",",
                "            sample, solution.status, solution.obj_val, seconds, solution_text",
                "        );",
                "    }",
                "    Ok(())",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return driver_dir


def _build_cgr_rust_driver(config: BenchmarkConfig, case: ProblemCase, fixture: ProblemFixture) -> Path:
    driver_dir = _write_cgr_rust_driver(config, case, fixture)
    cmd = ["cargo", "build", "--manifest-path", str(driver_dir / "Cargo.toml")]
    if config.rust_release:
        cmd.append("--release")
    subprocess.run(cmd, check=True)
    return driver_dir / "target" / ("release" if config.rust_release else "debug") / f"{case.name}_cgr_driver"


def _flatten_for_cgr_rust(name: str, value: np.ndarray | float, fixture: ProblemFixture) -> list[float]:
    parameter = fixture.parameters[name]
    array = np.asarray(value)
    if (
        bool(getattr(parameter, "attributes", {}).get("PSD"))
        and len(parameter.shape) == 2
        and parameter.shape[0] == parameter.shape[1]
    ):
        expected = int(parameter.shape[0] * (parameter.shape[0] + 1) / 2)
        if array.size != expected and array.ndim == 2:
            return array[np.triu_indices(parameter.shape[0])].astype(float).tolist()
    return array.reshape(-1, order="F").astype(float).tolist()


def _flatten_for_cvxpygen_c(name: str, value: np.ndarray | float, fixture: ProblemFixture) -> list[float]:
    return _flatten_for_cgr_rust(name, value, fixture)


def _write_cgr_rust_samples(
    config: BenchmarkConfig,
    case: ProblemCase,
    fixture: ProblemFixture,
    values_by_sample: list[dict[str, np.ndarray | float]],
) -> Path:
    sample_path = config.artifacts_dir / "rust_samples" / f"{case.name}.txt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for values in values_by_sample:
        assignments = []
        for name in fixture.parameters:
            flat = _flatten_for_cgr_rust(name, values[name], fixture)
            assignments.append(f"{name}=" + ",".join(repr(float(item)) for item in flat))
        lines.append(";".join(assignments))
    sample_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sample_path


def _run_cgr_rust_sample_file(
    executable: Path,
    sample_path: Path,
) -> list[tuple[int, float | None, str, float, list[float] | None]]:
    result = subprocess.run([str(executable), str(sample_path)], check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines()[1:]:
        sample_text, status, objective_text, seconds_text, solution_text = line.split(",", 4)
        solution = [float(value) for value in solution_text.split("|") if value]
        rows.append((int(sample_text), float(objective_text), status, float(seconds_text), solution or None))
    return rows


def _c_ident(name: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in name)


def _write_cvxpygen_c_driver(config: BenchmarkConfig, case: ProblemCase, fixture: ProblemFixture) -> Path:
    driver_dir = config.artifacts_dir / "cvxpygen_c_drivers" / case.name
    driver_dir.mkdir(parents=True, exist_ok=True)
    generated_root = cpg_dir(config, case.name).resolve()
    generated_c_dir = generated_root / "c"
    update_blocks = []
    for name, parameter in fixture.parameters.items():
        ident = _c_ident(name)
        if int(parameter.size) == 1 and not bool(getattr(parameter, "attributes", {}).get("PSD")):
            call = f"cpg_update_{ident}(values[0]);"
        else:
            call = (
                "for (size_t value_index = 0; value_index < value_count; value_index++) {\n"
                f"                cpg_update_{ident}((cpg_int)value_index, values[value_index]);\n"
                "            }"
            )
        update_blocks.append(
            f"""        if (strcmp(name, "{name}") == 0) {{
            {call}
        }}"""
        )
    driver_c = f"""
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cpg_solve.h"
#include "cpg_workspace.h"

static double elapsed_seconds(struct timespec start, struct timespec end) {{
    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}}

static void apply_assignment(char *assignment) {{
    char *equals = strchr(assignment, '=');
    if (!equals) {{
        return;
    }}
    *equals = '\\0';
    char *name = assignment;
    char *values_text = equals + 1;
    cpg_float values[4096];
    size_t value_count = 0;
    char *value_state = NULL;
    char *token = strtok_r(values_text, ",", &value_state);
    while (token != NULL && value_count < 4096) {{
        values[value_count++] = (cpg_float)strtod(token, NULL);
        token = strtok_r(NULL, ",", &value_state);
    }}
{chr(10).join(update_blocks)}
}}

int main(int argc, char **argv) {{
    if (argc < 2) {{
        fprintf(stderr, "usage: %s SAMPLE_FILE\\n", argv[0]);
        return 2;
    }}
    FILE *file = fopen(argv[1], "r");
    if (!file) {{
        perror("fopen");
        return 2;
    }}
    cpg_set_solver_default_settings();
    cpg_set_solver_verbose(0);
    printf("sample,status,objective,seconds\\n");
    char line[65536];
    size_t sample = 0;
    while (fgets(line, sizeof(line), file)) {{
        line[strcspn(line, "\\r\\n")] = '\\0';
        if (line[0] == '\\0') {{
            continue;
        }}
        struct timespec start;
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        char *assignment_state = NULL;
        char *assignment = strtok_r(line, ";", &assignment_state);
        while (assignment != NULL) {{
            apply_assignment(assignment);
            assignment = strtok_r(NULL, ";", &assignment_state);
        }}
        cpg_solve();
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("%zu,%lu,%.17g,%.17g\\n",
               sample,
               (unsigned long)CPG_Result.info->status,
               (double)CPG_Result.info->obj_val,
               elapsed_seconds(start, end));
        sample++;
    }}
    fclose(file);
    return 0;
}}
"""
    driver_path = driver_dir / "driver.c"
    driver_path.write_text(driver_c, encoding="utf-8")
    c_link = driver_dir / "c"
    if c_link.exists() or c_link.is_symlink():
        if c_link.resolve() != generated_c_dir:
            if c_link.is_dir() and not c_link.is_symlink():
                shutil.rmtree(c_link)
            else:
                c_link.unlink()
    if not c_link.exists():
        c_link.symlink_to(generated_c_dir, target_is_directory=True)
    (driver_dir / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.16)",
                f'project({case.name}_cvxpygen_c_driver C)',
                'if (NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")',
                '  set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES} -lm")',
                "endif()",
                "add_subdirectory(c cpg_generated)",
                "add_executable(cpg_c_driver driver.c ${cpg_head} ${cpg_src})",
                "target_include_directories(cpg_c_driver PRIVATE ${cpg_include})",
                "target_link_libraries(cpg_c_driver PRIVATE cpg libclarabel_c_static)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return driver_dir


def _build_cvxpygen_c_driver(config: BenchmarkConfig, case: ProblemCase, fixture: ProblemFixture) -> Path:
    driver_dir = _write_cvxpygen_c_driver(config, case, fixture)
    build_dir = driver_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    subprocess.run(["cmake", "-S", str(driver_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "cpg_c_driver", "--config", "Release", "-j"], check=True)
    return build_dir / "cpg_c_driver"


def _write_cvxpygen_c_samples(
    config: BenchmarkConfig,
    case: ProblemCase,
    fixture: ProblemFixture,
    values_by_sample: list[dict[str, np.ndarray | float]],
) -> Path:
    sample_path = config.artifacts_dir / "cvxpygen_c_samples" / f"{case.name}.txt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for values in values_by_sample:
        assignments = []
        for name in fixture.parameters:
            flat = _flatten_for_cvxpygen_c(name, values[name], fixture)
            assignments.append(f"{name}=" + ",".join(repr(float(item)) for item in flat))
        lines.append(";".join(assignments))
    sample_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sample_path


def _run_cvxpygen_c_sample_file(executable: Path, sample_path: Path) -> list[tuple[int, float | None, str, float]]:
    result = subprocess.run([str(executable), str(sample_path)], check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines()[1:]:
        sample_text, status, objective_text, seconds_text = line.split(",", 3)
        rows.append((int(sample_text), float(objective_text), status, float(seconds_text)))
    return rows


def _record_failure(problem: str, tool: str, sample: int, error: Exception) -> SolveRecord:
    case = selected_cases([problem])[0]
    return SolveRecord(
        problem,
        case.family,
        case.size_label,
        case.size,
        tool,
        sample,
        0.0,
        "error",
        None,
        None,
        None,
        None,
        repr(error),
    )


def _run_tool_samples(
    config: BenchmarkConfig,
    case: ProblemCase,
    tool: str,
    references: list[tuple[float | None, np.ndarray | None]],
    values_by_sample: list[dict[str, np.ndarray | float]],
) -> list[SolveRecord]:
    fixture = case.factory()
    parameter_names = list(fixture.parameters)
    if tool == "cgr_python":
        _register_cgr(fixture, config, case)
    elif tool == "cvxpygen":
        warmup_values = values_by_sample[: config.warmups]
        try:
            rows = _run_cvxpygen_subprocess(config, case, warmup_values + values_by_sample)
        except Exception as exc:
            return [_record_failure(case.name, tool, index, exc) for index in range(len(values_by_sample))]
        records = []
        for sample, seconds, objective, status, iterations, solution, error in rows:
            if sample < config.warmups:
                continue
            sample_index = sample - config.warmups
            reference_objective, reference_solution = references[sample_index]
            solution_array = np.asarray(solution) if solution is not None else None
            records.append(
                SolveRecord(
                    case.name,
                    case.family,
                    case.size_label,
                    case.size,
                    tool,
                    sample_index,
                    seconds,
                    status,
                    objective,
                    iterations,
                    _relative_error(objective, reference_objective),
                    _relative_vector_error(solution_array, reference_solution),
                    error,
                )
            )
        return records
    elif tool == "cgr_rust":
        executable = _build_cgr_rust_driver(config, case, fixture)
        warmup_values = values_by_sample[: config.warmups]
        sample_path = _write_cgr_rust_samples(config, case, fixture, warmup_values + values_by_sample)
        try:
            rows = _run_cgr_rust_sample_file(executable, sample_path)
        except Exception as exc:
            return [_record_failure(case.name, tool, index, exc) for index in range(len(values_by_sample))]
        records = []
        for sample, objective, status, seconds, solution in rows:
            if sample < config.warmups:
                continue
            sample_index = sample - config.warmups
            reference_objective, reference_solution = references[sample_index]
            solution_array = np.asarray(solution) if solution is not None else None
            records.append(
                SolveRecord(
                    case.name,
                    case.family,
                    case.size_label,
                    case.size,
                    tool,
                    sample_index,
                    seconds,
                    status,
                    objective,
                    None,
                    _relative_error(objective, reference_objective),
                    _relative_vector_error(solution_array, reference_solution),
                    None,
                )
            )
        return records
    elif tool == "cvxpygen_c":
        executable = _build_cvxpygen_c_driver(config, case, fixture)
        warmup_values = values_by_sample[: config.warmups]
        sample_path = _write_cvxpygen_c_samples(config, case, fixture, warmup_values + values_by_sample)
        try:
            rows = _run_cvxpygen_c_sample_file(executable, sample_path)
        except Exception as exc:
            return [_record_failure(case.name, tool, index, exc) for index in range(len(values_by_sample))]
        return [
            SolveRecord(
                case.name,
                case.family,
                case.size_label,
                case.size,
                tool,
                sample - config.warmups,
                seconds,
                status,
                objective,
                None,
                _relative_error(objective, references[sample - config.warmups][0]),
                None,
                None,
            )
            for sample, objective, status, seconds in rows
            if sample >= config.warmups
        ]
    records: list[SolveRecord] = []
    all_values = values_by_sample[: config.warmups] + values_by_sample
    for index, values in enumerate(all_values):
        sample_index = index - config.warmups
        try:
            _assign_values(fixture, values)
            if tool == "cvxpy_direct":
                elapsed, objective, status, iterations, solution = _run_cvxpy_direct(fixture, config.cvxpy_solver)
            elif tool == "cgr_python":
                elapsed, objective, status, iterations, solution = _run_registered(fixture, "CGR", parameter_names)
            else:
                raise ValueError(f"unknown tool: {tool}")
            if sample_index >= 0:
                reference_objective, reference_solution = references[sample_index] if references else (None, None)
                records.append(
                    SolveRecord(
                        case.name,
                        case.family,
                        case.size_label,
                        case.size,
                        tool,
                        sample_index,
                        elapsed,
                        status,
                        objective,
                        iterations,
                        _relative_error(objective, reference_objective),
                        _relative_vector_error(solution, reference_solution),
                        None,
                    )
                )
        except Exception as exc:
            if sample_index >= 0:
                records.append(_record_failure(case.name, tool, sample_index, exc))
    return records


def _reference_samples(
    config: BenchmarkConfig,
    case: ProblemCase,
    values_by_sample: list[dict[str, np.ndarray | float]],
) -> tuple[list[SolveRecord], list[tuple[float | None, np.ndarray | None]]]:
    records = _run_tool_samples(config, case, "cvxpy_direct", [], values_by_sample)
    references = [(record.objective, None) for record in records]
    fixture = case.factory()
    solved_references: list[tuple[float | None, np.ndarray | None]] = []
    for values in values_by_sample:
        _assign_values(fixture, values)
        try:
            fixture.problem.solve(solver=getattr(cp, config.cvxpy_solver), warm_start=False)
            solved_references.append((float(fixture.problem.value), _solution_vector(fixture)))
        except Exception:
            solved_references.append((None, None))
    return records, solved_references or references


def _write_records(config: BenchmarkConfig, solve_records: list[SolveRecord]) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in solve_records]
    pd.DataFrame(rows).to_csv(config.results_dir / "solve_times.csv", index=False)
    with (config.results_dir / "solve_times.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_benchmarks(config: BenchmarkConfig, generate: bool = True) -> None:
    if generate:
        generate_all(config)
    if config.artifacts_dir.exists():
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(config.results_dir)
    cases = selected_cases(config.enabled_problems)
    for case in cases:
        if "cgr_python" in config.enabled_tools or "cgr_rust" in config.enabled_tools:
            build_cvxgenrust(config, case)
        if "cvxpygen" in config.enabled_tools or "cvxpygen_c" in config.enabled_tools:
            build_cvxpygen(config, case)
    rng = np.random.default_rng(config.seed)
    solve_records: list[SolveRecord] = []
    for case in cases:
        fixture = case.factory()
        values = [fixture.sample_parameters(rng, index) for index in range(config.samples)]
        reference_records, references = _reference_samples(config, case, values)
        if "cvxpy_direct" in config.enabled_tools:
            solve_records.extend(reference_records)
        for tool in config.enabled_tools:
            if tool == "cvxpy_direct":
                continue
            solve_records.extend(_run_tool_samples(config, case, tool, references, values))
    _write_records(config, solve_records)
    write_report(config)
