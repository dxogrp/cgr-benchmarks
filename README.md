# CVXGenRust Benchmarks

This repository benchmarks [`cvxgenrust`](https://github.com/dxogrp/cvxgenrust)
against direct CVXPY solves and CVXPYgen-generated solvers.

The default comparison is Clarabel-aligned: `cvxgenrust` and direct CVXPY use
Clarabel, and CVXPYgen is asked to generate Clarabel-backed solvers where that
backend supports the problem.

## Setup

Install dependencies with uv:

```bash
uv sync
```

By default this project uses the sibling development checkout at
`/home/zhuh/src/cvxgenrust`. To benchmark a published or remote version instead,
edit the `cvxgenrust` dependency in `pyproject.toml`.

CVXPYgen's Clarabel backend also requires Eigen3 to be installed and visible to
CMake, for example through your system package manager or `CMAKE_PREFIX_PATH`.

## Commands

Generate solver projects:

```bash
uv run cgr-bench generate
```

Run the benchmark suite:

```bash
uv run cgr-bench run
```

The default tool list includes `cvxpygen_c`, a generated C driver around
CVXPYgen's C API. The `seconds` column is the main timing metric. For
`cgr_python` and `cvxpygen`, it measures the Python wrapper call including
parameter update overhead. For `cgr_rust` and `cvxpygen_c`, it measures the
direct native driver call including parameter updates plus the solver call.
The Markdown summary reports count, median, mean, min, and max; plots use median
solve time.

Write summary tables from existing raw results:

```bash
uv run cgr-bench report
```

## Reproducibility Checks

The default checks are fast experiment-validation tests. They verify that the
benchmark configuration loads, every problem family is DPP-valid and solvable,
sample generation matches the declared parameters, and result/report schemas stay
stable.

```bash
uv run pytest
```

The heavyweight toolchain checks exercise actual `cvxgenrust` and CVXPYgen code
generation for one QP and one SOCP. They are opt-in because they require the
full native build stack, including Cargo, CMake, and Eigen3 for CVXPYgen's
Clarabel backend.

```bash
CGR_BENCH_RUN_GENERATION_SMOKE=1 uv run pytest tests/test_generation_smoke.py
```

For a paper artifact review, run the fast checks first, install any missing
native dependencies, then run the heavyweight generation checks before producing
the final benchmark tables.

## Outputs

Generated solver projects and temporary build inputs are written under
`artifacts/`. CVXPYgen projects are generated in `cpg_code` folders to avoid a
Python import-name collision with the installed `cvxpygen` package. Benchmark
outputs are written under `results/`:

- `solve_times.csv`
- `solve_times.jsonl`
- `environment.json`
- `summary.md`

Timing excludes code generation and compilation. Python-level solve paths include
parameter update and wrapper overhead. Binary size is intentionally not reported:
the Rust and C/Python artifact boundaries are implementation-language dependent,
so the benchmark focuses on solve latency.

The default problem suite includes small, medium, and large variants for each
problem family. Raw timing rows include `problem_family`, `size_label`, and
`size` columns so scaling plots can compare solve time as problem dimensions
grow.
