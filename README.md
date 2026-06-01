# CVXGenRust Benchmarks

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

The benchmark workflow is exposed through `make` targets:

```bash
make sync
make generate
make generate-clean
make run
make run-skip-generate
make report
make plot
make experiment
make clean
```

Use a different config with:

```bash
make CONFIG=path/to/config.toml run
```

Run `make help` for the full target list and available options.

## Outputs

Generated artifacts:

- `artifacts/generated/`: cvxgenrust and CVXPYgen generated projects
- `artifacts/rust_drivers/`: generated Rust benchmark drivers
- `artifacts/cvxpygen_c_drivers/`: generated C benchmark drivers
- `artifacts/rust_samples/`, `artifacts/cvxpygen_samples/`, `artifacts/cvxpygen_c_samples/`: serialized benchmark inputs

Benchmark results:

- `results/solve_times.csv`: raw timing and comparison rows
- `results/solve_times.jsonl`: JSONL copy of raw rows
- `results/environment.json`: tool and environment metadata
- `results/summary.md`: Markdown summary report
- `results/plots/`: generated PNG plots

The `seconds` column is the main timing metric. For `cgr_python` and
`cvxpygen`, it measures the Python wrapper call including parameter update
overhead. For `cgr_rust` and `cvxpygen_c`, it measures the direct native driver
call including parameter updates plus the solver call.

Timing excludes code generation and compilation. Raw timing rows include
`problem_family`, `size_label`, and `size` columns so scaling plots can compare
solve time as problem dimensions grow.
