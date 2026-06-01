BLUE := \033[36m
BOLD := \033[1m
RESET := \033[0m

CONFIG ?= benchmarks.toml
RESULT_REL_TOL ?= 1e-4
ALLOW_RESULT_ERRORS ?= 0
CGR_BENCH := uv run cgr-bench --config $(CONFIG)

.DEFAULT_GOAL := help

.PHONY: sync
sync: ## install the benchmark environment
	@printf "$(BLUE)Syncing dependencies...$(RESET)\n"
	@uv sync --frozen

.PHONY: generate
generate: sync ## generate cvxgenrust and CVXPYgen projects
	@printf "$(BLUE)Generating solver projects with $(CONFIG)...$(RESET)\n"
	@$(CGR_BENCH) generate

.PHONY: generate-clean
generate-clean: sync ## remove generated projects, then generate them again
	@printf "$(BLUE)Regenerating solver projects with $(CONFIG)...$(RESET)\n"
	@$(CGR_BENCH) generate --clean

.PHONY: run
run: sync ## generate, build, and run the configured benchmark experiment
	@printf "$(BLUE)Running benchmark experiment with $(CONFIG)...$(RESET)\n"
	@$(CGR_BENCH) run

.PHONY: run-skip-generate
run-skip-generate: sync ## build and run benchmarks from existing generated projects
	@printf "$(BLUE)Running benchmark experiment from existing generated projects...$(RESET)\n"
	@$(CGR_BENCH) run --skip-generate

.PHONY: report
report: sync ## write the Markdown summary and plots from existing results
	@printf "$(BLUE)Writing benchmark report...$(RESET)\n"
	@$(CGR_BENCH) report

.PHONY: plot
plot: sync ## write PNG plots from existing results
	@printf "$(BLUE)Writing benchmark plots...$(RESET)\n"
	@$(CGR_BENCH) plot

.PHONY: check-results
check-results: sync ## validate result files, row counts, errors, and objective tolerances
	@printf "$(BLUE)Checking benchmark results...$(RESET)\n"
	@uv run python -c "from pathlib import Path; import pandas as pd; from cgr_benchmarks.config import BenchmarkConfig; from cgr_benchmarks.status import is_success_status; config = BenchmarkConfig.load(Path('$(CONFIG)')); path = config.results_dir / 'solve_times.csv'; assert path.exists(), f'missing result file: {path}'; frame = pd.read_csv(path); expected_cols = ['problem', 'problem_family', 'size_label', 'size', 'tool', 'sample', 'seconds', 'status', 'objective', 'iterations', 'rel_objective_error', 'rel_solution_error', 'error']; assert list(frame.columns) == expected_cols, f'unexpected columns: {list(frame.columns)}'; expected_rows = config.samples * len(config.enabled_problems) * len(config.enabled_tools); assert len(frame) == expected_rows, f'expected {expected_rows} rows, found {len(frame)}'; failed = frame[frame['error'].notna() | ~frame['status'].map(is_success_status)]; allow_errors = '$(ALLOW_RESULT_ERRORS)' == '1'; assert allow_errors or failed.empty, failed[['problem', 'tool', 'sample', 'error']].to_string(index=False); rel = frame.loc[frame['tool'] != 'cvxpy_direct', 'rel_objective_error'].dropna(); tol = float('$(RESULT_REL_TOL)'); assert rel.empty or float(rel.max()) <= tol, f'max rel_objective_error {float(rel.max())} exceeds {tol}'; summary = config.results_dir / 'summary.md'; assert summary.exists(), f'missing summary file: {summary}'; suffix = f' ({len(failed)} rows with recorded errors allowed)' if allow_errors and not failed.empty else ''; print(f'checked {len(frame)} rows in {path}{suffix}')"

.PHONY: experiment
experiment: run check-results ## run the full experiment and validate its outputs

.PHONY: clean
clean: ## remove local benchmark artifacts
	@printf "$(BLUE)Cleaning local artifacts...$(RESET)\n"
	@rm -rf artifacts results

.PHONY: help
help: ## display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET) [CONFIG=benchmarks.toml] [RESULT_REL_TOL=1e-4] [ALLOW_RESULT_ERRORS=0]\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-18s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
