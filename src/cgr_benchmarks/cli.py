from __future__ import annotations

import argparse
from pathlib import Path

from .config import BenchmarkConfig
from .generation import generate_all
from .plots import write_plots
from .report import write_report
from .runner import run_benchmarks


def _config(path: str | Path | None) -> BenchmarkConfig:
    return BenchmarkConfig.load(Path(path or "benchmarks.toml"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cgr-bench")
    parser.add_argument("--config", default="benchmarks.toml", help="Path to benchmark TOML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate cvxgenrust and CVXPYgen projects")
    generate_parser.add_argument("--clean", action="store_true", help="Remove generated projects before generating")

    run_parser = subparsers.add_parser("run", help="Build artifacts and run timing benchmarks")
    run_parser.add_argument("--skip-generate", action="store_true", help="Reuse existing generated projects")

    subparsers.add_parser("report", help="Write Markdown summary from existing CSV results")
    subparsers.add_parser("plot", help="Write PNG plots from existing CSV results")

    args = parser.parse_args(argv)
    config = _config(args.config)

    if args.command == "generate":
        generate_all(config, clean=args.clean)
    elif args.command == "run":
        run_benchmarks(config, generate=not args.skip_generate)
    elif args.command == "report":
        write_report(config)
    elif args.command == "plot":
        write_plots(config)
