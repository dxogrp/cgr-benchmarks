from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _command_output(command: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def metadata() -> dict[str, object]:
    cvxgenrust_path = Path("../cvxgenrust").resolve()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            name: _package_version(name)
            for name in ["cvxgenrust", "cvxpy", "cvxpygen", "numpy", "pandas", "scipy"]
        },
        "rustc": _command_output(["rustc", "--version"]),
        "cargo": _command_output(["cargo", "--version"]),
        "cvxgenrust_git_commit": _command_output(["git", "rev-parse", "HEAD"], cwd=cvxgenrust_path)
        if cvxgenrust_path.exists()
        else "unknown",
    }


def write_metadata(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "environment.json").write_text(
        json.dumps(metadata(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

