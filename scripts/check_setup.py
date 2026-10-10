#!/usr/bin/env python3
"""Run the deterministic setup regressions used by the common quality gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED_SETUP_TESTS = (
    "test_setup_amd.py",
    "test_setup_config.py",
    "test_setup_configs.py",
    "test_setup_deterministic_iq.py",
    "test_setup_hotfix_tag.py",
    "test_setup_rope.py",
    "test_setup_sycl.py",
)


def run_setup_modules() -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    for module in SELECTED_SETUP_TESTS:
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tools", "-p", module, "-q"]
        completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
        results.append((module, completed.returncode))
    return results


def main() -> int:
    results = run_setup_modules()
    for module, code in results:
        print(f"{'PASS' if code == 0 else 'FAIL'} {module}: exit {code}")
    return 0 if all(code == 0 for _, code in results) else 1


def quick_commands() -> tuple[str, ...]:
    """Commands contributors can copy for this focused setup domain."""
    return (
        *(f"python3 tools/{module}" for module in SELECTED_SETUP_TESTS),
        "python3 scripts/check_setup.py",
    )


if __name__ == "__main__":
    raise SystemExit(main())
