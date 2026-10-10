#!/usr/bin/env python3
"""Build and run the host-safe accelerator-plugin ABI checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "plugin-check"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
    return completed


def main() -> int:
    cmake = shutil.which("cmake")
    if cmake is None:
        print("FAIL plugin-abi: cmake is required")
        return 1
    generator = ["-G", "Ninja"] if shutil.which("ninja") is not None else []
    commands = (
        [cmake, "-S", str(ROOT), "-B", str(BUILD), *generator, "-DSTRATA_BUILD_TESTS=ON"],
        [cmake, "--build", str(BUILD), "--target", "plugin_loader_test"],
        ["ctest", "--test-dir", str(BUILD), "-R", "^plugin_loader_test$", "--output-on-failure"],
    )
    for command in commands:
        completed = run(command)
        print(f"{'PASS' if completed.returncode == 0 else 'FAIL'} plugin-abi: {' '.join(command)}")
        if completed.returncode != 0:
            return completed.returncode
    return 0


def quick_commands() -> tuple[str, ...]:
    return (
        "cmake --build build/plugin-check --target plugin_loader_test",
        "python3 scripts/check_plugins.py",
    )


if __name__ == "__main__":
    raise SystemExit(main())
