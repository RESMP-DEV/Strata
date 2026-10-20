#!/usr/bin/env python3
"""Build and run the disaggregated KV coordinator's host-safe contract checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "kv-coordinator-check"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=False, text=True)


def main() -> int:
    cmake = shutil.which("cmake")
    if cmake is None:
        print("FAIL kv-coordinator: cmake is required")
        return 1
    generator = ["-G", "Ninja"] if shutil.which("ninja") is not None else []
    commands = (
        [cmake, "-S", str(ROOT), "-B", str(BUILD), *generator, "-DSTRATA_BUILD_TESTS=ON"],
        [cmake, "--build", str(BUILD), "--target", "kv_coordinator_test", "kv_tier_session_test"],
        [
            "ctest",
            "--test-dir",
            str(BUILD),
            "-R",
            "^(kv_coordinator_test|kv_tier_session_test)$",
            "--output-on-failure",
        ],
    )
    for command in commands:
        completed = run(command)
        print(
            f"{'PASS' if completed.returncode == 0 else 'FAIL'} kv-coordinator: {' '.join(command)}"
        )
        if completed.returncode != 0:
            return completed.returncode
    return 0


def quick_commands() -> tuple[str, ...]:
    return (
        "cmake --build build/kv-coordinator-check --target kv_coordinator_test"
        " kv_tier_session_test",
        "python3 scripts/check_kv_coordinator.py",
    )


if __name__ == "__main__":
    raise SystemExit(main())
