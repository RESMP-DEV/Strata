#!/usr/bin/env python3
"""Run the host-safe DeepSeek-V4.1 contract gates."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "deepseek-contracts"
MANIFEST = ROOT / "data" / "deepseek_v41_manifest.json"


def commands() -> tuple[list[str], ...]:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("cmake is required")
    ctest = shutil.which("ctest")
    if ctest is None:
        raise RuntimeError("ctest is required")
    generator = ["-G", "Ninja"] if shutil.which("ninja") is not None else []
    return (
        [sys.executable, str(ROOT / "scripts" / "check_deepseek.py"), "--manifest", str(MANIFEST)],
        [
            cmake,
            "--fresh",
            "-S",
            str(ROOT / "src" / "deepseek"),
            "-B",
            str(BUILD),
            *generator,
        ],
        [
            cmake,
            "--build",
            str(BUILD),
            "--target",
            "deepseek_geometry_test",
            "deepseek_candidate_select_test",
        ],
        [
            ctest,
            "--test-dir",
            str(BUILD),
            "-R",
            "^(deepseek_geometry_test|deepseek_candidate_select_test)$",
            "--output-on-failure",
        ],
    )


def main() -> int:
    try:
        planned = commands()
    except RuntimeError as error:
        print(f"FAIL deepseek-contracts: {error}")
        return 1
    for command in planned:
        completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
        label = " ".join(command)
        print(f"{'PASS' if completed.returncode == 0 else 'FAIL'} deepseek-contracts: {label}")
        if completed.returncode != 0:
            return completed.returncode
    return 0


def quick_commands() -> tuple[str, ...]:
    return ("python3 scripts/check_deepseek_contracts.py",)


if __name__ == "__main__":
    raise SystemExit(main())
