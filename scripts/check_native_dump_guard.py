#!/usr/bin/env python3
"""Compile and run the host-safe native dump guard contract test."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "native-dump-guard"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=False, text=True)


def main() -> int:
    compiler = shutil.which("c++")
    if compiler is None:
        print("FAIL native-dump-guard: c++ is required")
        return 1
    BUILD.mkdir(parents=True, exist_ok=True)
    binary = BUILD / "native_dump_guard_test"
    commands = (
        [
            compiler,
            "-std=c++17",
            "-Iinclude",
            "src/program/native_dump_guard_test.cpp",
            "-o",
            str(binary),
        ],
        [str(binary)],
    )
    for command in commands:
        completed = run(command)
        print(
            f"{'PASS' if completed.returncode == 0 else 'FAIL'} native-dump-guard: "
            f"{' '.join(command)}"
        )
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
