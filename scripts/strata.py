#!/usr/bin/env python3
"""Dispatch Strata's local checks and write receipt JSON."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import check_structure

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
REPORT = ROOT / "build" / "reports" / "quality" / "receipt.json"
LOG_ROOT = ROOT / "build" / "reports" / "quality" / "logs"
SCOPE = (
    "Validates repository governance, ten selected setup regressions, the host-safe "
    "accelerator-plugin ABI test, DeepSeek manifest and host C++ contract tests, "
    "script tests, maintained-script Ruff lint/format, shell warning lint, and Git "
    "whitespace on this host. It does not compile CUDA, HIP, or SYCL, run the full "
    "setup matrix, start the "
    "server, download a model, or measure performance."
)


def _git_output(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_state() -> dict[str, Any]:
    return {
        "source_revision": _git_output(["rev-parse", "HEAD"]),
        "source_dirty": bool(_git_output(["status", "--porcelain"])),
    }


def write_receipt(
    *,
    passed: bool,
    checks: list[dict[str, Any]],
    state: dict[str, Any],
    startup_failure: str | None = None,
) -> None:
    receipt: dict[str, Any] = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "system": platform.platform(),
            "python": sys.version,
        },
        **state,
        "scope": SCOPE,
        "checks": checks,
        "aggregate_passed": passed,
    }
    if startup_failure is not None:
        receipt["startup_failure"] = startup_failure
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_name(f".{REPORT.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REPORT)


def run_logged(name: str, command: list[str]) -> dict[str, Any]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return {
        "check": name,
        "command": command,
        "exit": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "log": str(log_path.relative_to(ROOT)),
    }


def _structural_checks(checks: list[dict[str, Any]]) -> bool:
    started = time.monotonic()
    results = check_structure.check_all()
    passed = True
    for result in results:
        print(f"{'PASS' if result.passed else 'FAIL'} {result.name}: {result.detail}")
        passed = passed and result.passed
        checks.append(
            {
                "check": f"structure/{result.name}",
                "command": [sys.executable, "scripts/check_structure.py"],
                "exit": 0 if result.passed else 1,
                "elapsed_seconds": None,
                "log": None,
                "detail": result.detail,
            }
        )
    elapsed = round(time.monotonic() - started, 3)
    for entry in checks:
        if entry["check"].startswith("structure/"):
            entry["elapsed_seconds"] = elapsed
    return passed


def quality() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    try:
        REPORT.unlink(missing_ok=True)
        state = git_state()
        structural_passed = _structural_checks(checks)
        checks.extend(
            [
                run_logged(
                    "script-tests",
                    [sys.executable, "-m", "unittest", "discover", "-s", "scripts/tests", "-v"],
                ),
                run_logged(
                    "selected-setup-tests",
                    [sys.executable, "scripts/check_setup.py"],
                ),
                run_logged(
                    "plugin-abi",
                    [sys.executable, "scripts/check_plugins.py"],
                ),
                run_logged(
                    "deepseek-contracts",
                    [sys.executable, "scripts/check_deepseek_contracts.py"],
                ),
                run_logged("ruff-check", ["ruff", "check", "scripts"]),
                run_logged("ruff-format", ["ruff", "format", "--check", "scripts"]),
                run_logged(
                    "shellcheck-warning",
                    [
                        "shellcheck",
                        "-S",
                        "warning",
                        "setup.sh",
                        "update.sh",
                        "docker-entrypoint.sh",
                    ],
                ),
                run_logged("git-whitespace", ["git", "diff", "--check"]),
            ]
        )
        passed = structural_passed and all(entry["exit"] == 0 for entry in checks)
        write_receipt(passed=passed, checks=checks, state=state)
        return 0 if passed else 1
    except BaseException as error:
        write_receipt(
            passed=False,
            checks=checks,
            state=state,
            startup_failure=f"{type(error).__name__}: {error}",
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("quality", help="run the host-safe repository gate")
    arguments = parser.parse_args(argv)
    if arguments.command in {None, "quality"}:
        return quality()
    parser.error(f"unknown command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
