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
    "accelerator-plugin ABI test, the disaggregated KV coordinator's state-machine and "
    "store-refusal contract tests, the KV tier acceptance gate's own decision tests, "
    "DeepSeek manifest and host C++ contract tests, "
    "custom Coder token and variant-planning contracts, "
    "the DeepSeek-V4.1-Flash RCO budget planner's own unit tests, "
    "the optimizer-stage readiness gate's manifest, SHA-256, and host-command checks "
    "on synthetic fixtures, "
    "script tests, maintained-script Ruff lint/format, shell warning lint, and Git "
    "whitespace on this host. It does not compile CUDA, HIP, or SYCL, run the full "
    "setup matrix, start the "
    "server, download a model, move any KV byte to a cold tier, or measure performance. "
    "The RCO planner tests exercise index classification and integer budget arithmetic "
    "on synthetic fixtures; they do not load, convert, or run DeepSeek-V4.1-Flash. "
    "The optimizer gate does not invoke an optimizer or GPU and does not establish "
    "optimizer quality."
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
        docs_checks = run_logged(
            "docs-index",
            [sys.executable, "scripts/check_docs.py"],
        )
        checks.append(docs_checks)
        structural_passed = structural_passed and docs_checks["exit"] == 0
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
                run_logged(
                    "kv-coordinator",
                    [sys.executable, "scripts/check_kv_coordinator.py"],
                ),
                run_logged(
                    "kv-tier-acceptance",
                    [sys.executable, "tools/kv_tier_acceptance.py", "--self-test"],
                ),
                run_logged(
                    "compression-tools",
                    [
                        sys.executable,
                        "-m",
                        "unittest",
                        "tools.test_reap_mask_from_traces",
                        "tools.test_build_rco_tokens",
                        "tools.test_custom_coder_plan",
                        "tools.test_materialize_pruned_qwen4exp",
                        "tools.test_deepseek_rco_plan",
                        "tools.test_deepseek_rco_materialization_plan",
                        "tools.test_optimizer_gate",
                        "tools.test_quant_candidate_plan",
                        "tools.test_rco_adapter_runner",
                        "tools.test_calibration_npy",
                        "tools.test_qwen4exp_moe_repack",
                        "tools.test_qwen4exp_moe_probe",
                        "tools.test_w4a16_cost_model",
                        "tools.test_qwen4exp_partial_load",
                        "tools.test_qwen4exp_quant_aware_rco",
                        "tools.test_qwen4exp_ple_stream",
                        "tools.test_qwen4exp_chunked_rco",
                        "tools.test_qwen4exp_qdq_method_compare",
                        "tools.test_iq3_qdq",
                        "tools.test_qwen4exp_iq_arm_measure",
                    ],
                ),
                run_logged(
                    "prefiller-planner",
                    [sys.executable, "-m", "unittest", "tools.test_prefiller_plan"],
                ),
                run_logged(
                    "ruff-check",
                    [
                        "ruff",
                        "check",
                        "scripts",
                        "tools/kv_tier_acceptance.py",
                        "tools/prefiller_plan.py",
                        "tools/test_prefiller_plan.py",
                        "tools/build_rco_tokens.py",
                        "tools/test_build_rco_tokens.py",
                        "tools/custom_coder_plan.py",
                        "tools/test_custom_coder_plan.py",
                        "tools/exact_copy_recipe.py",
                        "tools/test_exact_copy_recipe.py",
                        "tools/materialize_pruned_qwen4exp.py",
                        "tools/test_materialize_pruned_qwen4exp.py",
                        "tools/deepseek_rco_plan.py",
                        "tools/test_deepseek_rco_plan.py",
                        "tools/deepseek_rco_materialization_plan.py",
                        "tools/test_deepseek_rco_materialization_plan.py",
                        "tools/verify_deepseek_v41_headers.py",
                        "tools/optimizer_gate.py",
                        "tools/test_optimizer_gate.py",
                        "tools/quant_candidate_plan.py",
                        "tools/test_quant_candidate_plan.py",
                        "tools/rco_adapter_runner.py",
                        "tools/test_rco_adapter_runner.py",
                        "tools/calibration_npy.py",
                        "tools/test_calibration_npy.py",
                        "tools/qwen4exp_moe_probe.py",
                        "tools/test_qwen4exp_moe_probe.py",
                        "tools/qwen4exp_moe_repack.py",
                        "tools/test_qwen4exp_moe_repack.py",
                        "tools/verify_fused_moe_artifact.py",
                        "tools/w4a16_cost_model.py",
                        "tools/test_w4a16_cost_model.py",
                        "tools/qwen4exp_partial_load.py",
                        "tools/test_qwen4exp_partial_load.py",
                        "tools/qwen4exp_quant_aware_rco.py",
                        "tools/test_qwen4exp_quant_aware_rco.py",
                        "tools/qwen4exp_ple_stream.py",
                        "tools/test_qwen4exp_ple_stream.py",
                        "tools/qwen4exp_chunked_rco.py",
                        "tools/test_qwen4exp_chunked_rco.py",
                        "tools/qwen4exp_qdq_method_compare.py",
                        "tools/test_qwen4exp_qdq_method_compare.py",
                        "tools/iq3_qdq.py",
                        "tools/test_iq3_qdq.py",
                        "tools/qwen4exp_iq_arm_measure.py",
                        "tools/test_qwen4exp_iq_arm_measure.py",
                        "tools/bench_iq3.py",
                        "tools/bench_iq3_scale.py",
                        "tools/moesq_pack.py",
                        "tools/test_moesq_pack.py",
                    ],
                ),
                run_logged(
                    "ruff-format",
                    [
                        "ruff",
                        "format",
                        "--check",
                        "scripts",
                        "tools/kv_tier_acceptance.py",
                        "tools/prefiller_plan.py",
                        "tools/test_prefiller_plan.py",
                        "tools/build_rco_tokens.py",
                        "tools/test_build_rco_tokens.py",
                        "tools/custom_coder_plan.py",
                        "tools/test_custom_coder_plan.py",
                        "tools/exact_copy_recipe.py",
                        "tools/test_exact_copy_recipe.py",
                        "tools/materialize_pruned_qwen4exp.py",
                        "tools/test_materialize_pruned_qwen4exp.py",
                        "tools/deepseek_rco_plan.py",
                        "tools/test_deepseek_rco_plan.py",
                        "tools/deepseek_rco_materialization_plan.py",
                        "tools/test_deepseek_rco_materialization_plan.py",
                        "tools/verify_deepseek_v41_headers.py",
                        "tools/optimizer_gate.py",
                        "tools/test_optimizer_gate.py",
                        "tools/quant_candidate_plan.py",
                        "tools/test_quant_candidate_plan.py",
                        "tools/rco_adapter_runner.py",
                        "tools/test_rco_adapter_runner.py",
                        "tools/calibration_npy.py",
                        "tools/test_calibration_npy.py",
                        "tools/qwen4exp_moe_probe.py",
                        "tools/test_qwen4exp_moe_probe.py",
                        "tools/qwen4exp_moe_repack.py",
                        "tools/test_qwen4exp_moe_repack.py",
                        "tools/verify_fused_moe_artifact.py",
                        "tools/w4a16_cost_model.py",
                        "tools/test_w4a16_cost_model.py",
                        "tools/qwen4exp_partial_load.py",
                        "tools/test_qwen4exp_partial_load.py",
                        "tools/qwen4exp_quant_aware_rco.py",
                        "tools/test_qwen4exp_quant_aware_rco.py",
                        "tools/qwen4exp_ple_stream.py",
                        "tools/test_qwen4exp_ple_stream.py",
                        "tools/qwen4exp_chunked_rco.py",
                        "tools/test_qwen4exp_chunked_rco.py",
                        "tools/qwen4exp_qdq_method_compare.py",
                        "tools/test_qwen4exp_qdq_method_compare.py",
                        "tools/iq3_qdq.py",
                        "tools/test_iq3_qdq.py",
                        "tools/qwen4exp_iq_arm_measure.py",
                        "tools/test_qwen4exp_iq_arm_measure.py",
                        "tools/bench_iq3.py",
                        "tools/bench_iq3_scale.py",
                        "tools/moesq_pack.py",
                        "tools/test_moesq_pack.py",
                    ],
                ),
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
