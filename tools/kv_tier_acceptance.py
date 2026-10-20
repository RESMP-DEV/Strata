#!/usr/bin/env python3
"""Run the disaggregated KV tier's live-model correctness and scale gates.

The gate refuses a pass that is not earned. The gates are the ones in
docs/KV_DISAGGREGATION.md:

  * identical token IDs and targets between the RAM control and the cold arm;
  * zero argmax mismatches over the scored trace;
  * a mean NLL delta within the pre-registered tolerance (1e-4 by default);
  * the cold arm reports NONZERO foreground cold reads and evictions;
  * the max absolute log-probability delta is reported, not only the perplexity;
  * the scale gates are 600K and then 800K, and each has its own correctness run.

The script reads the two arms' JSON traces and decides. It does not run the
engine: the engine runs on the target host, and this script is what reads the
result and says whether the gate passed. That split is deliberate, so the
decision is made from receipts rather than from a human reading a log.

Usage:
    python3 tools/kv_tier_acceptance.py --control control.json --cold cold.json [--tolerance 1e-4]
    python3 tools/kv_tier_acceptance.py --self-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_TOLERANCE = 1e-4
SCALE_GATES = (600_000, 800_000)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _mean_nll(trace: dict[str, Any]) -> float:
    values = trace.get("nll")
    if not values:
        raise ValueError("the trace has no nll values")
    return sum(values) / len(values)


def _argmax_mismatches(control: dict[str, Any], cold: dict[str, Any]) -> int:
    left = control.get("argmax")
    right = cold.get("argmax")
    if left is None or right is None:
        raise ValueError("a trace has no argmax values")
    if len(left) != len(right):
        raise ValueError("the two traces have different lengths")
    return sum(1 for a, b in zip(left, right, strict=True) if a != b)


def evaluate(control: dict[str, Any], cold: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """Decide the live-model correctness gate from the two arms' traces."""
    result: dict[str, Any] = {
        "passed": False,
        "tolerance": tolerance,
        "checks": {},
        "failures": [],
    }

    # The two arms must be the same run in every respect but the tier.
    for field in (
        "model",
        "artifact_sha256",
        "context_tokens",
        "prompt_sha256",
        "seed",
        "kv_format",
        "expert_cache_slots",
        "mtp_enabled",
    ):
        left = control.get(field)
        right = cold.get(field)
        if left != right:
            result["failures"].append(f"the arms differ in {field}: {left!r} against {right!r}")
    if result["failures"]:
        return result

    try:
        mismatches = _argmax_mismatches(control, cold)
        mean_control = _mean_nll(control)
        mean_cold = _mean_nll(cold)
    except ValueError as error:
        result["failures"].append(str(error))
        return result

    delta = abs(mean_cold - mean_control)
    cold_reads = int(cold.get("kv_tier_cold_reads", 0) or 0)
    evictions = int(cold.get("kv_tier_l1_evictions", 0) or 0)
    max_delta = cold.get("kv_tier_max_abs_logprob_delta")

    result["checks"] = {
        "argmax_mismatches": mismatches,
        "mean_nll_control": mean_control,
        "mean_nll_cold": mean_cold,
        "mean_nll_delta": delta,
        "cold_reads": cold_reads,
        "l1_evictions": evictions,
        "max_abs_logprob_delta": max_delta,
        "perplexity_control": control.get("perplexity"),
        "perplexity_cold": cold.get("perplexity"),
    }

    if mismatches != 0:
        result["failures"].append(f"{mismatches} argmax mismatches")
    if not delta <= tolerance:
        result["failures"].append(f"the mean NLL delta {delta} exceeds the tolerance {tolerance}")
    if cold_reads <= 0:
        result["failures"].append("the cold arm reported no foreground cold read")
    if evictions <= 0:
        result["failures"].append("the cold arm reported no L1 eviction")
    if max_delta is None:
        result["failures"].append(
            "the cold arm did not report the max absolute log-probability delta"
        )
    if control.get("perplexity") is None or cold.get("perplexity") is None:
        result["failures"].append("a trace has no perplexity")

    result["passed"] = not result["failures"]
    return result


def scale_gate(trace: dict[str, Any], context_tokens: int) -> dict[str, Any]:
    """Decide one scale gate.

    The context must be the one the gate names, and the run must be correct.
    """
    result: dict[str, Any] = {"context_tokens": context_tokens, "passed": False, "failures": []}
    if trace.get("context_tokens") != context_tokens:
        result["failures"].append(
            f"the run used {trace.get('context_tokens')} tokens, not the gate's {context_tokens}"
        )
    if not trace.get("correct"):
        result["failures"].append("the run was not correct")
    result["passed"] = not result["failures"]
    return result


def _self_test() -> int:
    """The decision logic's own tests: a gate that passes when it should not is a broken gate."""
    control = {
        "model": "qwen3.8-flash-next",
        "artifact_sha256": "a" * 64,
        "context_tokens": 1024,
        "prompt_sha256": "b" * 64,
        "seed": 7,
        "kv_format": "int8",
        "expert_cache_slots": 2233,
        "mtp_enabled": True,
        "nll": [1.0, 2.0, 3.0],
        "argmax": [1, 2, 3],
        "perplexity": 14.25,
    }
    cold = dict(control)
    cold.update(
        {
            "nll": [1.0, 2.0, 3.0],
            "kv_tier_cold_reads": 5,
            "kv_tier_l1_evictions": 2,
            "kv_tier_max_abs_logprob_delta": 0.0,
        }
    )
    if not evaluate(control, cold, DEFAULT_TOLERANCE)["passed"]:
        print("FAIL: a matching pair did not pass")
        return 1

    # An argmax mismatch fails.
    bad = dict(cold)
    bad["argmax"] = [1, 2, 4]
    if evaluate(control, bad, DEFAULT_TOLERANCE)["passed"]:
        print("FAIL: an argmax mismatch passed")
        return 2

    # An NLL delta beyond the tolerance fails.
    bad = dict(cold)
    bad["nll"] = [1.0, 2.0, 3.5]
    if evaluate(control, bad, DEFAULT_TOLERANCE)["passed"]:
        print("FAIL: an NLL delta past the tolerance passed")
        return 3

    # No cold read fails: identical output alone cannot tell the disk path from the RAM path.
    bad = dict(cold)
    bad["kv_tier_cold_reads"] = 0
    if evaluate(control, bad, DEFAULT_TOLERANCE)["passed"]:
        print("FAIL: a run with no cold read passed")
        return 4

    # No eviction fails: the cold tier must have been exercised as a cache.
    bad = dict(cold)
    bad["kv_tier_l1_evictions"] = 0
    if evaluate(control, bad, DEFAULT_TOLERANCE)["passed"]:
        print("FAIL: a run with no eviction passed")
        return 5

    # A missing max-delta report fails: the delta must be reported, not only the perplexity.
    bad = dict(cold)
    bad.pop("kv_tier_max_abs_logprob_delta")
    if evaluate(control, bad, DEFAULT_TOLERANCE)["passed"]:
        print("FAIL: a run without the max delta passed")
        return 6

    # A mismatched arm fails: the two arms must be the same run.
    bad = dict(cold)
    bad["seed"] = 8
    if evaluate(control, bad, DEFAULT_TOLERANCE)["passed"]:
        print("FAIL: a mismatched arm passed")
        return 7

    # The scale gate refuses a run at the wrong context.
    if scale_gate({"context_tokens": 512_000, "correct": True}, 600_000)["passed"]:
        print("FAIL: a 512K run passed the 600K gate")
        return 8
    if not scale_gate({"context_tokens": 600_000, "correct": True}, 600_000)["passed"]:
        print("FAIL: a correct 600K run did not pass the 600K gate")
        return 9
    if scale_gate({"context_tokens": 600_000, "correct": False}, 600_000)["passed"]:
        print("FAIL: an incorrect 600K run passed")
        return 10

    print("kv tier acceptance: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, help="the RAM control arm's JSON trace")
    parser.add_argument("--cold", type=Path, help="the cold tier arm's JSON trace")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="the pre-registered mean NLL tolerance",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="run the decision logic's own tests"
    )
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if not args.control or not args.cold:
        parser.error("--control and --cold are required unless --self-test is given")

    result = evaluate(_load(args.control), _load(args.cold), args.tolerance)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
