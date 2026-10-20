#!/usr/bin/env python3
"""Tests for the bounded B550 acceptance harness."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import b550_acceptance as acceptance


def completed(command: list[str], stdout: str = "", stderr: str = "", code: int = 0):
    return subprocess.CompletedProcess(command, code, stdout, stderr)


class ParsingTest(unittest.TestCase):
    def test_gpu_identity_is_parsed_strictly(self) -> None:
        rows = acceptance.parse_gpu_list(
            "NVIDIA GeForce RTX 3090 Ti, GPU-1, 0000:01:00.0, 8.6\n"
            "NVIDIA GeForce RTX 3090 Ti, GPU-2, 0000:02:00.0, 8.6\n"
        )
        self.assertEqual([row["compute_cap"] for row in rows], ["8.6", "8.6"])
        with self.assertRaises(ValueError):
            acceptance.parse_gpu_list("wrong, row\n")

    def test_optane_requires_exact_model_and_serial(self) -> None:
        good = json.dumps(
            {
                "blockdevices": [
                    {
                        "type": "disk",
                        "path": "/dev/nvme3n1",
                        "model": acceptance.EXPECTED_OPTANE_MODEL,
                        "serial": acceptance.EXPECTED_OPTANE_SERIAL,
                        "children": [{"type": "part", "path": "/dev/nvme3n1p2"}],
                    }
                ]
            }
        )
        node = acceptance.find_optane(good)
        self.assertEqual(node["path"], "/dev/nvme3n1")
        bad = json.loads(good)
        bad["blockdevices"][0]["serial"] = "wrong"
        with self.assertRaises(ValueError):
            acceptance.find_optane(json.dumps(bad))

    def test_findmnt_reports_read_only(self) -> None:
        mount = acceptance.parse_findmnt(
            json.dumps(
                {
                    "filesystems": [
                        {
                            "source": "/dev/nvme3n1p2",
                            "target": "/mnt/optane",
                            "fstype": "exfat",
                            "options": "rw,noatime",
                        }
                    ]
                }
            )
        )
        self.assertFalse(mount["read_only"])

    def test_engine_trace_requires_output_and_contiguous_nll(self) -> None:
        text = "prompt  : 1 2 3\noutput  : 9 8\n0\t2\t-1.0\t9\t-1.0\t1\n1\t3\t-2.0\t8\t-2.0\t1\n"
        trace = acceptance.parse_engine_output(text, "test")
        self.assertEqual(trace.output_tokens, [9, 8])
        self.assertEqual(trace.argmax_values, [9, 8])
        with self.assertRaises(ValueError):
            acceptance.parse_engine_output("prompt  : 1 2 3\noutput  : 2\n", "missing")

    def test_tier_report_requires_all_guard_counters(self) -> None:
        line = (
            "strata kv tier (control): l1_evictions=1 l2_reads=2 l2_read_failures=0 "
            "checksum_errors=0 readiness_waits=1 engine_host_restores=2 selection=1 readiness=1 "
            "resolve=1 windows=1 violations=0\n"
        )
        values = acceptance.parse_tier_report(line, "test")
        self.assertEqual(values["l2_reads"], 2)
        with self.assertRaises(ValueError):
            acceptance.parse_tier_report("strata kv tier (control): l2_reads=2\n", "test")


class ComparisonTest(unittest.TestCase):
    def trace(self, output: list[int], nll: list[float], argmax: list[int]) -> acceptance.RunTrace:
        return acceptance.RunTrace("arm", [1, 2, 3], output, nll, argmax, {"l2_reads": 1})

    def test_exact_identity_refuses_any_nll_delta(self) -> None:
        result = acceptance.compare_traces(
            self.trace([7], [-1.0], [7]),
            self.trace([7], [-1.000000001], [7]),
            tolerance=1e-4,
            exact=True,
        )
        self.assertFalse(result["passed"])

    def test_matched_control_uses_mean_not_max_tolerance(self) -> None:
        result = acceptance.compare_traces(
            self.trace([7, 8], [-1.0, -1.0], [7, 8]),
            self.trace([7, 8], [-1.0, -1.000009], [7, 8]),
            tolerance=1e-5,
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["max_abs_logprob_delta"], 0.000009)


class RemoteSafetyTest(unittest.TestCase):
    def test_engine_command_is_shell_quoted(self) -> None:
        seen: list[list[str]] = []

        def runner(command: list[str], **_kwargs: object):
            seen.append(command)
            return completed(command, stdout="")

        harness = acceptance.Harness(runner=runner)
        acceptance._remote_quoted(harness, "test", ["/path/engine", "--pack", "/a b/pack"])
        self.assertIn("/path/engine --pack '/a b/pack'", seen[0][-1])

    def test_engine_arguments_may_not_override_controlled_flags(self) -> None:
        with self.assertRaises(ValueError):
            acceptance._validate_engine_args(["--kv-resident", "32768", "--kv-tier", "/bad"], 32768)
        with self.assertRaises(ValueError):
            acceptance._validate_engine_args(["--kv-resident", "20479"], 20479)

    def test_failed_step_is_recorded_and_aggregated(self) -> None:
        def runner(command: list[str], **_kwargs: object):
            return completed(command, stdout="", stderr="connection refused", code=255)

        harness = acceptance.Harness(runner=runner)
        result = harness.local("probe", ["true"])
        self.assertFalse(result.passed)
        self.assertEqual(harness.failures, ["LOCAL probe exited 255: connection refused"])


if __name__ == "__main__":
    unittest.main()
