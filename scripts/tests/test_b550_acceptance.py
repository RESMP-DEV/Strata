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

    def test_storage_requires_exact_model_and_serial(self) -> None:
        good = json.dumps(
            {
                "blockdevices": [
                    {
                        "type": "disk",
                        "path": "/dev/nvme3n1",
                        "model": acceptance.EXPECTED_STORAGE_MODEL,
                        "serial": acceptance.EXPECTED_STORAGE_SERIAL,
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

    def test_optane_mount_must_be_the_requested_endpoint(self) -> None:
        optane = {"path": "/dev/nvme0n1"}
        root = {
            "source": "/dev/nvme1n1p2",
            "target": "/",
            "fstype": "ext4",
            "options": ["rw"],
            "read_only": False,
        }
        error = acceptance.optane_mount_error(optane, root, "/mnt/optane")
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("is not a mount point", error)
        mounted = dict(root, source="/dev/nvme0n1p2", target="/mnt/optane")
        self.assertIsNone(acceptance.optane_mount_error(optane, mounted, "/mnt/optane"))

    def test_serve_trace_requires_tokens_and_done(self) -> None:
        text = "INFO context=3\nT 9\nT 8\nDONE 2 3 1.0 2.0 length 0 0 0\n"
        tokens, done = acceptance.parse_serve_output(text, "test")
        self.assertEqual(tokens, [9, 8])
        self.assertTrue(done.startswith("DONE "))
        with self.assertRaises(ValueError):
            acceptance.parse_serve_output("T 9\n", "missing-done")
        with self.assertRaises(ValueError):
            acceptance.parse_serve_output("T 9\nDONE 1 1 1 1 cancel\n", "cancelled")

    def test_done_metrics_are_reported(self) -> None:
        metrics = acceptance.parse_done(
            "DONE 2 3 1000.0 2000.0 length 0 0 0 1 2 chunk=128 experts_streamed=3"
        )
        self.assertEqual(metrics["generated"], 2)
        self.assertEqual(metrics["decode_ms"], 2000.0)
        self.assertEqual(metrics["chunk"], 128)
        self.assertEqual(metrics["experts_streamed"], 3)

    def test_logpos_requires_contiguous_full_coverage(self) -> None:
        text = "0\t2\t-1.0\t9\t-1.0\t1\n1\t3\t-2.0\t8\t-2.0\t1\n"
        positions, nll, argmax = acceptance.parse_logpos(text, "test")
        self.assertEqual(positions, [0, 1])
        self.assertEqual(nll, [-1.0, -2.0])
        self.assertEqual(argmax, [9, 8])
        with self.assertRaises(ValueError):
            acceptance.parse_logpos("", "missing")
        with self.assertRaises(ValueError):
            acceptance.parse_logpos("0\t2\t-1.0\t9\t-1.0\t1\n5\t3\t-2.0\t8\t-2.0\t1\n", "gap")
        with self.assertRaises(ValueError):
            acceptance.parse_logpos("0\t2\tnan\t9\t-1.0\t1\n", "not-finite")

    def test_tier_report_requires_all_guard_counters(self) -> None:
        line = (
            "strata kv tier (control): l1_evictions=1 l2_reads=2 l2_read_failures=0 "
            "checksum_errors=0 readiness_waits=1 engine_host_restores=2 selection=1 "
            "readiness=1 resolve=1 windows=1 violations=0\n"
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

        def runner(command: list[str], **kwargs: object):
            seen.append((command, kwargs.get("input")))
            return completed(command, stdout="")

        harness = acceptance.Harness(runner=runner)
        acceptance._remote_quoted(
            harness, "test", ["/path/engine", "--pack", "/a b/pack"], input="GEN"
        )
        self.assertIn("/path/engine --pack '/a b/pack'", seen[0][0][-1])
        self.assertEqual(seen[0][1], "GEN")

    def test_remote_command_preserves_argument_boundaries(self) -> None:
        seen: list[list[str]] = []

        def runner(command: list[str], **_kwargs: object):
            seen.append(command)
            return completed(command, stdout="")

        harness = acceptance.Harness(runner=runner)
        harness.remote("configure", ["cmake", "-DCMAKE_EXE_LINKER_FLAGS=-Wl,a -Wl,b"])
        self.assertEqual(seen[0][-1], "cmake '-DCMAKE_EXE_LINKER_FLAGS=-Wl,a -Wl,b'")

    def test_engine_arguments_may_not_override_controlled_flags(self) -> None:
        with self.assertRaises(ValueError):
            acceptance._validate_engine_args(["--kv-resident", "32768", "--kv-tier", "/bad"], 32768)
        with self.assertRaises(ValueError):
            acceptance._validate_engine_args(["--kv-resident", "20479"], 20479)
        with self.assertRaises(ValueError):
            acceptance._validate_engine_args(["--kv-resident", "32768"], 32768)
        with self.assertRaises(ValueError):
            acceptance._validate_engine_args(
                ["--kv-resident", "32768", "--serve", "--spec", "2"], 32768
            )
        with self.assertRaises(ValueError):
            acceptance._validate_engine_args(
                ["--kv-resident", "32768", "--short-read", "1024", "--spec", "2"], 32768
            )
        acceptance._validate_engine_args(
            ["--pack", "/p", "--spec", "2", "--prefill", "auto", "--kv-resident", "32768"],
            32768,
        )

    def test_serve_command_takes_the_verify_window_path(self) -> None:
        command = acceptance._serve_command(
            engine="/engine",
            engine_args=["--pack", "/p", "--spec", "2", "--prefill", "auto"],
            max_context=8192,
            logpos_path="/tmp/logpos",
            cold_tier=None,
            l1_groups=0,
            sm86_perf=None,
        )
        self.assertIn("--short-read", command)
        self.assertEqual(
            command[command.index("--short-read") + 1], str(acceptance.REQUIRED_SHORT_READ)
        )

    def test_failed_step_is_recorded_and_aggregated(self) -> None:
        def runner(command: list[str], **_kwargs: object):
            return completed(command, stdout="", stderr="connection refused", code=255)

        harness = acceptance.Harness(runner=runner)
        result = harness.local("probe", ["true"])
        self.assertFalse(result.passed)
        self.assertEqual(harness.failures, ["LOCAL probe exited 255: connection refused"])

    def test_unanswered_optional_lane_does_not_fail_reachability(self) -> None:
        def runner(command: list[str], **_kwargs: object):
            if "10.243.130.224" in command:
                return completed(command, stdout="", stderr="host down", code=1)
            return completed(command, stdout="b550\n")

        harness = acceptance.Harness(runner=runner)
        lanes = acceptance.probe_reachability(harness, "10.0.0.2", "10.243.130.224")
        self.assertTrue(harness.facts["reachability"]["reachable"])
        self.assertFalse(lanes["zero_tier"]["reachable"])
        self.assertTrue(lanes["lan"]["reachable"])
        self.assertEqual(harness.failures, [])
        self.assertTrue(all(not step.aggregate_failure for step in harness.steps))
        lanes = acceptance.probe_reachability(harness, "10.0.0.2", "")
        self.assertNotIn("zero_tier", lanes)

    def test_identity_gate_requires_exact_nll_and_tokens(self) -> None:
        state = {"arm": 0, "path": 0}

        def runner(command: list[str], **kwargs: object):
            joined = " ".join(command)
            if "nvidia-smi --query-compute-apps" in joined:
                return completed(command)
            if "mkdir -p" in joined:
                return completed(command)
            if "mktemp" in joined and "gen." not in joined:
                state["path"] += 1
                return completed(command, stdout=f"/tmp/logpos-{state['path']}")
            if "mktemp" in joined:
                return completed(command, stdout="/tmp/request")
            if "cat -- /tokens" in joined:
                return completed(command, stdout="1 2 3\n")
            if "cat > /tmp/request" in joined:
                return completed(command)
            if "sha256sum" in joined:
                return completed(command, stdout="a  /engine\nb  /model\nc  /tokens\n")
            if joined.endswith("cat -- /tmp/logpos-1"):
                return completed(command, stdout="0\t2\t-1.0\t7\t-1.0\t1\n")
            if joined.endswith("cat -- /tmp/logpos-2"):
                return completed(command, stdout="0\t2\t-1.01\t7\t-1.01\t1\n")
            if "cat -- /tmp/request" in joined:
                state["arm"] += 1
                output = "9 8" if state["arm"] == 1 else "9 7"
                stdout = f"INFO context=3\nT 9\nT {output.split()[-1]}\nDONE 2 3 1 1 length 0 0 0\n"
                return completed(command, stdout=stdout)
            return completed(command, code=70, stderr=f"unexpected command: {joined}")

        harness = acceptance.Harness(runner=runner)
        args = ["--pack", "/model", "--spec", "2", "--prefill", "auto", "--kv-resident", "32768"]
        result = acceptance.run_identity_gate(
            harness,
            engine="/engine",
            engine_args=args,
            tokens_file="/tokens",
            max_new=2,
            max_context=32768,
            work_dir="/work",
        )
        self.assertFalse(result["passed"])
        self.assertIn("identity gate: output token IDs differ", result["failures"])
        self.assertIn(
            "identity gate: maximum prompt-NLL delta is 0.01, not zero", result["failures"]
        )


if __name__ == "__main__":
    unittest.main()
