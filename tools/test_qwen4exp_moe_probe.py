#!/usr/bin/env python3
"""Host-safe tests for the Qwen4Exp MoE probe helper contract."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("qwen4exp_moe_probe", TOOLS / "qwen4exp_moe_probe.py")
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class Qwen4ExpMoeProbeTest(unittest.TestCase):
    def test_expected_expert_linear_count_matches_the_own_coder_model(self) -> None:
        self.assertEqual(PROBE.expected_expert_linear_count(48, 288), 41_472)
        self.assertEqual(PROBE.expected_expert_linear_count(2, 8), 48)

    def test_probe_format_is_stable(self) -> None:
        self.assertEqual(PROBE.PROBE_FORMAT, "strata-qwen4exp-moe-probe-v1")


if __name__ == "__main__":
    unittest.main()
