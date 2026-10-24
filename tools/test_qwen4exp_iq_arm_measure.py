#!/usr/bin/env python3
"""Host-safe contract tests for the servable IQ arm measurement."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "qwen4exp_iq_arm_measure", TOOLS / "qwen4exp_iq_arm_measure.py"
)
assert SPEC is not None and SPEC.loader is not None
ARM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARM)
import iq3_qdq  # noqa: E402


class IqArmMeasureTest(unittest.TestCase):
    def test_servable_arms_use_engine_supported_types(self) -> None:
        self.assertEqual(set(ARM.ARMS), {"3", "3x", "4"})
        self.assertEqual(ARM.ARMS["3"]["gate_up"], iq3_qdq.IQ3_S)
        self.assertEqual(ARM.ARMS["3x"]["gate_up"], iq3_qdq.IQ3_XXS)
        self.assertEqual(ARM.ARMS["4"]["gate_up"], iq3_qdq.IQ4_XS)
        self.assertEqual({arm["down"] for arm in ARM.ARMS.values()}, {iq3_qdq.IQ4_NL})

    def test_receipt_format_is_stable(self) -> None:
        self.assertEqual(ARM.ARM_FORMAT, "strata-qwen4exp-iq-arm-measure-v1")


if __name__ == "__main__":
    unittest.main()
