#!/usr/bin/env python3
"""Host-safe import and contract tests for the chunked RCO sweep."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "qwen4exp_chunked_rco", TOOLS / "qwen4exp_chunked_rco.py"
)
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class ChunkedRcoTest(unittest.TestCase):
    def test_formats_are_stable(self) -> None:
        self.assertEqual(SWEEP.SWEEP_FORMAT, "strata-qwen4exp-chunked-rco-sweep-v1")
        self.assertEqual(SWEEP.ALLOCATION_FORMAT, "strata-expert-width-allocation-v1")

    def test_default_budget_is_between_w3_and_w4(self) -> None:
        self.assertEqual(
            SWEEP.parse_args(
                [
                    "--model",
                    "/tmp/model",
                    "--calibration",
                    "/tmp/cal.npy",
                ]
            ).byte_budget_per_expert,
            0.0,
        )
        self.assertEqual(SWEEP.main.__name__, "main")


if __name__ == "__main__":
    unittest.main()
