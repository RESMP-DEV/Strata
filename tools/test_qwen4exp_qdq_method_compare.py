#!/usr/bin/env python3
"""Host-safe contract tests for the QDQ methodology comparison."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "qwen4exp_qdq_method_compare", TOOLS / "qwen4exp_qdq_method_compare.py"
)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


class QdqMethodCompareTest(unittest.TestCase):
    def test_signal_set_is_stable(self) -> None:
        self.assertEqual(COMPARE.SIGNALS, ("weight", "router", "qdq"))

    def test_format_is_stable(self) -> None:
        self.assertEqual(COMPARE.COMPARE_FORMAT, "strata-qwen4exp-qdq-method-compare-v1")


if __name__ == "__main__":
    unittest.main()
