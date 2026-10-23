#!/usr/bin/env python3
"""Tests for Flint held-out quality evaluation helpers."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "evaluate_flint_quality", TOOLS / "evaluate_flint_quality.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
try:
    SPEC.loader.exec_module(MODULE)
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"evaluator dependency unavailable: {error}") from error


class EvaluateFlintQualityTest(unittest.TestCase):
    def test_select_windows_is_aligned_and_checks_bounds(self) -> None:
        import numpy as np

        source = np.arange(24, dtype=np.int64).reshape(2, 12)
        inputs, targets = MODULE.select_windows(source, 1, 1, 7)
        self.assertEqual(inputs[0].tolist(), source[1, :7].tolist())
        self.assertEqual(targets[0].tolist(), source[1, 1:8].tolist())
        with self.assertRaisesRegex(ValueError, "exceed the token file"):
            MODULE.select_windows(source, 1, 2, 7)

    def test_summary_reports_nll_bits_and_top1(self) -> None:
        import numpy as np

        nll = np.array([[0.0, math.log(2.0)]], dtype=np.float32)
        top1 = np.array([[True, False]])
        result = MODULE.summarize_token_metrics(nll, top1)
        self.assertEqual(result["tokens"], 2)
        self.assertAlmostEqual(result["mean_nll_nats"], math.log(2.0) / 2)
        self.assertAlmostEqual(result["mean_nll_bits"], 0.5)
        self.assertAlmostEqual(result["perplexity"], math.sqrt(2.0))
        self.assertEqual(result["top1_correct"], 1)
        self.assertEqual(result["top1_accuracy"], 0.5)

    def test_paired_bootstrap_is_deterministic_and_bounded(self) -> None:
        import numpy as np

        baseline = np.array([[0.1, 0.3], [0.2, 0.4]], dtype=np.float32)
        candidate = np.array([[0.2, 0.4], [0.1, 0.3]], dtype=np.float32)
        first = MODULE.paired_bootstrap(baseline, candidate, 64, 7)
        second = MODULE.paired_bootstrap(baseline, candidate, 64, 7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_delta_nats"], 0.0)
        self.assertEqual(first["delta_nats_ci95"][0], min(first["delta_nats_ci95"]))
        self.assertEqual(first["delta_nats_ci95"][-1], max(first["delta_nats_ci95"]))
        with self.assertRaisesRegex(ValueError, "different shapes"):
            MODULE.paired_bootstrap(baseline, candidate[:1], 8, 7)

    def test_comparison_reports_top1_contingency(self) -> None:
        import numpy as np

        baseline_nll = np.zeros((1, 3), dtype=np.float32)
        candidate_nll = np.full((1, 3), 0.1, dtype=np.float32)
        baseline_top1 = np.array([True, True, False])
        candidate_top1 = np.array([True, False, True])
        result = MODULE.compare_metrics(
            baseline_nll, candidate_nll, baseline_top1, candidate_top1, 8, 7
        )
        self.assertEqual(result["both_correct"], 1)
        self.assertEqual(result["baseline_only_correct"], 1)
        self.assertEqual(result["candidate_only_correct"], 1)
        self.assertAlmostEqual(result["top1_agreement"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
