#!/usr/bin/env python3
"""Tests for the streaming Flint router-control collector."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "collect_flint_router_scores", TOOLS / "collect_flint_router_scores.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
try:
    SPEC.loader.exec_module(MODULE)
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"collector dependency unavailable: {error}") from error


class CollectFlintRouterScoresTest(unittest.TestCase):
    def test_keep_rows_is_deterministic_and_exact_budget(self) -> None:
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest("torch is not installed") from error
        scores = torch.zeros(47, 256)
        for layer in range(47):
            scores[layer, 3] = 10.0
            scores[layer, 1] = 9.0
            scores[layer, 200] = 9.0
        rows = MODULE.keep_rows(scores, 192)
        self.assertEqual(len(rows), 47)
        self.assertTrue(all(len(row) == 192 for row in rows))
        self.assertEqual(rows[0][:3], [3, 1, 200])

    def test_keep_rows_rejects_bad_shape(self) -> None:
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest("torch is not installed") from error
        with self.assertRaisesRegex(ValueError, "router score shape"):
            MODULE.keep_rows(torch.zeros(46, 256), 192)


if __name__ == "__main__":
    unittest.main()
