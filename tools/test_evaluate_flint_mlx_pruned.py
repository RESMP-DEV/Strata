#!/usr/bin/env python3
"""Tests for the MLX runtime-pruned Flint evaluator."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "evaluate_flint_mlx_pruned", TOOLS / "evaluate_flint_mlx_pruned.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EvaluateFlintMlxPrunedTest(unittest.TestCase):
    def write_mask(self, document: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "mask.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_load_mask_accepts_valid_document(self) -> None:
        path = self.write_mask(
            {
                "schema": "strata-flint-expert-mask-v1",
                "source_experts": 4,
                "keep_ids": [[0, 2], [1, 3]],
            }
        )
        self.assertEqual(MODULE.load_mask(path), [[0, 2], [1, 3]])

    def test_load_mask_rejects_duplicate_expert(self) -> None:
        path = self.write_mask(
            {
                "schema": "strata-flint-expert-mask-v1",
                "source_experts": 4,
                "keep_ids": [[0, 0]],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate or out-of-range"):
            MODULE.load_mask(path)

    def test_load_mask_rejects_uneven_rows(self) -> None:
        path = self.write_mask(
            {
                "schema": "strata-flint-expert-mask-v1",
                "source_experts": 4,
                "keep_ids": [[0], [1, 2]],
            }
        )
        with self.assertRaisesRegex(ValueError, "different widths"):
            MODULE.load_mask(path)


if __name__ == "__main__":
    unittest.main()
