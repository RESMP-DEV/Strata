#!/usr/bin/env python3
"""Tests for Flint MLX MXFP4 expert slicing."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "materialize_pruned_flint_mlx", TOOLS / "materialize_pruned_flint_mlx.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MaterializePrunedFlintMlxTest(unittest.TestCase):
    def write_mask(self, document: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "mask.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_maps_logical_rows_to_physical_layers(self) -> None:
        self.assertEqual(
            MODULE.map_physical_rows(5, [0, 1], [[0, 1], [2, 3], [1, 4]]),
            {2: [0, 1], 3: [2, 3], 4: [1, 4]},
        )

    def test_recognizes_routed_mlx_tensors(self) -> None:
        self.assertEqual(MODULE.routed_layer("model.layers.7.mlp.switch_mlp.gate_proj.scales"), 7)
        self.assertEqual(MODULE.routed_layer("model.layers.7.mlp.gate.weight"), 7)
        self.assertIsNone(MODULE.routed_layer("model.layers.7.mlp.shared_expert.gate_proj.weight"))

    def test_rejects_duplicate_experts(self) -> None:
        path = self.write_mask(
            {
                "schema": "strata-flint-expert-mask-v1",
                "source_experts": 4,
                "keep_ids": [[0, 0]],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate or out-of-range"):
            MODULE.load_mask(path)


if __name__ == "__main__":
    unittest.main()
