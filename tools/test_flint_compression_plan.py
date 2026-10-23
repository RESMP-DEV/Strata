#!/usr/bin/env python3
"""Tests for the Flint compression metadata planner."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "flint_compression_plan", TOOLS / "flint_compression_plan.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FlintCompressionPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        weight_map = {"other.weight": "model.safetensors"}
        for layer in range(2, 49):
            weight_map[f"model.layers.{layer}.mlp.gate.weight"] = "model.safetensors"
            for expert in range(256):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    weight_map[f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"] = (
                        "model.safetensors"
                    )
            for projection in ("gate_proj", "up_proj", "down_proj"):
                weight_map[f"model.layers.{layer}.mlp.shared_expert.{projection}.weight"] = (
                    "model.safetensors"
                )
        self.config = self.root / "config.json"
        self.config.write_text(
            json.dumps({"model_type": "flint", "num_experts": 256, "num_experts_per_tok": 6})
        )
        self.index = self.root / "model.safetensors.index.json"
        self.index.write_text(json.dumps({"weight_map": weight_map}))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_primary_lane_removes_exactly_25_percent(self) -> None:
        result = MODULE.lane(192)
        self.assertEqual(result["prune_per_layer"], 64)
        self.assertEqual(result["pruned_expert_slots"], 3_008)
        self.assertEqual(result["bf16_removed_bytes"], 18_924_699_648)
        self.assertEqual(result["mxfp4_removed_bytes"], 5_026_873_344)

    def test_plan_validates_exact_tensor_inventory(self) -> None:
        plan = MODULE.make_plan(self.config, self.index)
        self.assertEqual(plan["index_tensors"], 36_285)
        self.assertEqual(plan["tensor_counts"]["routed_experts"], 36_096)
        self.assertEqual(plan["lanes"]["rco_primary"]["keep_per_layer"], 192)

    def test_rejects_wrong_model_type(self) -> None:
        document = json.loads(self.config.read_text())
        document["model_type"] = "qwen"
        self.config.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "model_type"):
            MODULE.make_plan(self.config, self.index)

    def test_rejects_missing_expert_tensor(self) -> None:
        document = json.loads(self.index.read_text())
        target = next(key for key in document["weight_map"] if ".experts.0.gate_proj" in key)
        del document["weight_map"][target]
        self.index.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "unexpected Flint tensor counts"):
            MODULE.make_plan(self.config, self.index)


if __name__ == "__main__":
    unittest.main()
