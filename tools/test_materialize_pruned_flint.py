#!/usr/bin/env python3
"""Tests for the Flint pruned-checkpoint materializer."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "materialize_pruned_flint", TOOLS / "materialize_pruned_flint.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MaterializePrunedFlintTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def source(self) -> tuple[Path, Path]:
        source = self.root / "source"
        source.mkdir()
        weight_map = {"model.embed_tokens.weight": "model.safetensors"}
        for layer in range(2, 49):
            weight_map[f"model.layers.{layer}.mlp.gate.weight"] = "model.safetensors"
            for expert in range(256):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    key = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                    weight_map[key] = "model.safetensors"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                weight_map[f"model.layers.{layer}.mlp.shared_expert.{projection}.weight"] = (
                    "model.safetensors"
                )
        (source / "config.json").write_text(json.dumps({"model_type": "flint", "num_experts": 256}))
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (source / "tokenizer.json").write_text("{}", encoding="utf-8")
        mask = self.root / "mask.json"
        mask.write_text(
            json.dumps(
                {
                    "schema": MODULE.MASK_FORMAT,
                    "source_experts": 256,
                    "keep_ids": [[0, 2, 4] for _ in range(47)],
                }
            )
        )
        return source, mask

    def test_dry_run_reconciles_exact_tensor_budget(self) -> None:
        source, mask = self.source()
        result = MODULE.materialize(source, mask, self.root / "out", dry_run=True)
        self.assertEqual(result["keep_per_layer"], 3)
        self.assertEqual(result["selected_tensors"], 612)
        self.assertEqual(result["omitted_tensors"], 35_673)

    def test_mask_rejects_duplicate_or_out_of_range_ids(self) -> None:
        source, mask = self.source()
        document = json.loads(mask.read_text())
        document["keep_ids"][0] = [0, 0, 2]
        mask.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.materialize(source, mask, self.root / "out", dry_run=True)


if __name__ == "__main__":
    unittest.main()
