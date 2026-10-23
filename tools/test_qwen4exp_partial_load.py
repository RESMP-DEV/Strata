#!/usr/bin/env python3
"""Host-safe tests for Qwen4Exp checkpoint name mapping and config trimming."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "qwen4exp_partial_load", TOOLS / "qwen4exp_partial_load.py"
)
assert SPEC is not None and SPEC.loader is not None
LOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOAD)


class FakeTextConfig:
    num_hidden_layers = 48
    layer_types = ["linear_attention"] * 48
    ple_layer_ids = [2]
    mtp = {"num_hidden_layers": 1}


class Qwen4ExpPartialLoadTest(unittest.TestCase):
    def test_checkpoint_name_maps_text_model_prefix(self) -> None:
        weight_map = {
            "model.language_model.embed_tokens.weight": "a.safetensors",
            "model.language_model.layers.0.mlp.gate.weight": "b.safetensors",
            "lm_head.weight": "c.safetensors",
        }
        self.assertEqual(
            LOAD.checkpoint_name("model.embed_tokens.weight", weight_map),
            "model.language_model.embed_tokens.weight",
        )
        self.assertEqual(
            LOAD.checkpoint_name("model.layers.0.mlp.gate.weight", weight_map),
            "model.language_model.layers.0.mlp.gate.weight",
        )
        self.assertEqual(LOAD.checkpoint_name("lm_head.weight", weight_map), "lm_head.weight")
        self.assertIsNone(LOAD.checkpoint_name("model.missing.weight", weight_map))

    def test_trim_config_keeps_only_the_requested_layers(self) -> None:
        config = LOAD.trim_config(FakeTextConfig(), 1)
        self.assertEqual(config.num_hidden_layers, 1)
        self.assertEqual(config.layer_types, ["linear_attention"])
        self.assertEqual(config.ple_layer_ids, [])
        self.assertIsNone(config.mtp)

    def test_trim_config_keeps_ple_when_its_layer_is_retained(self) -> None:
        config = LOAD.trim_config(FakeTextConfig(), 2)
        self.assertEqual(config.ple_layer_ids, [2])

    def test_trim_config_rejects_out_of_range_layers(self) -> None:
        with self.assertRaisesRegex(LOAD.LoadError, "keep_layers must be in"):
            LOAD.trim_config(FakeTextConfig(), 49)


if __name__ == "__main__":
    unittest.main()
