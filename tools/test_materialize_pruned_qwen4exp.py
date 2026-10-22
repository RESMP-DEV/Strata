#!/usr/bin/env python3
"""Tests for the streaming Flash-Next pruned-checkpoint materializer."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "materialize_pruned_qwen4exp", TOOLS / "materialize_pruned_qwen4exp.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def expert_names(layer: int) -> tuple[str, str, str]:
    prefix = f"model.language_model.layers.{layer}.mlp"
    return (
        f"{prefix}.gate.weight",
        f"{prefix}.experts.gate_up_proj",
        f"{prefix}.experts.down_proj",
    )


class MaterializePrunedQwen4ExpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def source(self, keep: int, *, tokenizer: bool = True) -> tuple[Path, Path]:
        source = self.root / "source"
        source.mkdir()
        weight_map = {
            "model.language_model.norm.weight": "model.safetensors",
            "lm_head.weight": "model.safetensors",
            "model.visual.weight": "model.safetensors",
            "mtp.weight": "model.safetensors",
        }
        rows = []
        for layer in range(48):
            for name in expert_names(layer):
                weight_map[name] = "model.safetensors"
            rows.append(
                [
                    (expert * 7 + layer) % MODULE.custom_coder_plan.N_EXPERTS
                    for expert in range(keep)
                ]
            )
        (source / "config.json").write_text(
            json.dumps({"text_config": {"num_experts": 512, "hidden_size": 3}}),
            encoding="utf-8",
        )
        (source / "model.safetensors").write_bytes(b"")
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}),
            encoding="utf-8",
        )
        mask = self.root / "mask.json"
        mask.write_text(
            json.dumps(
                {
                    "format": "strata-reap-mask-v1",
                    "n_expert": MODULE.custom_coder_plan.N_EXPERTS,
                    "keep_per_layer": keep,
                    "keep_ids": rows,
                }
            ),
            encoding="utf-8",
        )
        if tokenizer:
            (source / "tokenizer.json").write_text("{}", encoding="utf-8")
        return source, mask

    def test_tensor_classification_and_shape_contracts(self) -> None:
        router, gate_up, down = expert_names(7)
        self.assertEqual(MODULE.classify_tensor(router), ("router", 7))
        self.assertEqual(MODULE.classify_tensor(gate_up), ("gate_up", 7))
        self.assertEqual(MODULE.classify_tensor(down), ("down", 7))
        self.assertEqual(MODULE.prune_shape((512, 3), "router", 288), (288, 3))
        self.assertEqual(MODULE.prune_shape((512, 4, 3), "gate_up", 288), (288, 4, 3))
        self.assertEqual(MODULE.prune_shape((512, 3, 2), "down", 288), (288, 3, 2))

    def test_dry_run_counts_text_tensors_and_omits_vision_mtp(self) -> None:
        source, mask = self.source(288)
        result = MODULE.materialize(source, mask, self.root / "out", dry_run=True)
        self.assertEqual(result["keep_per_layer"], 288)
        self.assertEqual(result["selected_tensors"], 146)
        self.assertEqual(result["omitted_tensors"], 2)
        self.assertEqual(result["tokenizer_files"], ["tokenizer.json"])

    def test_dry_run_rejects_a_checkpoint_without_tokenizer_files(self) -> None:
        source, mask = self.source(288, tokenizer=False)
        output = self.root / "out"
        with self.assertRaisesRegex(ValueError, "no tokenizer files"):
            MODULE.materialize(source, mask, output, dry_run=True)
        self.assertFalse(output.exists())

    def test_mask_expert_dimension_mismatch_is_rejected(self) -> None:
        source, mask = self.source(288)
        document = json.loads(mask.read_text(encoding="utf-8"))
        document["n_expert"] = 256
        mask.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "n_expert must be 512"):
            MODULE.materialize(source, mask, self.root / "out", dry_run=True)

    def test_output_config_is_text_only_and_records_source_experts(self) -> None:
        config = MODULE.output_config(
            {
                "text_config": {
                    "num_experts": 512,
                    "mtp": {
                        "layer_types": ["full_attention"],
                        "num_hidden_layers": 1,
                    },
                    "mtp_num_hidden_layers": 1,
                }
            },
            288,
        )
        self.assertEqual(config["architectures"], ["Qwen4ExpForCausalLM"])
        self.assertEqual(config["num_experts"], 288)
        self.assertEqual(config["original_num_experts"], 512)
        self.assertEqual(config["torch_dtype"], "bfloat16")
        self.assertEqual(config["mtp_num_hidden_layers"], 0)
        self.assertEqual(config["mtp"]["num_hidden_layers"], 0)
        self.assertEqual(config["mtp"]["layer_types"], [])

    def test_optional_tiny_safetensors_materialization(self) -> None:
        try:
            import torch
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("torch and safetensors are required for the tiny round trip")

        source, mask = self.source(256)
        (source / "model.safetensors").unlink()
        selected_rows = json.loads(mask.read_text(encoding="utf-8"))["keep_ids"]
        tensors: dict[str, object] = {
            "model.language_model.norm.weight": torch.randn(3),
            "lm_head.weight": torch.randn(3, 3),
            "model.visual.weight": torch.randn(2),
            "mtp.weight": torch.randn(2),
        }
        for layer in range(48):
            router, gate_up, down = expert_names(layer)
            source_experts = torch.arange(512, dtype=torch.float32)
            tensors[router] = source_experts.reshape(512, 1).expand(512, 3).contiguous()
            tensors[gate_up] = source_experts.reshape(512, 1, 1).expand(512, 4, 3).contiguous()
            tensors[down] = source_experts.reshape(512, 1, 1).expand(512, 3, 2).contiguous()
        save_file(tensors, str(source / "model.safetensors"), metadata={"format": "pt"})
        output = self.root / "out"
        result = MODULE.materialize(source, mask, output, dry_run=False)
        self.assertEqual(result["materialized_tensor_counts"]["router"], 48)
        self.assertEqual(result["materialized_tensor_counts"]["gate_up"], 48)
        self.assertEqual(result["materialized_tensor_counts"]["down"], 48)
        index = json.loads((output / "model.safetensors.index.json").read_text())
        self.assertEqual(len(index["weight_map"]), 146)
        with safe_open(output / "model.safetensors", framework="pt") as handle:
            self.assertEqual(
                tuple(handle.get_tensor("model.language_model.layers.0.mlp.gate.weight").shape),
                (256, 3),
            )
            self.assertNotIn("model.visual.weight", handle.keys())
            self.assertNotIn("mtp.weight", handle.keys())
        expert_map = json.loads((output / "expert-id-map.json").read_text())
        self.assertEqual(expert_map["physical_to_source"][0], selected_rows[0])
        self.assertNotEqual(
            expert_map["physical_to_source"][0],
            expert_map["physical_to_source"][1],
        )
        with safe_open(output / "model.safetensors", framework="pt") as handle:
            for layer in range(48):
                selection = torch.tensor(selected_rows[layer], dtype=torch.long)
                for name in expert_names(layer):
                    selected = handle.get_tensor(name)
                    self.assertTrue(torch.equal(selected, tensors[name].index_select(0, selection)))

    def test_source_expert_dimension_is_validated_before_index_select(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("torch and safetensors are required for this round trip")

        source, mask = self.source(256)
        (source / "model.safetensors").unlink()
        tensors = {
            "model.language_model.norm.weight": torch.randn(3),
            "lm_head.weight": torch.randn(3, 3),
        }
        router, gate_up, down = expert_names(0)
        tensors[router] = torch.randn(256, 3)
        tensors[gate_up] = torch.randn(256, 4, 3)
        tensors[down] = torch.randn(256, 3, 2)
        for layer in range(1, 48):
            for name in expert_names(layer):
                tensors[name] = torch.randn(512, 3)
        save_file(tensors, str(source / "model.safetensors"), metadata={"format": "pt"})
        with self.assertRaisesRegex(ValueError, "source expert dimension 256"):
            MODULE.materialize(source, mask, self.root / "out", dry_run=False)


if __name__ == "__main__":
    unittest.main()
