#!/usr/bin/env python3
"""Tests for the compression preflight's checkpoint and runtime contracts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_compression


def text_config() -> dict[str, object]:
    return {
        "model_type": "qwen4_exp_text",
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "ple_layer_ids": [2],
        "split_ngram_parts": 128,
        "layer_types": ["linear_attention"] * 3
        + ["full_attention"]
        + (["linear_attention"] * 3 + ["full_attention"]) * 11,
    }


def write_model(root: Path) -> None:
    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": text_config(),
    }
    weight_map: dict[str, str] = {}
    for layer in range(48):
        prefix = f"model.language_model.layers.{layer}"
        weight_map[f"{prefix}.mlp.gate.weight"] = "model.safetensors"
        weight_map[f"{prefix}.mlp.experts.gate_up_proj"] = "model.safetensors"
        weight_map[f"{prefix}.mlp.experts.down_proj"] = "model.safetensors"
    for shard in range(128):
        weight_map[
            f"model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        ] = "model.safetensors"
    index = {"metadata": {"total_size": 1}, "weight_map": weight_map}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"x")


class CompressionPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_checkpoint_metadata_passes(self) -> None:
        write_model(self.root)
        findings = check_compression.validate_model(self.root)
        self.assertTrue(all(finding.ok for finding in findings), [f for f in findings if not f.ok])
        self.assertEqual(len(findings), 17)

    def test_missing_expert_tensor_is_rejected(self) -> None:
        write_model(self.root)
        path = self.root / "model.safetensors.index.json"
        index = json.loads(path.read_text(encoding="utf-8"))
        del index["weight_map"]["model.language_model.layers.7.mlp.experts.down_proj"]
        path.write_text(json.dumps(index), encoding="utf-8")
        by_name = {finding.name: finding for finding in check_compression.validate_model(self.root)}
        self.assertFalse(by_name["model.expert_tensors"].ok)
        self.assertIn("total_missing=1", by_name["model.expert_tensors"].detail)

    def test_wrong_layer_mix_is_rejected(self) -> None:
        write_model(self.root)
        path = self.root / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["text_config"]["layer_types"][3] = "linear_attention"
        path.write_text(json.dumps(config), encoding="utf-8")
        by_name = {finding.name: finding for finding in check_compression.validate_model(self.root)}
        self.assertFalse(by_name["model.layer_types"].ok)
        self.assertIn("linear_attention=37 full_attention=11", by_name["model.layer_types"].detail)

    def test_incomplete_transfer_marker_is_rejected(self) -> None:
        write_model(self.root)
        (self.root / "model.safetensors.aria2").write_bytes(b"")
        by_name = {finding.name: finding for finding in check_compression.validate_model(self.root)}
        self.assertFalse(by_name["model.download_complete"].ok)

    def test_runtime_requires_qwen4_exp_support(self) -> None:
        old = check_compression.validate_runtime("5.7.0")
        new = check_compression.validate_runtime("5.20.2")
        self.assertFalse(old[0].ok)
        self.assertTrue(new[0].ok)
        self.assertIn("rco_declared=5.7.0", new[0].detail)

    def test_version_parser_accepts_local_suffix(self) -> None:
        self.assertEqual(check_compression.parse_version("5.20.2.dev0"), (5, 20, 2))
        with self.assertRaises(ValueError):
            check_compression.parse_version("latest")

    def test_tool_revision_and_entrypoints_are_checked(self) -> None:
        root = self.root / "RCO"
        (root / "src").mkdir(parents=True)
        for name in ("rco_search_prune.py", "requirements.txt"):
            (root / name).write_text("", encoding="utf-8")
        (root / "src/search.py").write_text("", encoding="utf-8")
        completed = subprocess.CompletedProcess(
            ["git"], 0, stdout=check_compression.RCO_REVISION + "\n", stderr=""
        )
        with mock.patch.object(check_compression.subprocess, "run", return_value=completed):
            findings = check_compression.validate_tool(
                "rco",
                root,
                check_compression.RCO_REVISION,
                ("rco_search_prune.py", "src/search.py", "requirements.txt"),
            )
        self.assertTrue(all(finding.ok for finding in findings), findings)


if __name__ == "__main__":
    unittest.main()
