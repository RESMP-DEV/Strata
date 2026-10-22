#!/usr/bin/env python3
"""Tests for the DeepSeek-V4.1-Flash RCO planning and preflight tool."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent

SPEC = importlib.util.spec_from_file_location("deepseek_rco_plan", TOOLS / "deepseek_rco_plan.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

# The planner imports the checked-in shape gate itself and puts scripts/ on the
# path; reuse that module rather than importing it a second way.
check_deepseek = MODULE.check_deepseek

MANIFEST_PATH = ROOT / "data" / "deepseek_v41_manifest.json"


def build_index(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build a synthetic index that matches the pinned 1017 inventory exactly."""
    plan = check_deepseek.build_plan(manifest)
    names = sorted(check_deepseek.expected_tensor_names(plan))
    shards = int(manifest["checkpoint"]["shards"])
    weight_map = {
        name: f"model-{position % shards + 1:05d}-of-{shards:05d}.safetensors"
        for position, name in enumerate(names)
    }
    return {
        "metadata": {"total_size": int(manifest["tensors"]["index_total_size_bytes"])},
        "weight_map": weight_map,
    }


def build_config(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build a synthetic config that matches the pinned 1017 text geometry."""
    return {
        "model_type": manifest["model"]["model_type"],
        "architectures": copy.deepcopy(manifest["model"]["architectures"]),
        "text_config": copy.deepcopy(manifest["text_config_invariants"]),
        "vision_config": copy.deepcopy(manifest["vision_config_invariants"]),
        "quantization_config": copy.deepcopy(manifest["quantization_config"]),
    }


class ExactArithmeticTest(unittest.TestCase):
    def test_rounding_modes_are_exact(self) -> None:
        self.assertEqual(MODULE.round_ratio(Fraction(5, 2), "floor"), 2)
        self.assertEqual(MODULE.round_ratio(Fraction(5, 2), "nearest"), 3)
        self.assertEqual(MODULE.round_ratio(Fraction(5, 2), "ceil"), 3)
        self.assertEqual(MODULE.round_ratio(Fraction(7, 3), "floor"), 2)
        self.assertEqual(MODULE.round_ratio(Fraction(7, 3), "nearest"), 2)
        self.assertEqual(MODULE.round_ratio(Fraction(7, 3), "ceil"), 3)

    def test_retained_fraction_is_exact(self) -> None:
        retained = MODULE.retained_from_fraction(384, Fraction(9, 16), "floor")
        self.assertEqual(retained, 216)

    def test_retained_per_active_is_exact(self) -> None:
        ratio = Fraction(288, 10)
        self.assertEqual(MODULE.retained_from_active_ratio(6, ratio, "floor", cap=384), 172)
        self.assertEqual(MODULE.retained_from_active_ratio(6, ratio, "nearest", cap=384), 173)
        self.assertEqual(MODULE.retained_from_active_ratio(6, ratio, "ceil", cap=384), 173)

    def test_exact_bytes_rejects_a_fractional_byte_count(self) -> None:
        self.assertEqual(MODULE.exact_bytes(Fraction(1, 2), 100), 50)
        with self.assertRaisesRegex(MODULE.PlanError, "not an integer"):
            MODULE.exact_bytes(Fraction(1, 3), 100)

    def test_unknown_rounding_mode_is_refused(self) -> None:
        with self.assertRaisesRegex(MODULE.PlanError, "unknown rounding mode"):
            MODULE.round_ratio(Fraction(1, 2), "banker")


class PackingTest(unittest.TestCase):
    def test_expert_bytes_use_the_authoritative_shard_header(self) -> None:
        self.assertEqual(MODULE.expert_tensor_bytes("w1"), 5898240 + 368640)
        self.assertEqual(MODULE.expert_tensor_bytes("w2"), 5898240 + 368640)
        self.assertEqual(MODULE.expert_tensor_bytes("w3"), 5898240 + 368640)
        self.assertEqual(MODULE.expert_bytes(), 18800640)

    def test_scales_are_per_row_not_block_scales(self) -> None:
        self.assertEqual(MODULE.EXPERT_TENSOR_GEOMETRY["w1"]["scale"]["shape"], (2304, 160))
        self.assertEqual(MODULE.EXPERT_TENSOR_GEOMETRY["w2"]["scale"]["shape"], (5120, 72))
        self.assertEqual(MODULE.EXPERT_TENSOR_GEOMETRY["w1"]["weight"]["dtype"], "I8")
        self.assertEqual(MODULE.EXPERT_TENSOR_GEOMETRY["w1"]["scale"]["dtype"], "F8_E8M0")


class ClassificationTest(unittest.TestCase):
    def test_routed_tensor_is_classified(self) -> None:
        item = MODULE.classify_tensor("layers.0.ffn.experts.7.w1.weight")
        self.assertEqual(item.kind, "routed")
        self.assertEqual(item.scope, "layers")
        self.assertEqual(item.layer, 0)
        self.assertEqual(item.expert, 7)
        self.assertEqual(item.dtype, "I8")
        self.assertEqual(item.packed_bytes, 2304 * 2560)

    def test_mtp_routed_tensor_is_classified(self) -> None:
        item = MODULE.classify_tensor("mtp.2.ffn.experts.127.w3.scale")
        self.assertEqual(item.kind, "routed")
        self.assertEqual(item.scope, "mtp")
        self.assertEqual(item.layer, 2)
        self.assertEqual(item.expert, 127)
        self.assertEqual(item.dtype, "F8_E8M0")

    def test_shared_and_dense_tensors_are_classified(self) -> None:
        shared = MODULE.classify_tensor("layers.5.ffn.shared_experts.w2.weight")
        self.assertEqual(shared.kind, "shared")
        dense = MODULE.classify_tensor("layers.5.attn.wkv.weight")
        self.assertEqual(dense.kind, "dense")
        self.assertIsNone(dense.packed_bytes)

    def test_out_of_range_expert_is_refused(self) -> None:
        with self.assertRaisesRegex(MODULE.PlanError, "outside"):
            MODULE.classify_tensor("layers.0.ffn.experts.384.w1.weight")
        with self.assertRaisesRegex(MODULE.PlanError, "outside"):
            MODULE.classify_tensor("mtp.0.ffn.experts.128.w1.weight")
        with self.assertRaisesRegex(MODULE.PlanError, "outside"):
            MODULE.classify_tensor("layers.40.ffn.experts.0.w1.weight")


class InventoryGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.index = build_index(self.manifest)

    def test_official_inventory_classifies_exactly(self) -> None:
        classes, counts = MODULE.gate_inventory(self.index["weight_map"], self.manifest)
        self.assertEqual(len(classes), 96_085)
        self.assertEqual(counts["routed_backbone"], 92_160)
        self.assertEqual(counts["routed_mtp"], 2_304)
        self.assertEqual(counts["shared_expert"], 258)
        self.assertEqual(counts["dense"], 1_363)
        self.assertEqual(
            counts["routed_backbone"]
            + counts["routed_mtp"]
            + counts["shared_expert"]
            + counts["dense"],
            96_085,
        )

    def test_unknown_tensor_is_refused(self) -> None:
        weight_map = dict(self.index["weight_map"])
        weight_map["unexpected.tensor"] = "model-00001-of-00048.safetensors"
        with self.assertRaisesRegex(MODULE.PlanError, "match no known family"):
            MODULE.gate_inventory(weight_map, self.manifest)

    def test_missing_tensor_is_refused(self) -> None:
        weight_map = dict(self.index["weight_map"])
        removed = sorted(weight_map)[0]
        del weight_map[removed]
        with self.assertRaisesRegex(MODULE.PlanError, "absent from the index"):
            MODULE.gate_inventory(weight_map, self.manifest)

    def test_malformed_index_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.PlanError, "not valid JSON"):
                MODULE.read_index(path)
            path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.PlanError, "total_size"):
                MODULE.read_index(path)
            path.write_text(json.dumps({"metadata": {"total_size": 1}}), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.PlanError, "weight_map"):
                MODULE.read_index(path)

    def test_malformed_config_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"model_type": "deepseek_v41"}), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.PlanError, "text_config"):
                MODULE.read_config(path)


class PlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = Path(self.temp.name)
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.config_path = self.root / "config.json"
        self.index_path = self.root / "model.safetensors.index.json"
        self.config_path.write_text(json.dumps(build_config(self.manifest)), encoding="utf-8")
        self.index_path.write_text(json.dumps(build_index(self.manifest)), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def plan(self, **overrides: Any) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "config_path": self.config_path,
            "index_path": self.index_path,
            "manifest_path": MANIFEST_PATH,
            "retained_fraction": Fraction(9, 16),
            "active_ratio": Fraction(144, 5),
            "preserve_mtp": True,
        }
        arguments.update(overrides)
        return MODULE.build_plan(**arguments)

    def test_plan_emits_the_declared_format_with_hashes(self) -> None:
        plan = self.plan()
        self.assertEqual(plan["format"], "strata-deepseek-rco-plan-v1")
        self.assertEqual(len(plan["inputs"]["config_sha256"]), 64)
        self.assertEqual(len(plan["inputs"]["index_sha256"]), 64)
        self.assertEqual(len(plan["inputs"]["manifest_sha256"]), 64)
        self.assertTrue(plan["non_claims"])
        self.assertTrue(plan["scope"])

    def test_inventory_is_classified_strictly(self) -> None:
        inventory = self.plan()["inventory"]
        self.assertEqual(inventory["total_tensors"], 96_085)
        self.assertEqual(inventory["routed_backbone_tensors"], 92_160)
        self.assertEqual(inventory["routed_mtp_tensors"], 2_304)
        self.assertEqual(inventory["shared_expert_tensors"], 258)
        self.assertEqual(inventory["dense_tensors"], 1_363)

    def test_budget_candidates_match_the_agreed_arithmetic(self) -> None:
        budgets = {entry["name"]: entry for entry in self.plan()["budgets"]}
        self.assertEqual(budgets["retained-fraction"]["retained_experts"], 216)
        self.assertEqual(budgets["retained-per-active-floor"]["retained_experts"], 172)
        self.assertEqual(budgets["retained-per-active-nearest"]["retained_experts"], 173)
        self.assertEqual(budgets["retained-per-active-ceil"]["retained_experts"], 173)

    def test_budget_bytes_are_exact(self) -> None:
        budgets = {entry["name"]: entry for entry in self.plan()["budgets"]}
        self.assertEqual(budgets["retained-fraction"]["retained_bytes"], 216 * 18800640)
        self.assertEqual(budgets["retained-per-active-floor"]["retained_bytes"], 172 * 18800640)
        self.assertEqual(budgets["retained-per-active-nearest"]["retained_bytes"], 173 * 18800640)
        fraction = budgets["retained-fraction"]["retained_fraction"]
        self.assertEqual(fraction["numerator"], 9)
        self.assertEqual(fraction["denominator"], 16)

    def test_mtp_experts_are_preserved_by_default(self) -> None:
        mtp = self.plan()["mtp"]
        self.assertTrue(mtp["preserved"])
        self.assertEqual(mtp["routed_experts"], 128)
        self.assertEqual(mtp["layers"], 3)
        self.assertEqual(mtp["total_bytes"], 3 * 128 * 18800640)

    def test_mtp_pruning_can_be_requested_explicitly(self) -> None:
        mtp = self.plan(preserve_mtp=False)["mtp"]
        self.assertFalse(mtp["preserved"])

    def test_comparator_is_the_qwen_288_of_512(self) -> None:
        comparator = self.plan()["comparator"]
        self.assertEqual(comparator["retained_experts"], 288)
        self.assertEqual(comparator["routed_experts"], 512)
        self.assertEqual(comparator["active_experts"], 10)

    def test_wrong_geometry_is_refused(self) -> None:
        config = build_config(self.manifest)
        config["text_config"]["n_routed_experts"] = 256
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.PlanError, "routed"):
            self.plan()

    def test_short_inventory_is_refused(self) -> None:
        index = build_index(self.manifest)
        removed = sorted(index["weight_map"])[0]
        del index["weight_map"][removed]
        self.index_path.write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.PlanError, "absent"):
            self.plan()


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = Path(self.temp.name)
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.config_path = self.root / "config.json"
        self.index_path = self.root / "model.safetensors.index.json"
        self.config_path.write_text(json.dumps(build_config(self.manifest)), encoding="utf-8")
        self.index_path.write_text(json.dumps(build_index(self.manifest)), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_cli_writes_the_plan(self) -> None:
        out = self.root / "plan.json"
        code = MODULE.main(
            [
                "--config",
                str(self.config_path),
                "--index",
                str(self.index_path),
                "--manifest",
                str(MANIFEST_PATH),
                "--out",
                str(out),
            ]
        )
        self.assertEqual(code, 0)
        document = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(document["format"], "strata-deepseek-rco-plan-v1")
        budgets = {entry["name"]: entry for entry in document["budgets"]}
        self.assertEqual(budgets["retained-fraction"]["retained_experts"], 216)
        self.assertEqual(budgets["retained-per-active-floor"]["retained_experts"], 172)
        self.assertEqual(budgets["retained-per-active-nearest"]["retained_experts"], 173)

    def test_cli_refuses_a_malformed_index(self) -> None:
        self.index_path.write_text("{not json", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as captured:
            code = MODULE.main(
                [
                    "--config",
                    str(self.config_path),
                    "--index",
                    str(self.index_path),
                    "--manifest",
                    str(MANIFEST_PATH),
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("not valid JSON", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
