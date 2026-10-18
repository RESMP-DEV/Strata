#!/usr/bin/env python3
"""Tests for the exact DeepSeek-V4.1-Flash-1017 manifest gate."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_deepseek

MANIFEST_PATH = ROOT / "data" / "deepseek_v41_manifest.json"


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def build_config(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_type": manifest["model"]["model_type"],
        "architectures": copy.deepcopy(manifest["model"]["architectures"]),
        "text_config": copy.deepcopy(manifest["text_config_invariants"]),
        "vision_config": copy.deepcopy(manifest["vision_config_invariants"]),
        "quantization_config": copy.deepcopy(manifest["quantization_config"]),
    }


def build_index(manifest: dict[str, Any], plan: check_deepseek.Plan) -> dict[str, Any]:
    names = sorted(check_deepseek.expected_tensor_names(plan))
    shards = int(manifest["checkpoint"]["shards"])
    weight_map: dict[str, str] = {}
    for position, name in enumerate(names):
        shard = position % shards + 1
        weight_map[name] = f"model-{shard:05d}-of-{shards:05d}.safetensors"
    return {
        "metadata": {"total_size": int(manifest["tensors"]["index_total_size_bytes"])},
        "weight_map": weight_map,
    }


def results_by_name(
    results: list[check_deepseek.CheckResult],
) -> dict[str, check_deepseek.CheckResult]:
    return {result.name: result for result in results}


class ManifestContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest()
        self.plan = check_deepseek.build_plan(self.manifest)

    def test_exact_official_shape_is_compiled(self) -> None:
        self.assertEqual(len(self.plan.families), 114)
        self.assertEqual(self.plan.layer_width, 1)
        self.assertEqual(self.plan.expert_width, 1)
        self.assertTrue(all(family.status == "published" for family in self.plan.families))

    def test_expected_names_cover_official_index_exactly_once(self) -> None:
        expected = check_deepseek.expected_tensor_names(self.plan)
        self.assertEqual(len(expected), 96_085)
        self.assertEqual(len(set(expected)), len(expected))
        self.assertEqual(sum(1 for name in expected if ".ffn.experts." in name), 94_464)

    def test_duplicate_family_names_are_refused(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["tensor_families"].append(copy.deepcopy(manifest["tensor_families"][0]))
        with self.assertRaisesRegex(check_deepseek.ManifestError, "duplicate family"):
            check_deepseek.build_plan(manifest)

    def test_overlapping_patterns_are_refused(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        first = copy.deepcopy(manifest["tensor_families"][0])
        first["family"] = "overlap"
        manifest["tensor_families"].append(first)
        plan = check_deepseek.build_plan(manifest)
        with self.assertRaisesRegex(check_deepseek.ManifestError, "both claim"):
            check_deepseek.expected_tensor_names(plan)

    def test_routed_selector_requires_routed_kind(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["tensor_families"][0]["experts"] = "routed"
        with self.assertRaisesRegex(check_deepseek.ManifestError, "not routed"):
            check_deepseek.build_plan(manifest)

    def test_bad_layer_group_is_refused(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["tensor_families"][0]["layers"] = "missing-group"
        with self.assertRaisesRegex(check_deepseek.ManifestError, "unknown layer group"):
            check_deepseek.build_plan(manifest)


class ConfigGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest()
        self.config = build_config(self.manifest)

    def test_matching_config_passes(self) -> None:
        results = check_deepseek.check_config(self.config, self.manifest)
        self.assertTrue(all(result.passed for result in results), results)

    def test_changed_text_invariant_fails(self) -> None:
        self.config["text_config"]["n_routed_experts"] = 256
        results = results_by_name(check_deepseek.check_config(self.config, self.manifest))
        self.assertFalse(results["config-text-invariants"].passed)
        self.assertIn("n_routed_experts", results["config-text-invariants"].detail)

    def test_changed_compress_ratio_fails(self) -> None:
        ratios = list(self.config["text_config"]["compress_ratios"])
        ratios[2], ratios[0] = ratios[0], ratios[2]
        self.config["text_config"]["compress_ratios"] = ratios
        results = results_by_name(check_deepseek.check_config(self.config, self.manifest))
        self.assertFalse(results["config-text-invariants"].passed)

    def test_type_strict_scalar_comparison_fails(self) -> None:
        self.config["text_config"]["rms_norm_eps"] = 1
        results = results_by_name(check_deepseek.check_config(self.config, self.manifest))
        self.assertFalse(results["config-text-invariants"].passed)


class IndexGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest()
        self.plan = check_deepseek.build_plan(self.manifest)
        self.index = build_index(self.manifest, self.plan)

    def run_index(self, index: dict[str, Any]) -> dict[str, check_deepseek.CheckResult]:
        return results_by_name(check_deepseek.run_gate(self.manifest, self.plan, index=index))

    def test_balanced_official_shape_passes(self) -> None:
        results = self.run_index(self.index)
        failed = {name: result.detail for name, result in results.items() if not result.passed}
        self.assertEqual(failed, {}, failed)
        self.assertEqual(
            results["tensor-coverage"].detail,
            "all 96085 expected tensors are present and classified",
        )

    def test_missing_metadata_total_is_structural_failure(self) -> None:
        index = copy.deepcopy(self.index)
        del index["metadata"]["total_size"]
        results = self.run_index(index)
        self.assertFalse(results["index-structure"].passed)
        self.assertFalse(results["tensor-coverage"].passed)

    def test_wrong_total_fails_only_total_gate(self) -> None:
        index = copy.deepcopy(self.index)
        index["metadata"]["total_size"] += 1
        results = self.run_index(index)
        self.assertFalse(results["index-total-size"].passed)
        self.assertTrue(results["tensor-coverage"].passed)

    def test_missing_and_unknown_tensors_are_refused(self) -> None:
        index = copy.deepcopy(self.index)
        removed = sorted(index["weight_map"])[0]
        del index["weight_map"][removed]
        index["weight_map"]["unexpected.tensor"] = next(iter(index["weight_map"].values()))
        results = self.run_index(index)
        self.assertTrue(results["tensor-count"].passed)
        self.assertFalse(results["tensor-coverage"].passed)
        self.assertIn("absent", results["tensor-coverage"].detail)
        self.assertIn("no family", results["tensor-coverage"].detail)

    def test_underfilled_shard_is_refused(self) -> None:
        index = copy.deepcopy(self.index)
        first = "model-00001-of-00048.safetensors"
        second = "model-00002-of-00048.safetensors"
        donors = sorted(name for name, shard in index["weight_map"].items() if shard == first)
        self.assertGreaterEqual(len(donors), 2)
        for name in donors[1:]:
            index["weight_map"][name] = second
        results = self.run_index(index)
        self.assertTrue(results["shard-count"].passed)
        self.assertFalse(results["shard-consistency"].passed)
        self.assertIn("below", results["shard-consistency"].detail)

    def test_duplicate_tensor_mapping_is_refused(self) -> None:
        class Duplicating(dict):
            def items(self):  # type: ignore[override]
                yield from super().items()
                first = next(iter(super().keys()))
                yield first, "model-00002-of-00048.safetensors"

        index = copy.deepcopy(self.index)
        index["weight_map"] = Duplicating(index["weight_map"])
        results = self.run_index(index)
        self.assertFalse(results["tensor-duplicates"].passed)


class CliTest(unittest.TestCase):
    def test_json_cli_reports_pass_and_fail(self) -> None:
        manifest = load_manifest()
        plan = check_deepseek.build_plan(manifest)
        config = build_config(manifest)
        index = build_index(manifest, plan)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            index_path = root / "index.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            index_path.write_text(json.dumps(index), encoding="utf-8")
            passing = check_deepseek.main(
                ["--config", str(config_path), "--index", str(index_path), "--json"]
            )
            self.assertEqual(passing, 0)
            config["text_config"]["hidden_size"] = 4096
            config_path.write_text(json.dumps(config), encoding="utf-8")
            failing = check_deepseek.main(
                ["--config", str(config_path), "--index", str(index_path), "--json"]
            )
            self.assertEqual(failing, 1)


if __name__ == "__main__":
    unittest.main()
