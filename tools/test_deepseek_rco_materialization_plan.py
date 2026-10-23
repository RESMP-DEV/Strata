#!/usr/bin/env python3
"""Tests for the dependency-free DeepSeek RCO materialization planner."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tools.test_deepseek_rco_plan import MANIFEST_PATH, build_config, build_index

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
SPEC = importlib.util.spec_from_file_location(
    "deepseek_rco_materialization_plan",
    TOOLS / "deepseek_rco_materialization_plan.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


RUNTIME_DEPENDENCY_PROBE = '''"""Fresh-interpreter probe: plan without torch or safetensors."""

import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(sys.argv[1])
CONFIG, INDEX, MANIFEST, MASK = (Path(value) for value in sys.argv[2:6])
RUNTIME_MODULES = {"torch", "safetensors"}


def runtime_modules(loaded):
    return sorted(name for name in loaded if name.split(".")[0] in RUNTIME_MODULES)


before = set(sys.modules)
spec = importlib.util.spec_from_file_location(
    "deepseek_rco_materialization_plan_probe",
    TOOLS / "deepseek_rco_materialization_plan.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
plan = module.build_plan(
    config_path=CONFIG,
    index_path=INDEX,
    manifest_path=MANIFEST,
    mask_path=MASK,
)
after = set(sys.modules)
print(
    json.dumps(
        {
            "status": "ok",
            "format": plan["format"],
            "source_tensor_count": plan["tensor_count"]["source"],
            "loaded": runtime_modules(after),
            "imported_by_probe": runtime_modules(after - before),
        }
    )
)
'''


def build_mask(keep_ids: list[int] | None = None) -> dict[str, Any]:
    """Build a valid uniform mask using the contract's dependency-free format."""
    row = list(range(216)) if keep_ids is None else keep_ids
    return {
        "format": MODULE.MASK_FORMAT,
        "model": {
            "repo": MODULE.PINNED_REPO,
            "revision": MODULE.PINNED_REVISION,
        },
        "backbone": {
            "layers": 40,
            "routed_experts": 384,
            "active_experts": 6,
            "keep_ids": [list(row) for _ in range(40)],
        },
        "mtp": {"policy": "preserve", "layers": 3, "routed_experts": 128},
    }


class MaterializationTest(unittest.TestCase):
    """Shared synthetic inventory fixture; no ``*.safetensors`` file is created."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.root = Path(cls.temp.name)
        cls.manifest_value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.manifest_path = cls.root / "deepseek_v41_manifest.json"
        cls.config_path = cls.root / "config.json"
        cls.index_path = cls.root / "model.safetensors.index.json"
        cls.mask_path = cls.root / "mask.json"
        cls.manifest_path.write_text(json.dumps(cls.manifest_value), encoding="utf-8")
        cls.config_path.write_text(json.dumps(build_config(cls.manifest_value)), encoding="utf-8")
        cls.index_path.write_text(json.dumps(build_index(cls.manifest_value)), encoding="utf-8")
        cls.mask_path.write_text(json.dumps(build_mask()), encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def plan(
        self,
        *,
        config_path: Path | None = None,
        index_path: Path | None = None,
        mask_path: Path | None = None,
    ) -> dict[str, Any]:
        return MODULE.build_plan(
            config_path=config_path or self.config_path,
            index_path=index_path or self.index_path,
            manifest_path=self.manifest_path,
            mask_path=mask_path or self.mask_path,
        )

    def write_mask(self, mask: dict[str, Any]) -> Path:
        path = self.root / "mask-under-test.json"
        path.write_text(json.dumps(mask), encoding="utf-8")
        return path

    def write_config(self, mutate: Any) -> Path:
        config = build_config(self.manifest_value)
        mutate(config)
        path = self.root / "config-under-test.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path


class MetadataOnlyPlanningTest(MaterializationTest):
    def test_plans_without_shard_files(self) -> None:
        plan = self.plan()
        self.assertEqual(plan["format"], MODULE.PLAN_FORMAT)
        self.assertEqual(list(self.root.glob("*.safetensors")), [])

    def test_fresh_process_planning_imports_no_runtime_dependency(self) -> None:
        # A fresh interpreter is the only honest proof that the planner itself
        # needs neither torch nor safetensors: an earlier test in this process
        # may already have imported them into the shared module table.
        probe = self.root / "no_runtime_dependency_probe.py"
        probe.write_text(RUNTIME_DEPENDENCY_PROBE, encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(probe),
                str(TOOLS),
                str(self.config_path),
                str(self.index_path),
                str(self.manifest_path),
                str(self.mask_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["format"], MODULE.PLAN_FORMAT)
        self.assertEqual(report["source_tensor_count"], 96_085)
        self.assertEqual(report["loaded"], [])
        self.assertEqual(report["imported_by_probe"], [])

    def test_primary_lane_counts_and_bytes(self) -> None:
        plan = self.plan()
        self.assertEqual(
            plan["tensor_count"],
            {
                "source": 96_085,
                "omit": 40_320,
                "copy": 55_645,
                "slice_rows": 120,
                "target": 55_765,
                "retained_backbone_routed": 51_840,
                "preserved_mtp_routed": 2_304,
                "shared_pass_through": 258,
                "dense_total": 1_363,
                "dense_gate_slice": 120,
                "dense_pass_through": 1_243,
            },
        )
        self.assertEqual(plan["byte_accounting"]["omitted_backbone_routed_bytes"], 126_340_300_800)
        self.assertEqual(plan["byte_accounting"]["retained_backbone_routed_bytes"], 162_437_529_600)

    def test_one_pruned_expert_expands_to_exactly_six_routed_tensors(self) -> None:
        mask = build_mask(list(range(383)))
        plan = self.plan(mask_path=self.write_mask(mask))
        all_omitted = set(plan["selection"]["omit"])
        omitted = {name for name in all_omitted if name.startswith("layers.0.")}
        expected = {
            f"layers.0.ffn.experts.383.w{gate}.{part}"
            for gate in (1, 2, 3)
            for part in ("weight", "scale")
        }
        self.assertEqual(omitted, expected)
        self.assertEqual(len(omitted), 6)
        self.assertEqual(plan["tensor_count"]["omit"], 240)
        self.assertFalse(
            any(name.startswith("layers.") and ".shared_experts." in name for name in omitted)
        )

    def test_all_backbone_routers_are_sliced_and_not_copied(self) -> None:
        plan = self.plan()
        expected = {
            f"layers.{layer}.ffn.gate.{part}"
            for layer in range(40)
            for part in ("weight", "bias", "bias_vl")
        }
        sliced = {entry["name"] for entry in plan["selection"]["slice_rows"]}
        copied = set(plan["selection"]["copy"])
        self.assertEqual(sliced, expected)
        self.assertEqual(len(sliced), 120)
        self.assertTrue(all(entry["axis"] == 0 for entry in plan["selection"]["slice_rows"]))
        self.assertFalse(sliced & copied)

    def test_mtp_is_preserved(self) -> None:
        plan = self.plan()
        self.assertEqual(plan["mtp"]["routed_bytes"], 7_219_445_760)
        copied = set(plan["selection"]["copy"])
        omitted = set(plan["selection"]["omit"])
        index = build_index(self.manifest_value)
        mtp_names = {name for name in index["weight_map"] if name.startswith("mtp.")}
        self.assertTrue(mtp_names <= copied)
        self.assertFalse(any(name.startswith("mtp.") for name in omitted))
        self.assertEqual(
            sum(name.startswith("mtp.") and ".experts." in name for name in copied), 2_304
        )


class InputValidationTest(MaterializationTest):
    def assert_mask_refused(self, mutate: Any) -> None:
        mask = build_mask()
        mutate(mask)
        path = self.write_mask(mask)
        with self.assertRaisesRegex(MODULE.base.PlanError, "mask|keep_ids|mtp|retained|model"):
            self.plan(mask_path=path)

    def test_refuses_differing_uniform_retained_counts(self) -> None:
        def mutate(mask: dict[str, Any]) -> None:
            mask["backbone"]["keep_ids"][1] = mask["backbone"]["keep_ids"][1][:-1]

        self.assert_mask_refused(mutate)

    def test_refuses_retained_count_below_active_floor(self) -> None:
        def mutate(mask: dict[str, Any]) -> None:
            mask["backbone"]["keep_ids"] = [list(range(6)) for _ in range(40)]
            mask["backbone"]["keep_ids"][0] = list(range(5))

        self.assert_mask_refused(mutate)

    def test_refuses_wrong_mask_length(self) -> None:
        def mutate(mask: dict[str, Any]) -> None:
            del mask["backbone"]["keep_ids"][0]

        self.assert_mask_refused(mutate)

    def test_refuses_duplicate_and_noncanonical_ids(self) -> None:
        cases = (
            lambda mask: mask["backbone"]["keep_ids"][0].insert(0, 0),
            lambda mask: mask["backbone"]["keep_ids"][0].__setitem__(0, True),
            lambda mask: mask["backbone"]["keep_ids"][0].reverse(),
            lambda mask: mask["backbone"]["keep_ids"][0].__setitem__(0, 384),
            lambda mask: mask["backbone"]["keep_ids"][0].__setitem__(0, "0"),
            lambda mask: mask.__setitem__("extra", {}),
            lambda mask: mask["mtp"].__setitem__("policy", "prune"),
        )
        for mutate in cases:
            with self.subTest(mutate=mutate):
                self.assert_mask_refused(mutate)

    def test_refuses_malformed_index(self) -> None:
        def changed(mutate: Any) -> Path:
            index = build_index(self.manifest_value)
            mutate(index)
            path = self.root / "index-under-test.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            return path

        bad_paths = {
            "bad-json": '{"metadata":',
            "missing-metadata": json.dumps({"weight_map": {}}),
            "wrong-total": json.dumps(
                {
                    "metadata": {"total_size": 1},
                    "weight_map": build_index(self.manifest_value)["weight_map"],
                }
            ),
        }
        for label, text in bad_paths.items():
            with self.subTest(case=label):
                path = self.root / f"{label}.json"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(MODULE.base.PlanError):
                    self.plan(index_path=path)

        def remove_tensor(index: dict[str, Any]) -> None:
            del index["weight_map"][sorted(index["weight_map"])[0]]

        def remove_shard(index: dict[str, Any]) -> None:
            names = sorted(index["weight_map"])
            removed_shard = index["weight_map"][names[0]]
            for name in names:
                if index["weight_map"][name] == removed_shard:
                    index["weight_map"][name] = "model-00002-of-00048.safetensors"

        def bad_basename(index: dict[str, Any]) -> None:
            index["weight_map"][sorted(index["weight_map"])[0]] = "model-1-of-48.safetensors"

        def traversal(index: dict[str, Any]) -> None:
            index["weight_map"][sorted(index["weight_map"])[0]] = (
                "../model-00001-of-00048.safetensors"
            )

        def unknown(index: dict[str, Any]) -> None:
            index["weight_map"]["unexpected.tensor"] = "model-00001-of-00048.safetensors"

        for label, mutate in {
            "wrong-count": remove_tensor,
            "wrong-shard-count": remove_shard,
            "bad-basename": bad_basename,
            "path-traversal": traversal,
            "unknown-tensor": unknown,
        }.items():
            with self.subTest(case=label), self.assertRaises(MODULE.base.PlanError):
                self.plan(index_path=changed(mutate))

    def test_refuses_duplicate_weight_map_key(self) -> None:
        index = build_index(self.manifest_value)
        first = sorted(index["weight_map"])[0]
        shard = index["weight_map"][first]
        remainder = {name: value for name, value in index["weight_map"].items() if name != first}
        text = (
            json.dumps({"metadata": index["metadata"], "weight_map": {first: shard}})[:-2]
            + ","
            + json.dumps({first: shard})[1:-1]
            + ","
            + json.dumps({first: shard})[1:-1]
            + ","
            + json.dumps(remainder)[1:-1]
            + "}}"
        )
        path = self.root / "duplicate-index.json"
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(MODULE.base.PlanError, "duplicate JSON object key"):
            self.plan(index_path=path)

    def test_refuses_shard_tensor_count_over_manifest_maximum(self) -> None:
        index = build_index(self.manifest_value)
        source_shard = "model-00002-of-00048.safetensors"
        target_shard = "model-00001-of-00048.safetensors"
        moved = 0
        for name, shard in index["weight_map"].items():
            if shard == source_shard and moved < 341:
                index["weight_map"][name] = target_shard
                moved += 1
        path = self.root / "overfull-index.json"
        path.write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaisesRegex(MODULE.base.PlanError, "outside \\[2, 2342\\]"):
            self.plan(index_path=path)

    def test_refuses_changed_config_geometry(self) -> None:
        changes = {
            "num_hidden_layers": 39,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "hidden_size": 4096,
            "moe_intermediate_size": 2048,
            "num_nextn_predict_layers": 2,
            "dspark_n_routed_experts": 64,
            "dspark_num_experts_per_tok": 2,
        }
        for key, value in changes.items():
            with self.subTest(config_key=key):

                def mutate(config: dict[str, Any], key: str = key, value: int = value) -> None:
                    config["text_config"][key] = value

                with self.assertRaisesRegex(MODULE.base.PlanError, key):
                    self.plan(config_path=self.write_config(mutate))


class ActionAndCliTest(MaterializationTest):
    def test_every_tensor_has_exactly_one_action(self) -> None:
        plan = self.plan()
        index = build_index(self.manifest_value)
        omitted = set(plan["selection"]["omit"])
        copied = set(plan["selection"]["copy"])
        sliced = {entry["name"] for entry in plan["selection"]["slice_rows"]}
        actions = omitted | copied | sliced
        self.assertEqual(len(actions), 96_085)
        self.assertEqual(
            len(omitted) + len(copied) + len(sliced),
            96_085,
        )
        self.assertEqual(actions, set(index["weight_map"]))

    def test_header_deferred_fields_are_honest(self) -> None:
        plan = self.plan()
        bytes_document = plan["byte_accounting"]
        deferred = {
            "router_slice_delta_bytes",
            "target_payload_bytes",
            "target_header_bytes",
            "target_shard_file_bytes",
        }
        self.assertEqual(
            {key for key in deferred if bytes_document[key] == MODULE.HEADER_DEFERRED}, deferred
        )
        self.assertEqual(
            bytes_document["download_manifest_minus_index_bytes"],
            25_022_295,
        )

    def test_shard_group_emission_and_counts(self) -> None:
        index = build_index(self.manifest_value)
        only_omitted_shard = "model-00001-of-00048.safetensors"
        destination_shards = [
            f"model-{position:05d}-of-00048.safetensors" for position in range(2, 49)
        ]
        moved = 0
        for name, shard in list(index["weight_map"].items()):
            if shard == only_omitted_shard:
                classification = MODULE.base.classify_tensor(name)
                is_routed_backbone = (
                    classification.kind == "routed" and classification.scope == "layers"
                )
                if is_routed_backbone and classification.expert not in set(range(216)):
                    continue
                index["weight_map"][name] = destination_shards[moved % len(destination_shards)]
                moved += 1
        path = self.root / "emission-index.json"
        path.write_text(json.dumps(index), encoding="utf-8")
        plan = self.plan(index_path=path)
        groups = {group["source"]: group for group in plan["shards"]}
        self.assertEqual(len(groups), 48)
        self.assertFalse(groups[only_omitted_shard]["emit"])
        self.assertEqual(groups[only_omitted_shard]["target_tensor_count"], 0)
        self.assertTrue(any(group["emit"] for group in groups.values()))
        for group in groups.values():
            expected_source = sum(
                value == group["source"] for value in index["weight_map"].values()
            )
            self.assertEqual(group["source_tensor_count"], expected_source)

    def test_cli_writes_deterministic_json_and_refuses_bad_inputs(self) -> None:
        output = self.root / "materialization-plan.json"
        argv = [
            "--config",
            str(self.config_path),
            "--index",
            str(self.index_path),
            "--mask",
            str(self.mask_path),
            "--manifest",
            str(self.manifest_path),
            "--out",
            str(output),
        ]
        self.assertEqual(MODULE.main(argv), 0)
        payload = output.read_text(encoding="utf-8")
        document = json.loads(payload)
        self.assertEqual(payload, json.dumps(document, indent=2, sort_keys=True) + "\n")
        original_utc_now = MODULE.utc_now
        try:
            MODULE.utc_now = lambda: "2026-10-22T00:00:00Z"
            self.assertEqual(MODULE.main(argv), 0)
            first_fixed_payload = output.read_text(encoding="utf-8")
            self.assertEqual(MODULE.main(argv), 0)
            self.assertEqual(first_fixed_payload, output.read_text(encoding="utf-8"))
        finally:
            MODULE.utc_now = original_utc_now
        for label, path in {
            "config": self.config_path,
            "index": self.index_path,
            "manifest": self.manifest_path,
            "mask": self.mask_path,
        }.items():
            self.assertEqual(
                document["inputs"][label]["sha256"],
                MODULE.base.sha256(path),
            )

        bad_mask = self.write_mask({"not": "a mask"})
        bad_argv = list(argv)
        bad_argv[bad_argv.index("--mask") + 1] = str(bad_mask)
        with contextlib.redirect_stderr(io.StringIO()) as captured:
            exit_code = MODULE.main(bad_argv)
        self.assertEqual(exit_code, 2)
        self.assertIn("FAIL deepseek-rco-materialization-plan", captured.getvalue())
