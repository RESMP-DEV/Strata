#!/usr/bin/env python3
"""Tests for the custom Coder mask and footprint planner."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import get_type_hints

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("custom_coder_plan", TOOLS / "custom_coder_plan.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CustomCoderPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_mask(self, keep: int) -> Path:
        rows = [list(range(keep)) for _ in range(MODULE.N_LAYERS)]
        path = self.root / f"mask-{keep}.json"
        path.write_text(
            json.dumps(
                {
                    "format": "strata-reap-mask-v1",
                    "n_layers": MODULE.N_LAYERS,
                    "n_expert": MODULE.N_EXPERTS,
                    "keep_per_layer": keep,
                    "keep_ids": rows,
                }
            ),
            encoding="utf-8",
        )
        return path

    def write_parent(self) -> Path:
        path = self.root / "allocation.txt"
        lines = [
            "# RCO per-tensor GGUF allocation (expert-pruned release)",
            "# Section 2, per expert: tensor[index]: type",
        ]
        for layer in range(MODULE.N_LAYERS):
            for expert in range(MODULE.N_EXPERTS):
                quant = "IQ3_S" if expert < 256 else ""
                for kind in ("gate", "up", "down"):
                    lines.append(f"blk.{layer}.ffn_{kind}_exps.weight[{expert}]: {quant}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def make_args(self) -> Namespace:
        return Namespace(
            mask=self.write_mask(288),
            parent_allocation=self.write_parent(),
            model=self.root / "model",
            rco_root=self.root / "RCO",
            llama_convert=self.root / "convert.py",
            llama_imatrix=self.root / "llama-imatrix",
            llama_quantize=self.root / "llama-quantize",
            calibration_raw_tokens=self.root / "cal.i32",
            control_raw_tokens=self.root / "control.i32",
            calibration_tokens=self.root / "cal.pt",
            evaluation_tokens=self.root / "eval.pt",
            work_root=self.root / "work",
            rco_mask_out=self.root / "actual-own-mask.pt",
            parent_rco_mask_out=self.root / "actual-parent-mask.pt",
            recipe="iq3s-iq4nl",
            sequence_length=2048,
            calibration_chunks=256,
            batch_size=1,
            threads=32,
            python="python3",
            split_max_size="40G",
        )

    def test_parent_and_own_masks_are_validated_and_compared(self) -> None:
        own_path = self.write_mask(288)
        parent_path = self.write_parent()
        keep, own = MODULE.load_own_mask(own_path)
        parent = MODULE.parse_parent_allocation(parent_path)
        self.assertEqual((keep, len(own), len(parent)), (288, 48, 48))
        comparison = MODULE.compare_masks(own, parent)
        self.assertEqual(comparison.intersection, 48 * 256)
        self.assertEqual(comparison.union, 48 * 288)
        self.assertAlmostEqual(comparison.jaccard, 256 / 288)
        self.assertAlmostEqual(comparison.parent_capture, 1.0)
        self.assertAlmostEqual(comparison.own_capture, 256 / 288)

    def test_disagreeing_projection_is_rejected(self) -> None:
        path = self.write_parent()
        lines = path.read_text(encoding="utf-8").splitlines()
        target = "blk.0.ffn_down_exps.weight[0]: IQ3_S"
        lines[lines.index(target)] = "blk.0.ffn_down_exps.weight[0]:"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "retention disagreement"):
            MODULE.parse_parent_allocation(path)

    def test_invalid_keep_budget_is_rejected(self) -> None:
        path = self.write_mask(320)
        with self.assertRaisesRegex(ValueError, "keep count 320"):
            MODULE.load_own_mask(path)

    def test_mask_source_expert_count_is_required(self) -> None:
        path = self.write_mask(288)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["n_expert"] = 256
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "n_expert must be 512"):
            MODULE.load_own_mask(path)

    def test_budgets_use_exact_block_geometry(self) -> None:
        result = MODULE.budget(288, "iq3s-iq4nl")
        self.assertEqual(result["kept_experts"], 13_824)
        self.assertEqual(result["removed_experts"], 10_752)
        self.assertAlmostEqual(result["bpw"], 3.7916666666666665)
        self.assertEqual(result["expert_bytes"], 32_204_390_400)
        self.assertAlmostEqual(result["expert_gib"], 29.99267578125)
        self.assertAlmostEqual(
            result["relative_to_parent_expert_bytes"],
            32_204_390_400 / 25_145_942_999,
        )

    def test_manifest_contains_bounded_execution_plan(self) -> None:
        args = self.make_args()
        manifest = MODULE.build_manifest(args, 288)
        self.assertEqual(manifest["format"], "strata-custom-coder-plan-v1")
        self.assertEqual(manifest["selected_budget"]["kept_experts"], 13_824)
        self.assertIn("--per-layer-budget", manifest["commands"]["search_own_mask"]["argv"])
        quantize = manifest["commands"]["quantize_experts"]["argv"]
        self.assertIn("ffn_gate_exps=IQ3_S", quantize)
        self.assertIn("ffn_down_exps=IQ4_NL", quantize)
        self.assertIn("COPY", quantize)
        imatrix_value_index = quantize.index(str(self.root / "work" / "own-288-imatrix.gguf"))
        quantized_output_index = quantize.index(
            str(self.root / "work" / "flash-next-own-288-iq3s-iq4nl.gguf")
        )
        args.llama_quantize = Path(quantize[imatrix_value_index])
        control = MODULE.build_manifest(args, 288)["commands"][
            "quantize_standard_calibration_control"
        ]["argv"]
        self.assertEqual(control[0], str(args.llama_quantize))
        self.assertEqual(
            control[imatrix_value_index],
            str(self.root / "work" / "standard-ptq-imatrix.gguf"),
        )
        self.assertEqual(
            control[quantized_output_index],
            str(self.root / "work" / "flash-next-own-288-iq3s-iq4nl-standard-cal.gguf"),
        )

        args.llama_quantize = Path(quantize[quantized_output_index])
        control = MODULE.build_manifest(args, 288)["commands"][
            "quantize_standard_calibration_control"
        ]["argv"]
        self.assertEqual(control[0], str(args.llama_quantize))
        self.assertEqual(
            control[imatrix_value_index],
            str(self.root / "work" / "standard-ptq-imatrix.gguf"),
        )
        self.assertEqual(
            control[quantized_output_index],
            str(self.root / "work" / "flash-next-own-288-iq3s-iq4nl-standard-cal.gguf"),
        )
        actual_own = str(self.root / "actual-own-mask.pt")
        actual_parent = str(self.root / "actual-parent-mask.pt")
        self.assertEqual(manifest["commands"]["search_own_mask"]["argv"][-1], actual_own)
        self.assertEqual(
            manifest["commands"]["eval_own_mask"]["argv"][
                manifest["commands"]["eval_own_mask"]["argv"].index("--eval-mask") + 1
            ],
            actual_own,
        )
        self.assertEqual(
            manifest["commands"]["eval_parent_mask"]["argv"][
                manifest["commands"]["eval_parent_mask"]["argv"].index("--eval-mask") + 1
            ],
            actual_parent,
        )
        materialize = manifest["commands"]["materialize_pruned_bf16"]["argv"]
        self.assertEqual(materialize[materialize.index("--prune-mask") + 1], actual_own)

    def test_parent_control_requires_an_explicit_destination(self) -> None:
        args = Namespace(
            mask=self.write_mask(288),
            parent_allocation=self.write_parent(),
            work_root=self.root / "work",
            rco_mask_out=None,
            parent_rco_mask_out=None,
        )
        with self.assertRaisesRegex(ValueError, "--parent-rco-mask-out is required"):
            MODULE.rco_mask_paths(args)

    def test_rco_mask_destinations_reject_collisions_and_input_aliases(self) -> None:
        args = Namespace(
            mask=self.write_mask(288),
            parent_allocation=self.write_parent(),
            work_root=self.root / "work",
            rco_mask_out=self.root / "own.pt",
            parent_rco_mask_out=self.root / "own.pt",
        )
        with self.assertRaisesRegex(ValueError, "destinations must be different"):
            MODULE.rco_mask_paths(args)

        args.parent_rco_mask_out = self.root / "parent.pt"
        args.rco_mask_out = args.mask
        with self.assertRaisesRegex(ValueError, "must not overwrite own mask input"):
            MODULE.rco_mask_paths(args)

        args.rco_mask_out = self.root / "own.pt"
        args.parent_rco_mask_out = args.parent_allocation
        with self.assertRaisesRegex(ValueError, "must not overwrite parent allocation input"):
            MODULE.rco_mask_paths(args)

    def test_command_annotation_matches_the_returned_shape(self) -> None:
        hints = get_type_hints(MODULE.command)
        self.assertEqual(hints["return"], MODULE.CommandSpec)
        result = MODULE.command(["llama-quantize", "--imatrix", "a b.gguf"])
        self.assertEqual(result["argv"], ["llama-quantize", "--imatrix", "a b.gguf"])
        self.assertEqual(result["shell"], "llama-quantize --imatrix 'a b.gguf'")

    def test_quantize_argv_builder_and_layout_guard(self) -> None:
        quantizer = self.root / "llama-quantize"
        imatrix_path = self.root / "imatrix.gguf"
        input_gguf = self.root / "in.gguf"
        output_gguf = self.root / "out.gguf"
        argv = MODULE.quantize_arguments(
            quantizer, imatrix_path, input_gguf, output_gguf, "IQ3_S", 32
        )
        MODULE.validate_quantize_layout(argv, imatrix_path, input_gguf, output_gguf, 32)
        self.assertEqual(argv[0], str(quantizer))
        self.assertEqual(argv[argv.index("--imatrix") + 1], str(imatrix_path))
        self.assertEqual(argv[-2:], ["COPY", "32"])
        broken = list(argv)
        broken[-2] = "ADD"
        with self.assertRaisesRegex(ValueError, "must end with COPY"):
            MODULE.validate_quantize_layout(broken, imatrix_path, input_gguf, output_gguf, 32)

    def test_control_quantize_differs_only_in_imatrix_and_output(self) -> None:
        args = self.make_args()
        commands = MODULE.build_manifest(args, 288)["commands"]
        own = commands["quantize_experts"]["argv"]
        control = commands["quantize_standard_calibration_control"]["argv"]
        imatrix_index = own.index(str(args.work_root / "own-288-imatrix.gguf"))
        output_index = own.index(str(args.work_root / "flash-next-own-288-iq3s-iq4nl.gguf"))
        differing = {
            index
            for index, (left, right) in enumerate(zip(own, control, strict=True))
            if left != right
        }
        self.assertEqual(differing, {imatrix_index, output_index})
        self.assertEqual(control[imatrix_index], str(args.work_root / "standard-ptq-imatrix.gguf"))
        self.assertEqual(
            control[output_index],
            str(args.work_root / "flash-next-own-288-iq3s-iq4nl-standard-cal.gguf"),
        )

    def test_manifest_accepts_already_parsed_masks(self) -> None:
        args = self.make_args()
        direct = MODULE.build_manifest(args, 288)
        keep, own = MODULE.load_own_mask(args.mask)
        parent = MODULE.parse_parent_allocation(args.parent_allocation)
        threaded = MODULE.build_manifest(args, keep, own=own, parent=parent)
        self.assertEqual(threaded, direct)
        self.assertEqual(threaded["mask_sha256"], MODULE.sha256(args.mask))
        self.assertEqual(
            threaded["parent_allocation_sha256"], MODULE.sha256(args.parent_allocation)
        )

    def test_manifest_rejects_parsed_masks_that_do_not_match_the_budget(self) -> None:
        args = self.make_args()
        parent = MODULE.parse_parent_allocation(args.parent_allocation)
        short_own = [set(range(256)) for _ in range(MODULE.N_LAYERS)]
        with self.assertRaisesRegex(ValueError, "does not keep 288 experts"):
            MODULE.build_manifest(args, 288, own=short_own, parent=parent)


if __name__ == "__main__":
    unittest.main()
