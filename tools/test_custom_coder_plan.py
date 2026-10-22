#!/usr/bin/env python3
"""Tests for the custom Coder mask and footprint planner."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("custom_coder_plan", TOOLS / "custom_coder_plan.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
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
        mask = self.write_mask(288)
        parent = self.write_parent()
        args = Namespace(
            mask=mask,
            parent_allocation=parent,
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
            recipe="iq3s-iq4nl",
            sequence_length=2048,
            calibration_chunks=256,
            batch_size=1,
            threads=32,
            python="python3",
            split_max_size="40G",
        )
        manifest = MODULE.build_manifest(args, 288)
        self.assertEqual(manifest["format"], "strata-custom-coder-plan-v1")
        self.assertEqual(manifest["selected_budget"]["kept_experts"], 13_824)
        self.assertIn("--per-layer-budget", manifest["commands"]["search_own_mask"]["argv"])
        quantize = manifest["commands"]["quantize_experts"]["argv"]
        self.assertIn("ffn_gate_exps=IQ3_S", quantize)
        self.assertIn("ffn_down_exps=IQ4_NL", quantize)
        self.assertIn("COPY", quantize)


if __name__ == "__main__":
    unittest.main()
