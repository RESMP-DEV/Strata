#!/usr/bin/env python3
"""Host-safe tests for the exact MXFP4A16 byte cost model."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("w4a16_cost_model", TOOLS / "w4a16_cost_model.py")
assert SPEC is not None and SPEC.loader is not None
COST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COST)


class W4A16CostModelTest(unittest.TestCase):
    def test_linear_bytes_are_packed_words_plus_group_scales(self) -> None:
        packed, scales = COST.linear_bytes(640, 2560)
        self.assertEqual(packed, 640 * 320 * 4)
        self.assertEqual(scales, 640 * 80)
        self.assertEqual(packed + scales, 640 * 2560 // 2 + 640 * 80)

    def test_fused_expert_bytes_multiply_by_expert_count(self) -> None:
        packed, scales = COST.fused_expert_bytes(288, 1280, 2560)
        single_packed, single_scales = COST.linear_bytes(1280, 2560)
        self.assertEqual(packed, single_packed * 288)
        self.assertEqual(scales, single_scales * 288)

    def test_partial_groups_are_rounded_up(self) -> None:
        packed, scales = COST.linear_bytes(3, 40)
        self.assertEqual(packed, 3 * 5 * 4)
        self.assertEqual(scales, 3 * 2)

    def test_invalid_shape_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid linear shape"):
            COST.linear_bytes(0, 2560)

    def test_ignored_modules_cover_lm_head_and_router_gates(self) -> None:
        self.assertTrue(COST.ignored_module("lm_head"))
        self.assertTrue(COST.ignored_module("model.layers.3.mlp.gate"))
        self.assertFalse(COST.ignored_module("model.layers.3.mlp.shared_expert.gate_proj"))
        self.assertFalse(COST.ignored_module("model.layers.3.self_attn.q_proj"))


if __name__ == "__main__":
    unittest.main()
