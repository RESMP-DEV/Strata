#!/usr/bin/env python3
"""Host-safe tests for the Holo4 MXFP4 W4A16 quant-aware RCO prototype."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "holo4_mxfp4_quant_aware_rco",
    TOOLS / "holo4_mxfp4_quant_aware_rco.py",
)
assert SPEC is not None and SPEC.loader is not None
RCO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RCO)


class ContractTest(unittest.TestCase):
    def test_mxfp4_bytes_include_group_scales(self) -> None:
        self.assertEqual(RCO.mxfp4_bytes((1024, 2048)), 1_114_112)
        self.assertEqual(RCO.mxfp4_bytes((2048, 512)), 557_056)
        self.assertEqual(RCO.expert_bytes((1024, 2048), (2048, 512)), 1_671_168)

    def test_impact_prune_removes_lowest_measured_error(self) -> None:
        mass = [0.9, 0.1, 0.5, 0.2]
        error = [0.5, 0.01, 0.2, 0.02]
        self.assertEqual(RCO.choose_prune(mass, error, 2, mode="impact"), [1, 3])

    def test_router_prune_removes_lowest_mass(self) -> None:
        mass = [0.9, 0.1, 0.5, 0.2]
        error = [0.5, 0.01, 0.2, 0.02]
        self.assertEqual(RCO.choose_prune(mass, error, 2, mode="router"), [1, 3])

    def test_random_prune_is_seeded_and_bounded(self) -> None:
        mass = [0.1, 0.2, 0.3, 0.4]
        error = [0.1, 0.2, 0.3, 0.4]
        first = RCO.choose_prune(mass, error, 2, mode="random", seed=7)
        second = RCO.choose_prune(mass, error, 2, mode="random", seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(0 <= index < 4 for index in first))

    def test_bad_mode_and_count_are_rejected(self) -> None:
        with self.assertRaisesRegex(RCO.RcoError, "unknown prune mode"):
            RCO.choose_prune([0.1], [0.1], 1, mode="bogus")
        with self.assertRaisesRegex(RCO.RcoError, "prune count"):
            RCO.choose_prune([0.1], [0.1], 2, mode="impact")


class TorchQuantizerTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed on this host")

    def test_e2m1_ties_round_to_even_index(self) -> None:
        import torch

        magnitudes = torch.tensor(
            [[0.24, 0.25, 0.4, 0.74, 0.75, 1.0, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0]]
        )
        self.assertEqual(
            RCO.e2m1_index(magnitudes).tolist()[0],
            [0, 0, 1, 1, 2, 2, 2, 4, 4, 6, 6, 7],
        )

    def test_scale_keeps_the_block_maximum_inside_e2m1(self) -> None:
        import torch

        weight = torch.zeros(1, 32, dtype=torch.bfloat16)
        weight[0, 0] = 0.0576171875
        weight[0, 1] = 0.02294921875
        weight[0, 2] = -0.0032501220703125
        quantized = RCO.mxfp4_dequant(weight)
        self.assertEqual(
            quantized[0, :3].tolist(), [0.0625, 0.0234375, 0.0]
        )

    def test_mxfp4_block_scale_is_power_of_two(self) -> None:
        import torch

        weight = torch.tensor([[8.0, -8.0, 4.0, 2.0, 1.0, 0.0, 3.0, -6.0]], dtype=torch.bfloat16)
        quantized = RCO.mxfp4_dequant(weight)
        expected = torch.tensor([[8.0, -8.0, 4.0, 2.0, 1.0, 0.0, 3.0, -6.0]], dtype=torch.bfloat16)
        self.assertEqual(quantized.tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
