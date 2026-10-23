#!/usr/bin/env python3
"""Tests for exact leave-one-out expert impact search."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "flint_prune_impact_search", TOOLS / "flint_prune_impact_search.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
try:
    SPEC.loader.exec_module(MODULE)
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"impact search dependency unavailable: {error}") from error


class FlintPruneImpactSearchTest(unittest.TestCase):
    def test_parse_remove_rounds(self) -> None:
        self.assertEqual(MODULE.parse_remove_rounds("32, 16,8, 8"), [32, 16, 8, 8])
        with self.assertRaisesRegex(ValueError, "positive integers"):
            MODULE.parse_remove_rounds("16,0")

    def test_analyze_block_removes_unused_then_low_impact_experts(self) -> None:
        import torch

        hidden = 2
        experts = 4
        block = torch.nn.Module()
        block.gate = torch.nn.Linear(hidden, experts, bias=False)
        with torch.no_grad():
            block.gate.weight.copy_(torch.tensor([[3.0, 0.0], [2.0, 0.0], [0.1, 0.0], [1.0, 0.0]]))
        block.experts = torch.nn.Module()
        block.experts.gate_proj = torch.nn.Parameter(torch.eye(hidden).repeat(experts, 1, 1))
        block.experts.up_proj = torch.nn.Parameter(torch.eye(hidden).repeat(experts, 1, 1))
        block.experts.down_proj = torch.nn.Parameter(
            torch.stack(
                [
                    torch.eye(hidden) * 4.0,
                    torch.eye(hidden) * 2.0,
                    torch.eye(hidden),
                    torch.eye(hidden),
                ]
            )
        )
        activations = [torch.tensor([[[1.0, -0.5], [0.2, 0.8]]])]
        result = MODULE.analyze_block(
            block, activations, remove_rounds=[1, 1], top_k=2, num_experts=experts
        )
        self.assertEqual(result["keep_ids"], [0, 1])
        self.assertEqual(result["rounds"][0]["removed_experts"], [2])

    def test_round_impacts_are_finite_and_zero_for_unselected_experts(self) -> None:
        import torch

        hidden = 2
        experts = 4
        block = torch.nn.Module()
        block.gate = torch.nn.Linear(hidden, experts, bias=False)
        with torch.no_grad():
            block.gate.weight.copy_(torch.tensor([[4.0, 0.0], [3.0, 0.0], [0.1, 0.0], [2.0, 0.0]]))
        block.experts = torch.nn.Module()
        identity = torch.eye(hidden)
        block.experts.gate_proj = torch.nn.Parameter(identity.repeat(experts, 1, 1))
        block.experts.up_proj = torch.nn.Parameter(identity.repeat(experts, 1, 1))
        block.experts.down_proj = torch.nn.Parameter(identity.repeat(experts, 1, 1))
        removed = torch.zeros(experts, dtype=torch.bool)
        impacts, frequency, energy, score_sum = MODULE.round_impacts(
            block, torch.ones(1, 3, hidden), removed, 2
        )
        self.assertTrue(torch.isfinite(impacts).all())
        self.assertGreater(float(energy), 0.0)
        self.assertEqual(frequency[:2].tolist(), [3.0, 3.0])
        self.assertEqual(frequency[2:].tolist(), [0.0, 0.0])
        self.assertTrue(torch.isfinite(score_sum).all())


if __name__ == "__main__":
    unittest.main()
