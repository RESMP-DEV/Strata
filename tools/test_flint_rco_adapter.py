#!/usr/bin/env python3
"""Tests for the Flint stock-RCO topology adapter."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover - depends on the host
    torch = None

NNBase = torch.nn.Module if torch is not None else object


TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("flint_rco_adapter", TOOLS / "flint_rco_adapter.py")
assert SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class FlintRcoAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as error:
            raise unittest.SkipTest("torch is not installed") from error

        self.torch = torch
        torch.manual_seed(7)
        self.config = SimpleNamespace(
            num_hidden_layers=5,
            mlp_only_layers=[0, 1],
            num_experts=4,
            num_experts_per_tok=2,
        )
        self.blocks = [TinyMoeBlock(nn, 3) for _ in range(3)]
        layers = [
            SimpleNamespace(mlp=None),
            SimpleNamespace(mlp=None),
            *[SimpleNamespace(mlp=block) for block in self.blocks],
        ]
        self.model = SimpleNamespace(config=self.config, model=SimpleNamespace(layers=layers))

    def test_counts_use_routed_layers_not_dense_layers(self) -> None:
        self.assertEqual(ADAPTER.get_moe_info(self.model), (3, 4, 2))
        self.assertEqual(ADAPTER.resolve_layer_ids(self.config), (2, 3, 4))

    def test_all_ones_mask_preserves_baseline(self) -> None:
        hidden = self.torch.randn(2, 5, 3)
        expected = [block(hidden).clone() for block in self.blocks]
        state = {"ste_masks": self.torch.ones(3, 4)}
        wrappers = ADAPTER.install_wrappers(self.model, 3, 4, state)
        try:
            actual = [block(hidden) for block in self.blocks]
        finally:
            ADAPTER.remove_wrappers(wrappers)
        for got, want in zip(actual, expected, strict=True):
            self.assertTrue(self.torch.equal(got, want))

    def test_zero_mask_keeps_shared_expert_only(self) -> None:
        hidden = self.torch.randn(1, 4, 3)
        state = {"ste_masks": self.torch.zeros(3, 4)}
        wrappers = ADAPTER.install_wrappers(self.model, 3, 4, state)
        try:
            actual = self.blocks[0](hidden)
        finally:
            ADAPTER.remove_wrappers(wrappers)
        expected = self.blocks[0].shared_expert(hidden)
        self.assertTrue(self.torch.equal(actual, expected))

    def test_hard_mask_renormalizes_over_survivors(self) -> None:
        hidden = self.torch.randn(1, 3, 3)
        block = self.blocks[0]
        baseline = block(hidden)
        with self.torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
            block.gate.bias.copy_(self.torch.tensor([2.0, 1.0, 0.5, 4.0]))
            block.shared_expert.weight.copy_(self.torch.eye(3, 3))
        prune_mask = self.torch.zeros(1, 4, dtype=self.torch.bool)
        prune_mask[0, 3] = True
        wrapper = ADAPTER._HardMaskWrapper(block, 0, prune_mask[0], 2)
        try:
            masked = wrapper._forward(hidden)
        finally:
            wrapper.restore()
        expected = block(hidden)
        self.assertTrue(self.torch.allclose(masked, expected, atol=1e-6))
        self.assertFalse(self.torch.equal(masked, baseline))

    def test_mask_gradient_flows_through_survival(self) -> None:
        mask = self.torch.ones(4, requires_grad=True)
        state = {"ste_masks": self.torch.stack([mask, self.torch.ones(4), self.torch.ones(4)])}
        wrappers = ADAPTER.install_wrappers(self.model, 3, 4, state)
        try:
            output = self.blocks[0](self.torch.randn(1, 3, 3)).sum()
            output.backward()
        finally:
            ADAPTER.remove_wrappers(wrappers)
        self.assertTrue(self.torch.isfinite(mask.grad).all())

    def test_physical_keep_rows_use_survival_semantics(self) -> None:
        prune_mask = self.torch.zeros(2, 4, dtype=self.torch.bool)
        prune_mask[0, 1] = True
        prune_mask[1, 3] = True
        self.assertEqual(ADAPTER.physical_keep_rows(prune_mask), [[0, 2, 3], [0, 1, 2]])

    def test_bad_mask_shape_fails(self) -> None:
        state = {"ste_masks": self.torch.ones(3, 3)}
        wrappers = ADAPTER.install_wrappers(self.model, 3, 4, state)
        try:
            with self.assertRaisesRegex(ADAPTER.RcoAdapterError, "mask shape"):
                self.blocks[0](self.torch.randn(1, 2, 3))
        finally:
            ADAPTER.remove_wrappers(wrappers)


class TinyExperts(NNBase):
    def __init__(self, torch_nn, expert_count: int, hidden: int):
        import torch

        super().__init__()
        self.expert_count = expert_count
        self.hidden = hidden
        self.weight = torch_nn.Parameter(torch.Tensor(expert_count, hidden))

    def __call__(self, hidden, indices, weights):
        selected = self.weight.index_select(0, indices.reshape(-1))
        weighted = selected * weights.reshape(-1, 1).to(selected.dtype)
        return weighted.sum(dim=0).expand(hidden.shape[0], hidden.shape[1], self.hidden)


class TinyMoeBlock(NNBase):
    def __init__(self, torch_nn, hidden: int):
        import torch

        super().__init__()
        self.torch = torch
        self.gate = torch_nn.Linear(hidden, 4, bias=True)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.copy_(torch.Tensor([1.0, 2.0, 3.0, 4.0]))
        self.experts = TinyExperts(torch_nn, 4, hidden)
        with torch.no_grad():
            self.experts.weight.normal_(0, 0.1)
        self.shared_expert = torch_nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden):
        scores = self.gate(hidden).float().sigmoid()
        weights, indices = self.torch.topk(scores, 2, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return self.experts(hidden, indices, weights.to(hidden.dtype)) + self.shared_expert(hidden)


if __name__ == "__main__":
    unittest.main()
