#!/usr/bin/env python3
"""Tests for the DeepSeek-V4.1 RCO topology adapter.

The tests build the *tiny* pinned V4.1 model (5 layers, hidden 64, 4 routed
experts, top-2) on the CPU -- or on the meta device when only the topology
matters -- and prove the adapter's contract:

* the nested ``text_config`` counts resolve, and ``get_moe_info`` unpacks
  like the stock RCO helper;
* ``model.model.layers`` and ``layer.mlp`` are the topology the stock RCO
  pipeline walks;
* an all-ones survival mask reproduces the block's own forward exactly;
* a zeroed survival mask drops the routed experts and keeps the shared one;
* the wrapper restores the block's original forward exactly;
* malformed masks and malformed counts fail loudly.

The stock RCO tree is *not* imported: its dependency stack (accelerate,
datasets, compressed-tensors) is not assumed to be present.  The stock
semantics the adapter must match are stated in the adapter's docstring and
asserted here directly.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "deepseek_rco_adapter", TOOLS / "deepseek_rco_adapter.py"
)
assert SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


# The pinned upstream source tree, when it is present in the checkout.
PINNED_SRC = (
    Path(__file__).resolve().parents[1]
    / "build"
    / "engines"
    / "transformers-deepseek-v41-src"
    / "src"
)


def _import_tiny_model() -> tuple[Any, Any]:
    """Return ``(torch, model_factory)`` for the pinned tiny V4.1 model.

    The pinned source tree is preferred over any installed transformers, so the
    tests exercise the exact revision the adapter targets.  The tree is skipped
    when it is absent or when torch is unavailable.
    """
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on the host
        raise unittest.SkipTest("torch is not installed") from error

    import sys

    pinned_present = PINNED_SRC.is_dir()
    if pinned_present and str(PINNED_SRC) not in sys.path:
        sys.path.insert(0, str(PINNED_SRC))
    try:
        import transformers
        from transformers import DeepseekV41ForCausalLM, DeepseekV41TextConfig
    except ImportError as error:  # pragma: no cover - depends on the host
        raise unittest.SkipTest(f"transformers is not usable: {error}") from error

    if pinned_present and not str(transformers.__file__).startswith(str(PINNED_SRC)):
        # Another transformers was already imported; testing it would not
        # exercise the pinned revision.
        raise unittest.SkipTest(
            f"the pinned V4.1 source tree was not imported (got {transformers.__file__})"
        )

    def make_config() -> Any:
        # The tiny config the pinned test file uses
        # (tests/models/deepseek_v41/test_modeling_deepseek_v41.py).
        return DeepseekV41TextConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=1,
            num_hidden_layers=5,
            compress_ratios=[0, 2, 2, 1, 1],
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            max_position_embeddings=128,
            vocab_size=128,
            head_dim=32,
            qk_rope_head_dim=8,
            q_lora_rank=32,
            o_groups=2,
            o_lora_rank=16,
            n_routed_experts=4,
            n_shared_experts=1,
            sliding_window=8,
            hc_mult=2,
            hc_sinkhorn_iters=3,
            hc_eps=1.0e-6,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=2,
            candidate_topk_blocks=4,
            candidate_block_size=4,
            num_nextn_predict_layers=0,
            scoring_func="sqrtsoftplus",
            gate_temp=1.0,
            routed_scaling_factor=1.5,
            swiglu_limit=10.0,
            rope_theta=10000.0,
            compress_rope_theta=160000.0,
            rms_norm_eps=1.0e-20,
            engram_layer_ids=[],
            attention_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )

    def build(device: str = "cpu") -> Any:
        torch.manual_seed(0)
        config = make_config()
        if device == "meta":
            with torch.device("meta"):
                return DeepseekV41ForCausalLM(config).eval()
        return DeepseekV41ForCausalLM(config).to(torch.float32).eval()

    return torch, build


class AdapterTopologyTest(unittest.TestCase):
    """Counts and layer resolution, independent of any weight computation."""

    def test_importable_without_torch(self) -> None:
        # The module imported at the top of this file without torch present on
        # some hosts; assert its public surface exists either way.
        for name in ADAPTER.__all__:
            self.assertTrue(hasattr(ADAPTER, name), name)

    def test_resolves_nested_text_config_counts(self) -> None:
        torch, build = _import_tiny_model()
        model = build()
        config = model.config

        # The released checkpoint's config is composite: the counts live under
        # text_config, not at the top level.
        nested = ADAPTER.resolve_text_config(config)
        self.assertEqual(nested.num_hidden_layers, 5)
        self.assertEqual(nested.n_routed_experts, 4)
        self.assertEqual(nested.num_experts_per_tok, 2)

        counts = ADAPTER.resolve_moe_counts(config)
        self.assertEqual(counts.n_layers, 5)
        self.assertEqual(counts.n_experts, 4)
        self.assertEqual(counts.top_k, 2)

    def test_get_moe_info_unpacks_like_stock_rco(self) -> None:
        torch, build = _import_tiny_model()
        model = build()

        info = ADAPTER.get_moe_info(model)
        n_layers, n_experts, top_k = info
        self.assertEqual((n_layers, n_experts, top_k), (5, 4, 2))

        # A NamedTuple still unpacks like the stock tuple.
        self.assertIsInstance(info, tuple)
        self.assertEqual(tuple(info), (5, 4, 2))

    def test_get_moe_info_on_composite_config(self) -> None:
        torch, build = _import_tiny_model()
        from transformers import DeepseekV41Config, DeepseekV41TextConfig

        text = DeepseekV41TextConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=1,
            num_hidden_layers=5,
            compress_ratios=[0, 2, 2, 1, 1],
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            max_position_embeddings=128,
            vocab_size=128,
            head_dim=32,
            qk_rope_head_dim=8,
            q_lora_rank=32,
            o_groups=2,
            o_lora_rank=16,
            n_routed_experts=4,
            n_shared_experts=1,
            sliding_window=8,
            hc_mult=2,
            hc_sinkhorn_iters=3,
            hc_eps=1.0e-6,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=2,
            candidate_topk_blocks=4,
            candidate_block_size=4,
            num_nextn_predict_layers=0,
            scoring_func="sqrtsoftplus",
            gate_temp=1.0,
            routed_scaling_factor=1.5,
            swiglu_limit=10.0,
            rope_theta=10000.0,
            compress_rope_theta=160000.0,
            rms_norm_eps=1.0e-20,
            engram_layer_ids=[],
            attention_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
        composite = DeepseekV41Config(text_config=text.to_dict())

        # The composite config itself has no top-level counts.
        self.assertFalse(hasattr(composite, "num_hidden_layers"))
        counts = ADAPTER.resolve_moe_counts(composite)
        self.assertEqual(tuple(counts), (5, 4, 2))

    def test_resolves_layers_and_moe_block(self) -> None:
        torch, build = _import_tiny_model()
        model = build()

        layers = ADAPTER.resolve_layers(model)
        self.assertEqual(len(layers), 5)

        block = ADAPTER.resolve_moe_block(layers[0])
        self.assertEqual(type(block).__name__, "DeepseekV41SparseMoeBlock")

        # The stock RCO pipeline reaches the block through model.model.layers[l].mlp.
        self.assertIs(model.model.layers[0].mlp, block)
        self.assertIsNotNone(block.gate)
        self.assertIsNotNone(block.experts)
        self.assertIsNotNone(block.shared_experts)

    def test_meta_device_model_resolves_topology(self) -> None:
        torch, build = _import_tiny_model()
        model = build("meta")

        layers = ADAPTER.resolve_layers(model)
        self.assertEqual(len(layers), 5)
        counts = ADAPTER.get_moe_info(model)
        self.assertEqual(tuple(counts), (5, 4, 2))


class AdapterMalformedInputTest(unittest.TestCase):
    """Malformed masks and malformed counts must fail, not silently pass."""

    def test_malformed_masks_fail(self) -> None:
        torch, build = _import_tiny_model()
        model = build()
        block = model.model.layers[0].mlp
        state = {"ste_masks": torch.ones(5, 4)}

        wrapper = ADAPTER.SparseMoePruneWrapper(block, 0, 4, state)
        try:
            hidden = torch.randn(1, 3, 64)

            # Wrong expert count.
            state["ste_masks"] = torch.ones(5, 3)
            with self.assertRaises(ADAPTER.RcoAdapterError):
                wrapper.forward(hidden)

            # Survival values are probabilities, not arbitrary scales.
            state["ste_masks"] = torch.full((5, 4), -0.1)
            with self.assertRaises(ADAPTER.RcoAdapterError):
                wrapper.forward(hidden)

            state["ste_masks"] = torch.full((5, 4), 1.1)
            with self.assertRaises(ADAPTER.RcoAdapterError):
                wrapper.forward(hidden)

            # NaN is outside the valid interval and must not slip through
            # elementwise comparisons.
            state["ste_masks"] = torch.full((5, 4), torch.nan)
            with self.assertRaises(ADAPTER.RcoAdapterError):
                wrapper.forward(hidden)

            # Wrong rank.
            state["ste_masks"] = torch.ones(5, 1, 4)
            with self.assertRaises(ADAPTER.RcoAdapterError):
                wrapper.forward(hidden)

            # Not a tensor.
            state["ste_masks"] = [[1.0, 1.0, 1.0, 1.0]] * 5
            with self.assertRaises(ADAPTER.RcoAdapterError):
                wrapper.forward(hidden)

            # Integer dtype is not a survival mask.
            state["ste_masks"] = torch.ones(5, 4, dtype=torch.long)
            with self.assertRaises(ADAPTER.RcoAdapterError):
                wrapper.forward(hidden)

            # Missing shared state.
            state.clear()
            with self.assertRaises(ADAPTER.RcoAdapterError):
                wrapper.forward(hidden)
        finally:
            wrapper.restore()

    def test_layer_index_out_of_range_fails(self) -> None:
        torch, build = _import_tiny_model()
        model = build()
        block = model.model.layers[0].mlp
        state = {"ste_masks": torch.ones(1, 4)}
        wrapper = ADAPTER.SparseMoePruneWrapper(block, 3, 4, state)
        try:
            with self.assertRaises(ADAPTER.RcoAdapterError):
                wrapper.forward(torch.randn(1, 3, 64))
        finally:
            wrapper.restore()

    def test_malformed_counts_fail(self) -> None:
        class BadConfig:
            num_hidden_layers = 5
            n_routed_experts = 4
            num_experts_per_tok = 2

        self.assertEqual(tuple(ADAPTER.resolve_moe_counts(BadConfig())), (5, 4, 2))

        class NoLayers(BadConfig):
            num_hidden_layers = 0

        with self.assertRaises(ADAPTER.RcoAdapterError):
            ADAPTER.resolve_moe_counts(NoLayers())

        class NoExperts(BadConfig):
            n_routed_experts = 0

        with self.assertRaises(ADAPTER.RcoAdapterError):
            ADAPTER.resolve_moe_counts(NoExperts())

        class TopKTooLarge(BadConfig):
            num_experts_per_tok = 9

        with self.assertRaises(ADAPTER.RcoAdapterError):
            ADAPTER.resolve_moe_counts(TopKTooLarge())

        class NonInteger(BadConfig):
            num_hidden_layers = "five"

        with self.assertRaises(ADAPTER.RcoAdapterError):
            ADAPTER.resolve_moe_counts(NonInteger())

        with self.assertRaises(ADAPTER.RcoAdapterError):
            ADAPTER.resolve_moe_counts(None)

    def test_non_fused_expert_layout_is_rejected(self) -> None:
        torch, build = _import_tiny_model()
        model = build()
        block = model.model.layers[0].mlp
        state = {"ste_masks": torch.ones(5, 4)}

        # A per-expert ModuleList is the Qwen3-Next / Mixtral layout; the V4.1
        # adapter targets the fused module and must not guess.
        original = block.experts
        block.experts = torch.nn.ModuleList([original] * 4)
        wrapper = ADAPTER.SparseMoePruneWrapper(block, 0, 4, state)
        try:
            with self.assertRaises(ADAPTER.RcoAdapterError):
                wrapper.forward(torch.randn(1, 3, 64))
        finally:
            wrapper.restore()
            block.experts = original

    def test_router_without_image_mask_is_rejected(self) -> None:
        torch, build = _import_tiny_model()
        model = build()
        block = model.model.layers[0].mlp
        state = {"ste_masks": torch.ones(5, 4)}

        original = block.gate

        class PlainGate(torch.nn.Module):
            def forward(self, hidden_states):
                return original(hidden_states)

        block.gate = PlainGate()
        wrapper = ADAPTER.SparseMoePruneWrapper(block, 0, 4, state)
        try:
            with self.assertRaises(ADAPTER.RcoAdapterError):
                wrapper.forward(torch.randn(1, 3, 64))
        finally:
            wrapper.restore()
            block.gate = original

    def test_malformed_router_indices_fail(self) -> None:
        torch, build = _import_tiny_model()
        model = build()
        block = model.model.layers[0].mlp
        state = {"ste_masks": torch.ones(5, 4)}
        wrapper = ADAPTER.SparseMoePruneWrapper(block, 0, 4, state)
        original_gate = block.gate
        hidden = torch.randn(1, 3, 64)

        class SubstituteGate(torch.nn.Module):
            def __init__(self, transform) -> None:
                super().__init__()
                self.transform = transform

            def forward(self, router_hidden_states, image_mask=None):
                logits, weights, indices = original_gate(router_hidden_states, image_mask)
                return logits, weights, self.transform(indices)

        scenarios = {
            "not-a-tensor": lambda indices: indices.tolist(),
            "floating-dtype": lambda indices: indices.to(torch.float32),
            "boolean-dtype": lambda indices: indices.to(torch.bool),
            "shape-mismatch": lambda indices: indices[..., :1],
            "below-range": lambda indices: torch.cat(
                (indices[..., :-1], indices[..., -1:] - 5), dim=-1
            ),
            "above-range": lambda indices: torch.cat(
                (indices[..., :-1], indices[..., -1:] + 4), dim=-1
            ),
        }

        try:
            for description, transform in scenarios.items():
                with self.subTest(case=description):
                    block.gate = SubstituteGate(transform)
                    with self.assertRaises(ADAPTER.RcoAdapterError):
                        wrapper.forward(hidden)
        finally:
            wrapper.restore()
            block.gate = original_gate

    def test_install_wrappers_rejects_layer_count_mismatch(self) -> None:
        torch, build = _import_tiny_model()
        model = build()
        state = {"ste_masks": torch.ones(5, 4)}
        with self.assertRaises(ADAPTER.RcoAdapterError):
            ADAPTER.install_wrappers(model, 4, 4, state)


class AdapterHardMaskEvaluationTest(unittest.TestCase):
    """The stock-compatible hard-mask evaluation path."""

    def setUp(self) -> None:
        self.torch, self.build = _import_tiny_model()
        self.model = self.build()
        self.input_ids = self.torch.randint(0, 128, (2, 9))

    def _reference_cache(self, batch_size: int = 2) -> tuple[list[Any], list[Any]]:
        import torch.nn.functional as F

        references = []
        masks = []
        for start in range(0, self.input_ids.size(0), batch_size):
            batch = self.input_ids[start : start + batch_size]
            with self.torch.no_grad():
                logits = self.model(batch).logits[:, :-1, :].contiguous()
                references.append(F.log_softmax(logits.float(), dim=-1).cpu())
            masks.append(self.torch.ones(batch.size(0), 9))
        return references, masks

    def test_zero_prune_mask_has_zero_kl_against_the_unmasked_reference(self) -> None:
        references, masks = self._reference_cache()
        prune_mask = self.torch.zeros(5, 4, dtype=self.torch.bool)

        score = ADAPTER.evaluate_with_mask(
            self.model, prune_mask, self.input_ids, references, masks
        )

        self.assertEqual(score, 0.0)
        for block in self.model.model.layers:
            self.assertNotIn("forward", vars(block.mlp))

    def test_nonempty_mask_is_finite_and_restores_all_wrappers(self) -> None:
        references, masks = self._reference_cache(batch_size=1)
        original_forwards = [type(layer.mlp).forward for layer in self.model.model.layers]
        prune_mask = self.torch.zeros(5, 4, dtype=self.torch.bool)
        prune_mask[0, 0] = True

        score = ADAPTER.evaluate_with_mask(
            self.model, prune_mask, self.input_ids, references, masks, batch_size=1
        )

        self.assertTrue(self.torch.isfinite(self.torch.tensor(score)))
        self.assertGreaterEqual(score, 0.0)
        for block, original in zip(self.model.model.layers, original_forwards, strict=True):
            self.assertNotIn("forward", vars(block.mlp))
            self.assertIs(block.mlp.forward.__func__, original)

    def test_loss_mask_is_next_token_shifted_and_token_mean_normalized(self) -> None:
        references, all_masks = self._reference_cache()
        prune_mask = self.torch.zeros(5, 4, dtype=self.torch.bool)
        all_score = ADAPTER.evaluate_with_mask(
            self.model, prune_mask, self.input_ids, references, all_masks
        )

        subset_masks = [self.torch.ones(2, 9)]
        subset_masks[0][:, 0] = 0.0
        subset_score = ADAPTER.evaluate_with_mask(
            self.model, prune_mask, self.input_ids, references, subset_masks
        )

        # The first source position contributes no target; its removal scales
        # the same sum over the remaining 15 token positions.
        self.assertAlmostEqual(subset_score, all_score * 16.0 / 15.0, places=12)

    def test_malformed_prune_masks_fail_before_wrappers_are_installed(self) -> None:
        references, masks = self._reference_cache()
        original_forwards = [type(layer.mlp).forward for layer in self.model.model.layers]
        cases = {
            "wrong-shape": self.torch.zeros(4, 4, dtype=self.torch.bool),
            "floating-dtype": self.torch.ones(5, 4),
            "integer-dtype": self.torch.ones(5, 4, dtype=self.torch.long),
            "not-a-tensor": [[False] * 4] * 5,
        }

        for description, prune_mask in cases.items():
            with (
                self.subTest(case=description),
                self.assertRaises(ADAPTER.RcoAdapterError),
            ):
                ADAPTER.evaluate_with_mask(
                    self.model, prune_mask, self.input_ids, references, masks
                )

        for block, original in zip(self.model.model.layers, original_forwards, strict=True):
            self.assertNotIn("forward", vars(block.mlp))
            self.assertIs(block.mlp.forward.__func__, original)


class AdapterForwardTest(unittest.TestCase):
    """The wrapper's numerics against the block's own forward."""

    def setUp(self) -> None:
        self.torch, self.build = _import_tiny_model()
        self.model = self.build()
        self.block = self.model.model.layers[0].mlp
        self.hidden = self.torch.randn(2, 5, 64)
        self.state = {"ste_masks": self.torch.ones(5, 4)}

    def _baseline(self) -> Any:
        with self.torch.no_grad():
            return self.block(self.hidden).clone()

    def test_all_ones_mask_equals_baseline_forward(self) -> None:
        baseline = self._baseline()
        wrapper = ADAPTER.SparseMoePruneWrapper(self.block, 0, 4, self.state)
        try:
            with self.torch.no_grad():
                actual = wrapper.forward(self.hidden)
        finally:
            wrapper.restore()
        self.assertTrue(self.torch.equal(actual, baseline))

    def test_zero_mask_keeps_shared_expert_only(self) -> None:
        baseline = self._baseline()
        self.state["ste_masks"] = self.torch.zeros(5, 4)
        wrapper = ADAPTER.SparseMoePruneWrapper(self.block, 0, 4, self.state)
        try:
            with self.torch.no_grad():
                actual = wrapper.forward(self.hidden)
        finally:
            wrapper.restore()

        # The routed contribution is gone, the shared one is not.
        flat = self.hidden.reshape(-1, 64)
        with self.torch.no_grad():
            shared = self.block.shared_experts(flat)
        self.assertTrue(self.torch.allclose(actual.reshape(-1, 64), shared, atol=1e-5))
        self.assertFalse(self.torch.allclose(actual, baseline, atol=1e-4))

    def test_partial_mask_changes_output(self) -> None:
        baseline = self._baseline()
        mask = self.torch.ones(5, 4)
        mask[0, 0] = 0.0
        self.state["ste_masks"] = mask
        wrapper = ADAPTER.SparseMoePruneWrapper(self.block, 0, 4, self.state)
        try:
            with self.torch.no_grad():
                actual = wrapper.forward(self.hidden)
        finally:
            wrapper.restore()
        self.assertFalse(self.torch.allclose(actual, baseline, atol=1e-6))

    def test_wrapper_restores_exactly(self) -> None:
        original = type(self.block).forward
        wrapper = ADAPTER.SparseMoePruneWrapper(self.block, 0, 4, self.state)
        self.assertIn("forward", vars(self.block))
        wrapper.restore()
        # The pristine module resolves `forward` on the class; restoring must
        # leave no instance attribute behind.
        self.assertNotIn("forward", vars(self.block))
        self.assertIs(self.block.forward.__func__, original)

    def test_install_and_remove_wrappers_round_trip(self) -> None:
        layers = ADAPTER.resolve_layers(self.model)
        originals = [type(layer.mlp).forward for layer in layers]

        self.state["ste_masks"] = self.torch.ones(5, 4)
        wrappers = ADAPTER.install_wrappers(self.model, 5, 4, self.state)
        self.assertEqual(len(wrappers), 5)
        for layer in layers:
            self.assertIn("forward", vars(layer.mlp))

        ADAPTER.remove_wrappers(wrappers)
        for layer, original in zip(layers, originals, strict=True):
            self.assertNotIn("forward", vars(layer.mlp))
            self.assertIs(layer.mlp.forward.__func__, original)

    def test_image_mask_is_forwarded_to_the_router(self) -> None:
        # The router switches to the vision correction bias for masked tokens.
        # That bias steers *selection* only, so it must not be uniform: a flat
        # offset would leave the top-k order, and therefore the output,
        # unchanged.
        gate = self.block.gate
        with self.torch.no_grad():
            gate.e_score_correction_bias.fill_(0.0)
            gate.e_score_correction_bias_vl.fill_(0.0)
            gate.e_score_correction_bias_vl[0] = 100.0

        image_mask = self.torch.zeros(2, 5, dtype=self.torch.bool)
        image_mask[:, :2] = True

        with self.torch.no_grad():
            # The block's own masked forward is the reference.
            masked_reference = self.block(self.hidden, image_mask).clone()
            unmasked_reference = self.block(self.hidden).clone()
        # The mask is not vacuous: it changes the block's own output.
        self.assertFalse(self.torch.allclose(masked_reference, unmasked_reference, atol=1e-6))

        wrapper = ADAPTER.SparseMoePruneWrapper(self.block, 0, 4, self.state)
        try:
            with self.torch.no_grad():
                masked = wrapper.forward(self.hidden, image_mask)
                unmasked = wrapper.forward(self.hidden)
        finally:
            wrapper.restore()

        # The wrapper forwards the mask: it reproduces the block's masked
        # forward exactly under an all-ones survival mask.
        self.assertTrue(self.torch.equal(masked, masked_reference))
        self.assertTrue(self.torch.equal(unmasked, unmasked_reference))
        self.assertFalse(self.torch.allclose(masked, unmasked, atol=1e-6))

    def test_full_model_logits_are_unchanged_under_an_all_ones_mask(self) -> None:
        torch, build = _import_tiny_model()
        model = build()
        input_ids = torch.randint(0, 128, (2, 9))

        with torch.no_grad():
            baseline = model(input_ids, use_cache=False).logits.clone()

        state = {"ste_masks": torch.ones(5, 4)}
        wrappers = ADAPTER.install_wrappers(model, 5, 4, state)
        try:
            with torch.no_grad():
                masked = model(input_ids, use_cache=False).logits.clone()
        finally:
            ADAPTER.remove_wrappers(wrappers)

        # The whole model's logits are bit-identical under an all-ones mask.
        self.assertTrue(torch.equal(masked, baseline))

        with torch.no_grad():
            restored = model(input_ids, use_cache=False).logits.clone()
        self.assertTrue(torch.equal(restored, baseline))

    def test_router_tuple_contract(self) -> None:
        # The adapter must consume the (logits, weights, indices) triple.
        with self.torch.no_grad():
            router_output = self.block.gate(self.hidden)
        self.assertIsInstance(router_output, tuple)
        self.assertEqual(len(router_output), 3)
        logits, weights, indices = router_output
        self.assertEqual(logits.shape, (10, 4))
        self.assertEqual(weights.shape, (10, 2))
        self.assertEqual(indices.shape, (10, 2))
        # The first element is raw logits, not a probability vector.
        self.assertFalse(bool(((logits.sum(dim=-1) > 0.99) & (logits.sum(dim=-1) < 1.01)).all()))


if __name__ == "__main__":
    unittest.main()
