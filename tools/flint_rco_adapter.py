#!/usr/bin/env python3
"""Flint topology adapter for stock RCO expert pruning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

__all__ = [
    "MoeCounts",
    "RcoAdapterError",
    "SparseMoePruneWrapper",
    "compute_frequency",
    "compute_router_scores",
    "evaluate_with_mask",
    "get_moe_info",
    "install_wrappers",
    "physical_keep_rows",
    "remove_wrappers",
    "resolve_layer_ids",
    "resolve_moe_counts",
]


class RcoAdapterError(RuntimeError):
    """Raised when a model does not match the Flint topology."""


class MoeCounts(NamedTuple):
    n_layers: int
    n_experts: int
    top_k: int


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RcoAdapterError("torch is required for Flint RCO wrappers") from error
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RcoAdapterError(f"{name} must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RcoAdapterError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _integer_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RcoAdapterError(f"{name} must be a list of integers")
    result = []
    for item in value:
        result.append(_nonnegative_int(item, f"{name} entry"))
    return result


def _layer_count(config: Any) -> int:
    value = getattr(config, "num_hidden_layers", None)
    return _positive_int(value, "num_hidden_layers")


def resolve_layer_ids(config: Any) -> tuple[int, ...]:
    """Return physical decoder-layer indices that contain routed experts."""
    total = _layer_count(config)
    dense = set(_integer_list(getattr(config, "mlp_only_layers", None) or [], "mlp_only_layers"))
    if any(index >= total for index in dense):
        raise RcoAdapterError("mlp_only_layers contains an out-of-range layer")
    routed = tuple(index for index in range(total) if index not in dense)
    if not routed:
        raise RcoAdapterError("Flint config declares no routed-expert layers")
    return routed


def resolve_moe_counts(config: Any) -> MoeCounts:
    routed_layers = len(resolve_layer_ids(config))
    experts = _positive_int(getattr(config, "num_experts", None), "num_experts")
    top_k = _positive_int(getattr(config, "num_experts_per_tok", None), "num_experts_per_tok")
    if top_k > experts:
        raise RcoAdapterError("num_experts_per_tok exceeds num_experts")
    return MoeCounts(routed_layers, experts, top_k)


def get_moe_info(model: Any) -> tuple[int, int, int]:
    return resolve_moe_counts(model.config)


def _model_layers(model: Any) -> Any:
    try:
        return model.model.layers
    except AttributeError as error:
        raise RcoAdapterError("model.model.layers is missing") from error


def _moe_block(model: Any, logical_layer: int, layer_ids: Sequence[int]) -> Any:
    try:
        physical_layer = layer_ids[logical_layer]
        block = _model_layers(model)[physical_layer].mlp
    except (IndexError, AttributeError, TypeError) as error:
        raise RcoAdapterError(
            f"could not resolve logical Flint MoE layer {logical_layer}: {error}"
        ) from error
    for required in ("gate", "experts", "shared_expert"):
        if not hasattr(block, required):
            raise RcoAdapterError(f"Flint MoE block lacks {required}")
    return block


def _router_scores(block: Any, hidden: Any, prune_mask: Any = None) -> Any:
    torch = _torch()
    logits = block.gate(hidden).float()
    scores = torch.sigmoid(logits)
    if prune_mask is not None:
        if prune_mask.device != scores.device:
            prune_mask = prune_mask.to(scores.device)
        scores = scores.masked_fill(prune_mask, 0.0)
    return scores


def _select(block: Any, hidden: Any, top_k: int, prune_mask: Any = None) -> tuple[Any, Any]:
    scores = _router_scores(block, hidden, prune_mask)
    weights, indices = torch_topk(scores, top_k)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, indices


def torch_topk(values: Any, k: int) -> tuple[Any, Any]:
    torch = _torch()
    return torch.topk(values, k, dim=-1)


def _call_experts(block: Any, hidden: Any, indices: Any, weights: Any) -> Any:
    return block.experts(hidden, indices, weights)


class SparseMoePruneWrapper:
    """Patch a Flint MoE block with RCO's survival-mask semantics."""

    def __init__(
        self,
        block: Any,
        logical_layer: int,
        n_experts: int,
        shared_state: dict[str, Any],
        top_k: int,
    ):
        self.block = block
        self.logical_layer = logical_layer
        self.n_experts = n_experts
        self.top_k = top_k
        self.shared = shared_state
        self._orig_forward = block.forward

        wrapper = self

        def patched_forward(hidden_states: Any) -> Any:
            return wrapper._forward(hidden_states)

        block.forward = patched_forward

    def _forward(self, hidden_states: Any) -> Any:
        masks = self.shared.get("ste_masks")
        if masks is None:
            raise RcoAdapterError("shared state lacks ste_masks")
        layer_mask = masks[self.logical_layer]
        if layer_mask.shape != (self.n_experts,):
            raise RcoAdapterError(
                f"layer {self.logical_layer} mask shape {tuple(layer_mask.shape)} "
                f"does not match ({self.n_experts},)"
            )
        if layer_mask.device != hidden_states.device:
            layer_mask = layer_mask.to(hidden_states.device)

        weights, indices = _select(self.block, hidden_states, self.top_k)
        weights = weights * layer_mask[indices]
        routed = _call_experts(self.block, hidden_states, indices, weights.to(hidden_states.dtype))
        return routed + self.block.shared_expert(hidden_states)

    def restore(self) -> None:
        self.block.forward = self._orig_forward


def install_wrappers(
    model: Any,
    n_layers: int,
    n_experts: int,
    shared_state: dict[str, Any],
) -> list[SparseMoePruneWrapper]:
    counts = resolve_moe_counts(model.config)
    if (n_layers, n_experts) != (counts.n_layers, counts.n_experts):
        raise RcoAdapterError(f"RCO geometry {(n_layers, n_experts)} does not match Flint {counts}")
    layer_ids = resolve_layer_ids(model.config)
    return [
        SparseMoePruneWrapper(
            _moe_block(model, logical, layer_ids),
            logical,
            n_experts,
            shared_state,
            counts.top_k,
        )
        for logical in range(n_layers)
    ]


def remove_wrappers(wrappers: Sequence[SparseMoePruneWrapper]) -> None:
    for wrapper in wrappers:
        wrapper.restore()


class _HardMaskWrapper(SparseMoePruneWrapper):
    def __init__(self, block: Any, logical_layer: int, prune_mask: Any, top_k: int):
        super().__init__(
            block,
            logical_layer,
            prune_mask.numel(),
            {"ste_masks": None},
            top_k,
        )
        self.prune_mask = prune_mask

    def _forward(self, hidden_states: Any) -> Any:
        prune_mask = self.prune_mask
        if prune_mask.device != hidden_states.device:
            prune_mask = prune_mask.to(hidden_states.device)
        weights, indices = _select(self.block, hidden_states, self.top_k, prune_mask)
        routed = _call_experts(self.block, hidden_states, indices, weights.to(hidden_states.dtype))
        return routed + self.block.shared_expert(hidden_states)


def evaluate_with_mask(
    model: Any,
    prune_mask: Any,
    cal_data: Any,
    ref_log_probs: Sequence[Any],
    ref_masks: Sequence[Any],
    batch_size: int = 4,
) -> float:
    torch = _torch()
    gradients_were_enabled = torch.is_grad_enabled()
    if not ref_log_probs or len(ref_log_probs) != len(ref_masks):
        raise RcoAdapterError("reference cache and masks must be non-empty and aligned")
    counts = resolve_moe_counts(model.config)
    if tuple(prune_mask.shape) != (counts.n_layers, counts.n_experts):
        raise RcoAdapterError(f"prune mask shape {tuple(prune_mask.shape)} does not match {counts}")
    layer_ids = resolve_layer_ids(model.config)
    wrappers = [
        _HardMaskWrapper(
            _moe_block(model, logical, layer_ids),
            logical,
            prune_mask[logical].to(_model_device(model)),
            counts.top_k,
        )
        for logical in range(counts.n_layers)
    ]
    total_kl = 0.0
    total_tokens = 0.0
    try:
        torch.set_grad_enabled(False)
        for start in range(0, cal_data.size(0), batch_size):
            batch = cal_data[start : start + batch_size].to(_model_device(model))
            index = min(start // batch_size, len(ref_log_probs) - 1)
            ref_lp = ref_log_probs[index].to(_model_device(model))
            token_mask = ref_masks[index][:, 1:].to(_model_device(model)).float()
            output = model(input_ids=batch)
            logits = getattr(output, "logits", output)[:, :-1, :].float()
            q_lp = torch.log_softmax(logits, dim=-1)
            p = ref_lp.float().exp()
            kl = (p * (ref_lp.float() - q_lp)).sum(dim=-1)
            total_kl += (kl * token_mask).sum().item()
            total_tokens += token_mask.sum().item()
    finally:
        torch.set_grad_enabled(gradients_were_enabled)
        remove_wrappers(wrappers)
    return total_kl / max(total_tokens, 1.0)


def _model_device(model: Any) -> Any:
    try:
        return next(model.get_input_embeddings().parameters()).device
    except (StopIteration, AttributeError):
        return next(model.parameters()).device


def compute_frequency(
    model: Any,
    data: Any,
    n_layers: int,
    n_experts: int,
    top_k: int,
    batch_size: int = 4,
) -> Any:
    torch = _torch()
    gradients_were_enabled = torch.is_grad_enabled()
    counts = resolve_moe_counts(model.config)
    if (n_layers, n_experts, top_k) != counts:
        raise RcoAdapterError(f"RCO geometry {(n_layers, n_experts, top_k)} != {counts}")
    frequency = torch.zeros(n_layers, n_experts, dtype=torch.float32)
    layer_ids = resolve_layer_ids(model.config)
    handles = []
    try:
        torch.set_grad_enabled(False)
        for logical, physical in enumerate(layer_ids):
            gate = _model_layers(model)[physical].mlp.gate

            def make_hook(logical_layer: int):
                def hook(_module: Any, args: tuple[Any, ...], _output: Any) -> None:
                    gate_scores = torch.sigmoid(args[0].float())
                    _, indices = torch.topk(gate_scores, top_k, dim=-1)
                    frequency[logical_layer] += (
                        torch.bincount(indices.reshape(-1), minlength=n_experts).float().cpu()
                    )

                return hook

            handles.append(gate.register_forward_pre_hook(make_hook(logical)))
        for start in range(0, data.size(0), batch_size):
            batch = data[start : start + batch_size].to(_model_device(model))
            model(input_ids=batch)
    finally:
        for handle in handles:
            handle.remove()
        torch.set_grad_enabled(gradients_were_enabled)
    return frequency


def compute_router_scores(
    model: Any,
    data: Any,
    n_layers: int,
    n_experts: int,
    batch_size: int = 4,
) -> Any:
    torch = _torch()
    gradients_were_enabled = torch.is_grad_enabled()
    counts = resolve_moe_counts(model.config)
    if (n_layers, n_experts) != (counts.n_layers, counts.n_experts):
        raise RcoAdapterError(f"RCO geometry {(n_layers, n_experts)} != {counts}")
    scores = torch.zeros(n_layers, n_experts, dtype=torch.float32)
    layer_ids = resolve_layer_ids(model.config)
    handles = []
    try:
        torch.set_grad_enabled(False)
        for logical, physical in enumerate(layer_ids):
            gate = _model_layers(model)[physical].mlp.gate

            def make_score_hook(logical_layer: int):
                def hook(_module: Any, args: tuple[Any, ...], _output: Any) -> None:
                    gate_scores = torch.sigmoid(args[0].float())
                    scores[logical_layer] += gate_scores.sum(dim=0).cpu()

                return hook

            handles.append(gate.register_forward_pre_hook(make_score_hook(logical)))
        for start in range(0, data.size(0), batch_size):
            batch = data[start : start + batch_size].to(_model_device(model))
            model(input_ids=batch)
    finally:
        for handle in handles:
            handle.remove()
        torch.set_grad_enabled(gradients_were_enabled)
    return scores


def physical_keep_rows(prune_mask: Any) -> list[list[int]]:
    _torch()
    survival = prune_mask.eq(False)
    return [row.nonzero(as_tuple=False).flatten().tolist() for row in survival]
