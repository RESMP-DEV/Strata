#!/usr/bin/env python3
"""DeepSeek-V4.1 topology adapter for the stock RCO expert-pruning pipeline.

The stock RCO tree (``~/.local/share/strata-tools/RCO``) prunes experts on
Qwen3-Next / OLMoE style MoE blocks.  DeepSeek-V4.1's
``DeepseekV41SparseMoeBlock`` breaks four of that pipeline's assumptions, and
every one of them silently changes the pruned model's numbers instead of
raising:

1. The router returns a ``(logits, weights, indices)`` triple whose first
   element is raw logits, not a full softmax.  The stock mask hook treats
   ``output[0]`` as a probability vector and rewrites it in place.
2. The router takes an ``image_mask`` argument that switches a token to the
   vision correction bias (``e_score_correction_bias_vl``).  The stock wrapper
   calls ``block.gate(h)`` and drops the mask, so image tokens route as text.
3. The shared expert is ``shared_experts`` (plural) and is *added* unscaled.
   The stock wrapper looks for ``shared_expert`` and, finding none, drops the
   shared contribution entirely.
4. The routed experts are one fused module called as
   ``experts(hidden, indices, weights)``.

This module keeps the stock RCO contract -- the same ``get_moe_info`` shape,
the same wrapper constructor arguments, the same
``shared_state['ste_masks']`` convention -- while implementing V4.1's block
faithfully.  It has no import-time dependency on torch: the module imports
cleanly in a bare interpreter and only needs torch once a wrapper runs.

Nothing here touches a checkpoint: the adapter only reads a loaded model's
config and module tree.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, NamedTuple

__all__ = [
    "MoeCounts",
    "RcoAdapterError",
    "SparseMoePruneWrapper",
    "evaluate_with_mask",
    "get_moe_info",
    "install_wrappers",
    "remove_wrappers",
    "resolve_layers",
    "resolve_moe_block",
    "resolve_moe_counts",
    "resolve_text_config",
]


class RcoAdapterError(RuntimeError):
    """Raised when a model or a mask does not match the DeepSeek-V4.1 topology."""


class MoeCounts(NamedTuple):
    """MoE shape of a DeepSeek-V4.1 model.

    A plain tuple, so it unpacks exactly like the stock RCO
    ``get_moe_info`` result: ``n_layers, n_experts, top_k = get_moe_info(model)``.
    """

    n_layers: int
    n_experts: int
    top_k: int


# Config keys that carry the routed-expert count, most specific first.  V4.1
# spells it ``n_routed_experts`` and maps ``num_local_experts`` onto it; the
# stock RCO pipeline reads ``num_experts`` (the Qwen3-Next / OLMoE spelling).
_EXPERT_COUNT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")


def _torch() -> Any:
    """Import torch on demand so the module stays importable without it."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on the host
        raise RcoAdapterError("torch is required for this operation") from error
    return torch


def _first_attr(source: Any, names: tuple[str, ...]) -> Any:
    """First present, non-None attribute (or mapping key) among ``names``."""
    for name in names:
        if isinstance(source, Mapping):
            if name in source:
                return source[name]
            continue
        value = getattr(source, name, None)
        if value is not None:
            return value
    return None


def _positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a positive integer count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise RcoAdapterError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise RcoAdapterError(f"{name} must be positive, got {value}")
    return value


def resolve_text_config(config: Any) -> Any:
    """Return the config object that actually carries the V4.1 MoE counts.

    The released checkpoint ships a composite ``DeepseekV41Config`` whose
    counts live under ``text_config``; a bare ``DeepseekV41TextConfig`` carries
    them at the top level.  Both are accepted, as is a plain mapping.
    """
    if config is None:
        raise RcoAdapterError("a config is required")
    nested = getattr(config, "text_config", None)
    if nested is not None:
        return nested
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            text = getter()
        except Exception:  # pragma: no cover - defensive, transformers internals
            text = None
        if text is not None:
            return text
    return config


def resolve_moe_counts(config: Any) -> MoeCounts:
    """Resolve ``(n_layers, n_experts, top_k)`` from a nested or flat config."""
    text = resolve_text_config(config)
    n_layers = _positive_int(_first_attr(text, ("num_hidden_layers",)), "num_hidden_layers")
    n_experts = _positive_int(
        _first_attr(text, _EXPERT_COUNT_KEYS), "num_experts (n_routed_experts)"
    )
    top_k = _positive_int(_first_attr(text, ("num_experts_per_tok",)), "num_experts_per_tok")
    if top_k > n_experts:
        raise RcoAdapterError(
            f"num_experts_per_tok ({top_k}) exceeds the routed expert count ({n_experts})"
        )
    return MoeCounts(n_layers, n_experts, top_k)


def resolve_layers(model: Any) -> Any:
    """Return the decoder-layer list of a V4.1 model.

    ``DeepseekV41ForCausalLM`` keeps the backbone at ``model.model.layers``;
    the bare ``DeepseekV41TextModel`` keeps it at ``model.layers``.  The
    released checkpoint is loaded through the causal-LM wrapper, so
    ``model.model.layers`` is the canonical path, but both are accepted.
    """
    if model is None:
        raise RcoAdapterError("a model is required")
    candidates = []
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(inner)
    candidates.append(model)
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None and len(layers) > 0:
            return layers
    raise RcoAdapterError(
        "could not find decoder layers: expected model.model.layers or model.layers"
    )


def resolve_moe_block(layer: Any) -> Any:
    """Return the MoE block of a V4.1 decoder layer."""
    block = getattr(layer, "mlp", None)
    if block is None:
        raise RcoAdapterError("decoder layer has no 'mlp' attribute")
    return block


def _ste_masks_for_layer(shared_state: Mapping[str, Any], layer_idx: int) -> Any:
    """Read the per-layer STE survival mask out of RCO's shared state."""
    if not isinstance(shared_state, Mapping):
        raise RcoAdapterError("shared_state must be a mapping")
    masks = shared_state.get("ste_masks")
    if masks is None:
        raise RcoAdapterError("shared_state['ste_masks'] is not set")
    if len(masks) <= layer_idx:
        raise RcoAdapterError(f"ste_masks has {len(masks)} layers, cannot index layer {layer_idx}")
    return masks[layer_idx]


def _check_survival_mask(mask: Any, n_experts: int) -> Any:
    """Validate a per-expert survival mask against the layer's expert count."""
    torch = _torch()
    if not isinstance(mask, torch.Tensor):
        raise RcoAdapterError(f"expert mask must be a tensor, got {type(mask).__name__}")
    if mask.ndim != 1:
        raise RcoAdapterError(f"expert mask must be 1-D, got shape {tuple(mask.shape)}")
    if mask.shape[0] != n_experts:
        raise RcoAdapterError(f"expert mask has {mask.shape[0]} entries, expected {n_experts}")
    if mask.is_complex():
        raise RcoAdapterError("expert mask must be real-valued")
    if not mask.is_floating_point() and mask.dtype != torch.bool:
        raise RcoAdapterError(f"expert mask must be floating point, got {mask.dtype}")
    if mask.dtype != torch.bool:
        non_finite = (~torch.isfinite(mask)).any()
        if non_finite:
            raise RcoAdapterError("expert mask values must be finite")
        minimum, maximum = mask.aminmax()
        out_of_range = (minimum < 0) | (maximum > 1)
        if out_of_range:
            raise RcoAdapterError("expert mask values must be in [0, 1]")
    return mask


def _check_router_indices(selected: Any, routing_weights: Any, n_experts: int) -> None:
    """Validate router indices before using them to index the survival mask."""
    torch = _torch()
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }
    if not isinstance(selected, torch.Tensor):
        raise RcoAdapterError(f"router indices must be a tensor, got {type(selected).__name__}")
    if selected.dtype not in integer_dtypes:
        raise RcoAdapterError(f"router indices must have an integer dtype, got {selected.dtype}")
    if selected.shape != routing_weights.shape:
        raise RcoAdapterError(
            f"router weights {tuple(routing_weights.shape)} and indices "
            f"{tuple(selected.shape)} disagree"
        )
    # Use tensor reductions and bitwise boolean operators: never convert the
    # whole index tensor to Python or use ``and``/``or`` between tensors.
    out_of_range = (selected < 0) | (selected >= n_experts)
    if out_of_range.any():
        raise RcoAdapterError(
            f"router indices must be in [0, {n_experts}), got out-of-range indices"
        )


def _require_fused_experts(block: Any) -> None:
    """Fail loudly when the block's experts are not the fused V4.1 module.

    V4.1 calls its routed experts as ``experts(hidden, indices, weights)``.
    A stack of per-expert modules (the Qwen3-Next / Mixtral layout the stock
    RCO wrapper also supports) would silently produce different numbers, so it
    is rejected instead of guessed at.
    """
    torch = _torch()
    experts = getattr(block, "experts", None)
    if experts is None:
        raise RcoAdapterError("the MoE block has no 'experts' attribute")
    if isinstance(experts, (list, tuple, torch.nn.ModuleList)):
        raise RcoAdapterError(
            "expected the fused V4.1 experts module, got a list of expert modules"
        )


def _require_image_mask_capable(block: Any) -> None:
    """Fail loudly when the router cannot take an ``image_mask``.

    The V4.1 router switches masked (image-span) tokens to the vision
    correction bias.  A router without that argument would route image tokens
    as text, which is the defect this adapter exists to prevent.
    """
    gate = getattr(block, "gate", None)
    if gate is None:
        raise RcoAdapterError("the MoE block has no 'gate' attribute")
    try:
        parameters = inspect.signature(gate.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return
    if "image_mask" not in parameters:
        raise RcoAdapterError(
            f"{type(gate).__name__}.forward does not accept an image_mask; "
            "this adapter targets the DeepSeek-V4.1 router"
        )


class SparseMoePruneWrapper:
    """Patch a ``DeepseekV41SparseMoeBlock`` so RCO's STE mask scales its experts.

    The wrapper keeps the block's real semantics:

    * the router is called with its ``image_mask`` argument, so image-span
      tokens still select against ``e_score_correction_bias_vl``;
    * the router's ``(logits, weights, indices)`` triple is consumed as-is:
      ``weights`` are the top-k routing weights (already scaled by
      ``routed_scaling_factor``) and ``indices`` the selected expert ids;
    * the routed experts are called through the fused interface
      ``experts(hidden, indices, weights)``;
    * the shared expert is added unscaled, exactly as the block does.

    The only difference from the block's own forward is that each selected
    expert's routing weight is multiplied by that expert's STE survival
    value.  A survival value of 1.0 reproduces the block exactly; 0.0 drops
    the expert's contribution.  The survival values are *not* renormalized:
    renormalizing would break the straight-through gradient RCO's search
    relies on.
    """

    def __init__(
        self,
        moe_block: Any,
        layer_idx: int,
        n_experts: int,
        shared_state: Mapping[str, Any],
    ) -> None:
        self.block = moe_block
        self.layer_idx = int(layer_idx)
        self.n_experts = _positive_int(n_experts, "n_experts")
        self.shared = shared_state
        self._orig_forward = moe_block.forward
        # Whether `forward` was an instance attribute before the patch.  A
        # pristine nn.Module resolves `forward` on the class, so restoring
        # means deleting the instance attribute, not rebinding it.
        self._had_own_forward = "forward" in vars(moe_block)
        self._wrapper_ref = self
        self.block.forward = self._patched_forward

    # -- lifecycle ---------------------------------------------------------------

    def _patched_forward(self, hidden_states: Any, image_mask: Any = None) -> Any:
        return self._wrapper_ref.forward(hidden_states, image_mask)

    def restore(self) -> None:
        """Put the block's original forward back, exactly as it was."""
        if self._had_own_forward:
            self.block.forward = self._orig_forward
        else:
            self.block.__dict__.pop("forward", None)

    # -- forward -----------------------------------------------------------------

    def forward(self, hidden_states: Any, image_mask: Any = None) -> Any:
        """V4.1 MoE forward with RCO's per-expert survival mask applied."""
        torch = _torch()
        block = self.block

        mask = _check_survival_mask(
            _ste_masks_for_layer(self.shared, self.layer_idx), self.n_experts
        )
        _require_fused_experts(block)
        _require_image_mask_capable(block)

        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])

        router_output = block.gate(hidden_states, image_mask)
        if not isinstance(router_output, (tuple, list)) or len(router_output) != 3:
            raise RcoAdapterError(
                "expected the V4.1 router to return (logits, weights, indices), "
                f"got {type(router_output).__name__}"
            )
        _, routing_weights, selected = router_output
        _check_router_indices(selected, routing_weights, self.n_experts)

        if mask.device != flat.device:
            mask = mask.to(flat.device)
        survival = mask.to(dtype=torch.float32)
        ste_weights = routing_weights.float() * survival[selected]
        ste_weights = ste_weights.to(routing_weights.dtype)

        # The block casts the routed output to fp32, adds the shared expert
        # unscaled, and casts the sum back to the hidden dtype exactly once.
        # Mirror that, so an all-ones mask reproduces the block bit for bit.
        routed = block.experts(flat, selected, ste_weights).float()

        shared = getattr(block, "shared_experts", None)
        if shared is None:
            shared = getattr(block, "shared_expert", None)
        if shared is not None:
            routed = routed + shared(flat).float()

        return routed.to(hidden_states.dtype).view(shape)

    # -- diagnostics -------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Snapshot of what this wrapper is bound to, for receipts and logs."""
        return {
            "layer_idx": self.layer_idx,
            "n_experts": self.n_experts,
            "block": type(self.block).__name__,
            "gate": type(getattr(self.block, "gate", None)).__name__,
            "experts": type(getattr(self.block, "experts", None)).__name__,
            "shared_experts": type(getattr(self.block, "shared_experts", None)).__name__,
        }


def install_wrappers(
    model: Any,
    n_layers: int,
    n_experts: int,
    shared_state: Mapping[str, Any],
) -> list[SparseMoePruneWrapper]:
    """Wrap every decoder layer's MoE block, in layer order."""
    layers = resolve_layers(model)
    if n_layers != len(layers):
        raise RcoAdapterError(
            f"n_layers ({n_layers}) does not match the model's layer count ({len(layers)})"
        )
    wrappers = []
    for layer_idx in range(n_layers):
        block = resolve_moe_block(layers[layer_idx])
        wrappers.append(SparseMoePruneWrapper(block, layer_idx, n_experts, shared_state))
    return wrappers


def remove_wrappers(wrappers: list[SparseMoePruneWrapper]) -> None:
    """Restore every wrapped block's original forward."""
    for wrapper in wrappers:
        wrapper.restore()


def evaluate_with_mask(
    model: Any,
    prune_mask: Any,
    cal_data: Any,
    ref_log_probs: Any,
    ref_masks: Any,
    batch_size: int = 4,
) -> float:
    """Evaluate full-vocabulary KL(reference || hard-masked model).
    ``prune_mask`` uses RCO's discrete convention: ``True`` is pruned and
    ``False`` is kept.  It is converted once to the adapter's survival-mask
    convention and applied without renormalizing the surviving routing
    weights.
    """
    torch = _torch()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise RcoAdapterError(f"batch_size must be a positive integer, got {batch_size!r}")

    n_layers, n_experts, _ = get_moe_info(model)
    if not isinstance(prune_mask, torch.Tensor):
        raise RcoAdapterError(f"prune_mask must be a tensor, got {type(prune_mask).__name__}")
    if prune_mask.dtype != torch.bool:
        raise RcoAdapterError(f"prune_mask must have dtype bool, got {prune_mask.dtype}")
    if prune_mask.shape != (n_layers, n_experts):
        raise RcoAdapterError(
            f"prune_mask has shape {tuple(prune_mask.shape)}, expected {(n_layers, n_experts)}"
        )
    if not isinstance(cal_data, torch.Tensor) or cal_data.ndim != 2:
        raise RcoAdapterError("cal_data must be an integer token tensor with shape (B, T)")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }
    if cal_data.is_complex() or cal_data.dtype not in integer_dtypes:
        raise RcoAdapterError(f"cal_data must have an integer dtype, got {cal_data.dtype}")
    if len(ref_log_probs) != len(ref_masks):
        raise RcoAdapterError(
            f"ref_log_probs has {len(ref_log_probs)} batches but ref_masks has {len(ref_masks)}"
        )

    survival_mask = ~prune_mask
    shared_state = {"ste_masks": survival_mask}
    device = getattr(model, "device", None)
    if device is None:
        parameters = getattr(model, "parameters", None)
        device = next(parameters()).device if callable(parameters) else None
    if device is None:
        raise RcoAdapterError("could not resolve the model's input device")

    wrappers = install_wrappers(model, n_layers, n_experts, shared_state)
    total_kl = 0.0
    total_tokens = 0.0
    with torch.no_grad():
        try:
            for start in range(0, cal_data.size(0), batch_size):
                batch = cal_data[start : start + batch_size].to(device)
                batch_index = min(start // batch_size, len(ref_log_probs) - 1)
                reference = ref_log_probs[batch_index].to(device)
                loss_mask = ref_masks[batch_index].to(device)

                if not isinstance(reference, torch.Tensor) or reference.ndim != 3:
                    raise RcoAdapterError(
                        "each ref_log_probs entry must be a tensor with shape (B, T-1, V)"
                    )
                if not reference.is_floating_point() or reference.is_complex():
                    raise RcoAdapterError(
                        f"ref_log_probs must have a real floating dtype, got {reference.dtype}"
                    )
                if not bool(torch.isfinite(reference).all()):
                    raise RcoAdapterError("ref_log_probs values must be finite")
                if not isinstance(loss_mask, torch.Tensor) or loss_mask.shape != batch.shape:
                    actual_mask_shape = (
                        tuple(loss_mask.shape)
                        if isinstance(loss_mask, torch.Tensor)
                        else "non-tensor"
                    )
                    raise RcoAdapterError(
                        "ref_masks entry has shape "
                        f"{actual_mask_shape}, "
                        f"expected {tuple(batch.shape)}"
                    )
                if loss_mask.is_complex() or not loss_mask.is_floating_point():
                    raise RcoAdapterError(
                        f"ref_masks must have a real floating dtype, got {loss_mask.dtype}"
                    )
                if not bool(torch.isfinite(loss_mask).all()):
                    raise RcoAdapterError("ref_masks values must be finite")
                minimum, maximum = loss_mask.aminmax()
                if bool((minimum < 0).any() or (maximum > 1).any()):
                    raise RcoAdapterError("ref_masks values must be in [0, 1]")
                if tuple(reference.shape[:2]) != (batch.size(0), batch.size(1) - 1):
                    raise RcoAdapterError(
                        f"ref_log_probs has shape {tuple(reference.shape)}, expected "
                        f"({batch.size(0)}, {batch.size(1) - 1}, V)"
                    )

                logits = model(input_ids=batch).logits[:, :-1, :].contiguous().float()
                masked_log_probs = torch.log_softmax(logits, dim=-1)
                reference = reference.float()
                per_token_kl = (reference.exp() * (reference - masked_log_probs)).sum(-1)
                shifted_mask = loss_mask[:, 1:].float()
                total_kl += float((per_token_kl * shifted_mask).sum().item())
                total_tokens += float(shifted_mask.sum().item())
        finally:
            remove_wrappers(wrappers)

    return total_kl / max(total_tokens, 1.0)


def get_moe_info(model: Any) -> MoeCounts:
    """Stock-RCO-compatible MoE info: ``(n_layers, n_experts, top_k)``.

    Same name, same argument and same unpacking as the stock RCO
    ``src/search/prune.py::get_moe_info``; the counts are resolved through the
    nested ``text_config`` when the model carries a composite config.
    """
    return resolve_moe_counts(getattr(model, "config", None))
