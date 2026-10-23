#!/usr/bin/env python3
"""Measure exact leave-one-out expert impact on the full Flint source.

The search runs the real 256-expert router on real calibration activations.
For every candidate expert it computes the exact routed-output change when
that expert is removed and the router re-selects the next best expert, using
the deployed sigmoid score normalization. Greedy rounds then prune the
lowest-impact experts per layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__:
    from .flint_compression_plan import ROUTED_LAYERS, SOURCE_EXPERTS
else:
    import flint_compression_plan as plan

    ROUTED_LAYERS = plan.ROUTED_LAYERS
    SOURCE_EXPERTS = plan.SOURCE_EXPERTS


def parse_remove_rounds(value: str) -> list[int]:
    rounds = [int(item) for item in value.split(",") if item.strip()]
    if not rounds or any(count <= 0 for count in rounds):
        raise ValueError("remove rounds must be positive integers")
    return rounds


def expert_outputs(experts: Any, hidden: Any, slot_experts: Any) -> Any:
    """Return the expert output of every token and routing slot.

    Args:
        experts: EagerExperts module with stacked gate, up, and down weights.
        hidden: [tokens, hidden] router input.
        slot_experts: [tokens, slots] expert id per routing slot.

    Returns:
        [tokens, slots, hidden] tensor of routed expert outputs.
    """
    import torch
    import torch.nn.functional as F

    tokens, slots = slot_experts.shape
    width = hidden.shape[-1]
    outputs = torch.zeros(tokens, slots, width, dtype=hidden.dtype, device=hidden.device)
    for expert in torch.unique(slot_experts).tolist():
        token_index, slot_index = (slot_experts == expert).nonzero(as_tuple=True)
        if token_index.numel() == 0:
            continue
        sample = hidden[token_index]
        gate = F.linear(sample, experts.gate_proj[expert])
        up = F.linear(sample, experts.up_proj[expert])
        value = F.linear(F.silu(gate) * up, experts.down_proj[expert])
        outputs[token_index, slot_index] = value
    return outputs


@__import__("torch").no_grad()
def round_impacts(
    block: Any,
    hidden: Any,
    removed: Any,
    top_k: int,
) -> tuple[Any, Any, Any, Any]:
    """Score the exact impact of removing each still-available expert.

    The routed output with the current removal set is compared against the
    routed output when one extra expert is removed and the router promotes the
    next best expert. Weights follow the deployed rule: sigmoid scores of the
    selected experts normalized by their sum.

    Returns:
        impacts: [experts] float64 squared output change per expert.
        frequency: [experts] float64 selection count per expert.
        energy: float64 reference routed output energy.
        score_sum: [experts] float64 sigmoid router mass per expert.
    """
    import torch

    flat = hidden.reshape(-1, hidden.shape[-1])
    scores = torch.sigmoid(block.gate(flat).float())
    masked = scores.masked_fill(removed.unsqueeze(0), float("-inf"))
    values, indices = torch.topk(masked, top_k + 1, dim=-1)
    slot_outputs = expert_outputs(block.experts, flat, indices)

    selected_scores = values[:, :top_k]
    selected_outputs = slot_outputs[:, :top_k]
    denominator = selected_scores.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    weights = selected_scores / denominator
    current = (selected_outputs * weights.unsqueeze(-1)).sum(dim=1)
    energy = float((current.float() ** 2).sum().item())

    impacts = torch.zeros(scores.shape[-1], dtype=torch.float64, device=flat.device)
    frequency = torch.zeros(scores.shape[-1], dtype=torch.float64, device=flat.device)
    score_sum = scores.sum(dim=0).double()
    next_scores = values[:, top_k]
    next_outputs = slot_outputs[:, top_k]
    for slot in range(top_k):
        slot_score = selected_scores[:, slot]
        others = (selected_scores.sum(dim=-1) - slot_score).unsqueeze(-1)
        new_denominator = (others[:, 0] + next_scores).clamp(min=1e-20)
        keep_total = (selected_scores.unsqueeze(-1) * selected_outputs).sum(dim=1)
        keep_total = keep_total - slot_score.unsqueeze(-1) * selected_outputs[:, slot]
        new_output = (keep_total + next_scores.unsqueeze(-1) * next_outputs) / (
            new_denominator.unsqueeze(-1)
        )
        delta = (new_output - current).float()
        change = (delta**2).sum(dim=-1)
        impacts.index_add_(0, indices[:, slot], change.double())
        frequency.index_add_(0, indices[:, slot], torch.ones_like(change, dtype=torch.float64))
    return impacts, frequency, energy, score_sum


def analyze_block(
    block: Any,
    hidden_batches: list[Any],
    remove_rounds: list[int],
    top_k: int,
    num_experts: int,
) -> dict[str, Any]:
    """Run the greedy leave-one-out search for one routed layer."""
    import torch

    device = hidden_batches[0].device
    removed = torch.zeros(num_experts, dtype=torch.bool, device=device)
    reference_energy = 0.0
    round_receipts = []
    kept = list(range(num_experts))
    for round_index, remove_count in enumerate(remove_rounds):
        impacts = torch.zeros(num_experts, dtype=torch.float64, device=device)
        frequency = torch.zeros(num_experts, dtype=torch.float64, device=device)
        score_sums = torch.zeros(num_experts, dtype=torch.float64, device=device)
        for hidden in hidden_batches:
            round_impact, round_frequency, energy, round_score_sum = round_impacts(
                block, hidden, removed, top_k
            )
            impacts += round_impact
            frequency += round_frequency
            score_sums += round_score_sum
            if round_index == 0:
                reference_energy += energy
        available = [expert for expert in kept if not bool(removed[expert])]
        ranked = sorted(
            available,
            key=lambda expert: (
                float(impacts[expert]),
                float(score_sums[expert]),
                expert,
            ),
        )
        victims = ranked[:remove_count]
        for expert in victims:
            removed[expert] = True
            kept.remove(expert)
        round_receipts.append(
            {
                "round": round_index,
                "removed": len(victims),
                "removed_experts": victims,
                "impact_min": float(impacts[[v for v in victims]].min().item()),
                "impact_max": float(impacts[[v for v in victims]].max().item()),
                "reference_energy": reference_energy,
            }
        )
    return {
        "keep_ids": sorted(kept),
        "impacts": impacts.tolist(),
        "frequency": frequency.tolist(),
        "router_score_sums": score_sums.tolist(),
        "rounds": round_receipts,
        "reference_energy": reference_energy,
    }


def build_model(config_path: Path, model_type_module: Any):
    from accelerate import init_empty_weights

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "flint":
        raise ValueError("model_type is not flint")
    flint_config = model_type_module.FlintConfig.from_dict(config)
    with init_empty_weights(include_buffers=False):
        return model_type_module.FlintForCausalLM(flint_config)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from mlx_gptq.flint_torch import FlintConfig, FlintForCausalLM  # noqa: F401
    from mlx_gptq.sequential import (
        Pipeline,
        RawShardIndex,
        free_module,
        materialize_module,
    )

    if args.samples <= 0 or args.sequence_length <= 0 or args.batch_size <= 0:
        raise ValueError("samples, sequence length, and batch size must be positive")
    remove_rounds = parse_remove_rounds(args.remove_rounds)
    pruned_total = sum(remove_rounds)
    if pruned_total >= SOURCE_EXPERTS:
        raise ValueError("remove rounds would prune every expert")

    tokens = torch.from_numpy(np.load(args.calibration_tokens))
    if tokens.ndim != 2:
        raise ValueError("calibration tokens must have shape [samples, sequence]")
    start = args.sample_offset
    window = tokens[start : start + args.samples, : args.sequence_length].contiguous()
    if window.shape[0] != args.samples or window.shape[1] != args.sequence_length:
        raise ValueError("calibration token file is smaller than the requested slice")

    model = build_model(
        args.model / "config.json",
        SimpleNamespace(FlintConfig=FlintConfig, FlintForCausalLM=FlintForCausalLM),
    )
    pipeline = Pipeline(
        model,
        RawShardIndex(str(args.model)),
        SimpleNamespace(
            devices=[args.device],
            dtype="bfloat16",
            layers_attr="model.layers",
            checkpoint_model_prefix=None,
            artifact_layers_prefix=None,
        ),
    )
    batches = pipeline.catch_layer0_inputs(window, args.batch_size)
    dense_layers = set(model.config.mlp_only_layers or [])
    top_k = model.config.num_experts_per_tok
    layer_receipts = []
    keep_rows: list[list[int]] = []
    started = time.monotonic()

    for physical_layer in range(model.config.num_hidden_layers):
        layer_started = time.monotonic()
        layer = pipeline.layers[physical_layer]
        materialize_module(
            layer,
            f"model.layers.{physical_layer}",
            pipeline.shards,
            args.device,
            torch.bfloat16,
        )
        captured: list[Any] = []
        hook = None
        if physical_layer not in dense_layers:

            def capture(_module, args_tuple, sink=captured):
                sink.append(args_tuple[0])

            hook = layer.mlp.register_forward_pre_hook(capture)
        contexts = [item[1] for item in batches]
        try:
            outputs = pipeline.forward_layer(layer, batches, physical_layer)
            if hook is not None:
                result = analyze_block(
                    layer.mlp,
                    captured,
                    remove_rounds,
                    top_k,
                    model.config.num_experts,
                )
                keep_rows.append(result["keep_ids"])
        finally:
            if hook is not None:
                hook.remove()
            free_module(layer)
        batches = list(zip(outputs, contexts, strict=True))
        row = {
            "physical_layer": physical_layer,
            "kind": "dense" if physical_layer in dense_layers else "routed",
            "elapsed_seconds": round(time.monotonic() - layer_started, 3),
        }
        if row["kind"] == "routed":
            row["logical_layer"] = len(layer_receipts)
            row["keep_count"] = len(keep_rows[-1])
            layer_receipts.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if len(keep_rows) != ROUTED_LAYERS:
        raise RuntimeError(f"observed {len(keep_rows)} routed layers, expected {ROUTED_LAYERS}")
    if any(len(row) != SOURCE_EXPERTS - pruned_total for row in keep_rows):
        raise RuntimeError("keep rows do not carry the exact prune budget")

    mask_document = {
        "schema": "strata-flint-expert-mask-v1",
        "source_experts": SOURCE_EXPERTS,
        "keep_ids": keep_rows,
    }
    mask_bytes = (json.dumps(mask_document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.mask_output.parent.mkdir(parents=True, exist_ok=True)
    args.mask_output.write_bytes(mask_bytes)
    receipt = {
        "schema": "strata-flint-prune-impact-v1",
        "model": str(args.model),
        "calibration_tokens": str(args.calibration_tokens),
        "calibration_sha256": sha256(args.calibration_tokens),
        "sample_offset": args.sample_offset,
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "device": args.device,
        "remove_rounds": remove_rounds,
        "pruned_per_layer": pruned_total,
        "kept_per_layer": SOURCE_EXPERTS - pruned_total,
        "selection": "greedy-exact-leave-one-out-reroute",
        "objective": "relative squared routed-output change with deployed routing",
        "keep_ids": keep_rows,
        "mask_output": str(args.mask_output),
        "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "layer_rows": layer_receipts,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "non_claims": [
            "This is a layer-local leave-one-out search, not a full-model Gumbel-STE search.",
            "Impact is measured with upstream layers unchanged.",
            "No materialized checkpoint, quantization, or serving result follows.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calibration-tokens", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mask-output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--remove-rounds", default="32,16,8,8")
    args = parser.parse_args()
    receipt = run(args)
    summary = {
        key: value for key, value in receipt.items() if key not in {"keep_ids", "layer_rows"}
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
