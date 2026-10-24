#!/usr/bin/env python3
"""Layer-scope quantization-aware RCO prototype for Qwen4Exp experts.

This is a measured prototype, not the full-model RCO optimizer.  It answers one
falsifiable question on real weights: for one decoder layer, which per-expert
bitwidth assignment under a byte budget best preserves the BF16 layer output?

Method:
1. Stream the real layer prefix (embeddings plus the selected layer).
2. Capture the BF16 layer output on calibration and held-out tokens.
3. For every expert and every candidate bitwidth, quantize that expert's fused
   gate_up/down weights with group-32 symmetric round-to-nearest, patch them in,
   and measure the layer-output error against BF16.
4. Choose a per-expert assignment under a byte budget with a greedy Lagrangian
   sweep, then measure the assembled layer once.

The receipt reports exact candidate bytes, per-expert errors, the chosen
assignment, and the measured assembled error.  It makes no end-to-end quality
or serving claim.

For a correctness-first control while a calibrated artifact is unavailable, use
``--assignment-mode all-width --all-width-bits 4``.  This skips the greedy
per-expert sweep and measures one fixed group-32 RTN width for every expert.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

RCO_FORMAT = "strata-qwen4exp-quant-aware-rco-layer-v1"
GROUP_SIZE = 32
CANDIDATE_BITS = (4, 3, 2)


class RcoError(ValueError):
    """Raised when the layer-scope RCO contract cannot be satisfied."""


def packed_bytes(out_features: int, in_features: int, bits: int) -> int:
    """Exact int32-word bytes for group-packed weights."""
    if bits not in (2, 3, 4):
        raise RcoError(f"unsupported bits {bits}")
    words = (in_features * bits + 31) // 32
    return out_features * words * 4


def scale_bytes(out_features: int, in_features: int, *, scale_dtype_bytes: int = 2) -> int:
    return out_features * ((in_features + GROUP_SIZE - 1) // GROUP_SIZE) * scale_dtype_bytes


def candidate_bytes(out_features: int, in_features: int, bits: int) -> int:
    return packed_bytes(out_features, in_features, bits) + scale_bytes(out_features, in_features)


def quantize_group(weight: Any, bits: int) -> tuple[Any, Any]:
    """Symmetric group-32 round-to-nearest quantization of a 2D weight."""
    import torch

    out_features, in_features = weight.shape
    groups = (in_features + GROUP_SIZE - 1) // GROUP_SIZE
    padded = groups * GROUP_SIZE
    if padded != in_features:
        weight = torch.nn.functional.pad(weight, (0, padded - in_features))
    shaped = weight.reshape(out_features, groups, GROUP_SIZE).float()
    qmax = float((1 << (bits - 1)) - 1)
    scales = shaped.abs().amax(dim=-1).clamp_min(1e-8) / qmax
    codes = torch.round(shaped / scales.unsqueeze(-1)).clamp(-qmax, qmax).to(torch.int32)
    return codes, scales


def dequantize_group(codes: Any, scales: Any, shape: tuple[int, int]) -> Any:

    out_features, in_features = shape
    groups = scales.shape[1]
    values = codes.float() * scales.unsqueeze(-1)
    return values.reshape(out_features, groups * GROUP_SIZE)[:, :in_features]


def capture_layer_inputs(model: Any, tokens: Any, layer_index: int = 0) -> dict[str, Any]:
    """Capture the exact arguments the real model passes to layer 0."""
    import torch

    captured: dict[str, Any] = {}

    def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured["args"] = args
        captured["kwargs"] = dict(kwargs)

    handle = model.model.layers[layer_index].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=tokens, use_cache=False)
    finally:
        handle.remove()
    if "args" not in captured:
        raise RcoError("layer-0 inputs were not captured")
    return captured


def call_layer(layer: Any, captured: dict[str, Any]) -> Any:
    import torch

    kwargs = dict(captured["kwargs"])
    kwargs["past_key_values"] = None
    args = list(captured["args"])
    with torch.no_grad():
        output = layer(*args, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def quantize_expert(gate_up: Any, down: Any, bits: int) -> tuple[Any, Any]:
    gate_codes, gate_scales = quantize_group(gate_up, bits)
    down_codes, down_scales = quantize_group(down, bits)
    gate = dequantize_group(gate_codes, gate_scales, tuple(gate_up.shape))
    down_deq = dequantize_group(down_codes, down_scales, tuple(down.shape))
    return gate, down_deq


def relative_error(reference: Any, observed: Any) -> float:

    numerator = (reference.float() - observed.float()).pow(2).sum().sqrt()
    denominator = reference.float().pow(2).sum().sqrt().clamp_min(1e-12)
    return float((numerator / denominator).item())


def measure_candidates(
    model: Any,
    captured: dict[str, Any],
    layer_index: int,
    reference: Any,
    bits_list: tuple[int, ...],
    *,
    limit_experts: int | None,
) -> dict[str, Any]:
    """Measure each candidate on one expert at a time against BF16 output."""

    layer = model.model.layers[layer_index]
    experts = layer.mlp.experts
    gate_up = experts.gate_up_proj.detach()
    down = experts.down_proj.detach()
    num_experts = int(gate_up.shape[0])
    count = num_experts if limit_experts is None else min(limit_experts, num_experts)
    results: dict[str, list[float]] = {str(bits): [] for bits in bits_list}
    for index in range(count):
        original_gate = gate_up[index].detach().clone()
        original_down = down[index].detach().clone()
        try:
            for bits in bits_list:
                gate, down_deq = quantize_expert(original_gate, original_down, bits)
                gate_up[index].copy_(gate.to(gate_up.dtype))
                down[index].copy_(down_deq.to(down.dtype))
                output = call_layer(layer, captured)
                results[str(bits)].append(relative_error(reference, output))
        finally:
            gate_up[index].copy_(original_gate)
            down[index].copy_(original_down)
    return {"experts_measured": count, "per_expert_relative_error": results}


def greedy_assignment(
    errors: dict[str, list[float]],
    *,
    bits_list: tuple[int, ...],
    byte_budget_total: float,
    per_expert_bytes: dict[str, int],
) -> dict[str, Any]:
    """Start every expert at the cheapest width and buy the best upgrades.

    Each step applies the expert and width upgrade with the largest error
    reduction per marginal byte that still fits the remaining budget.
    """
    order = sorted(bits_list)
    expert_count = len(next(iter(errors.values())))
    current = [order[0]] * expert_count
    spent = per_expert_bytes[str(order[0])] * expert_count
    while True:
        best: tuple[float, int, int, int] | None = None
        for index in range(expert_count):
            current_bits = current[index]
            for candidate in order:
                if candidate <= current_bits:
                    continue
                cost = per_expert_bytes[str(candidate)] - per_expert_bytes[str(current_bits)]
                if spent + cost > byte_budget_total:
                    continue
                gain = errors[str(current_bits)][index] - errors[str(candidate)][index]
                if gain <= 0:
                    continue
                score = gain / cost
                if best is None or score > best[0]:
                    best = (score, index, candidate, cost)
        if best is None:
            break
        _, index, candidate, cost = best
        current[index] = candidate
        spent += cost
    histogram: dict[str, int] = {}
    for bits in current:
        histogram[str(bits)] = histogram.get(str(bits), 0) + 1
    return {
        "bits": current,
        "histogram": histogram,
        "total_bytes": spent,
        "mean_bytes_per_expert": spent / expert_count,
    }


def all_width_assignment(
    expert_count: int,
    *,
    bits: int,
    per_expert_bytes: dict[str, int],
) -> dict[str, Any]:
    """Return the correctness-first, no-iMatrix fixed-width RTN assignment."""
    if bits not in CANDIDATE_BITS:
        raise RcoError(f"unsupported fixed width {bits}; choose from {CANDIDATE_BITS}")
    if expert_count < 0 or str(bits) not in per_expert_bytes:
        raise RcoError(f"invalid expert count or byte table for width {bits}")
    spent = per_expert_bytes[str(bits)] * expert_count
    return {
        "bits": [bits] * expert_count,
        "histogram": {str(bits): expert_count},
        "total_bytes": spent,
        "mean_bytes_per_expert": per_expert_bytes[str(bits)],
    }


def measure_assignment(
    model: Any,
    captured: dict[str, Any],
    layer_index: int,
    reference: Any,
    assignment: dict[str, Any],
) -> dict[str, Any]:
    """Measure the assembled mixed-width layer once against BF16."""
    layer = model.model.layers[layer_index]
    experts = layer.mlp.experts
    gate_up = experts.gate_up_proj.detach()
    down = experts.down_proj.detach()
    original_gate = gate_up.clone()
    original_down = down.clone()
    try:
        for index, bits in enumerate(assignment["bits"]):
            gate, down_deq = quantize_expert(original_gate[index], original_down[index], bits)
            gate_up[index].copy_(gate.to(gate_up.dtype))
            down[index].copy_(down_deq.to(down.dtype))
        output = call_layer(layer, captured)
    finally:
        gate_up.copy_(original_gate)
        down.copy_(original_down)
    return {"assembled_relative_error": relative_error(reference, output)}


def measure_all_width(
    model: Any,
    captured: dict[str, Any],
    layer_index: int,
    reference: Any,
    bits: int,
) -> dict[str, Any]:
    """Measure every expert at one width, as an all-Wn control."""
    layer = model.model.layers[layer_index]
    experts = layer.mlp.experts
    gate_up = experts.gate_up_proj.detach()
    down = experts.down_proj.detach()
    original_gate = gate_up.clone()
    original_down = down.clone()
    try:
        for index in range(int(gate_up.shape[0])):
            gate, down_deq = quantize_expert(original_gate[index], original_down[index], bits)
            gate_up[index].copy_(gate.to(gate_up.dtype))
            down[index].copy_(down_deq.to(down.dtype))
        output = call_layer(layer, captured)
    finally:
        gate_up.copy_(original_gate)
        down.copy_(original_down)
    return {"bits": bits, "relative_error": relative_error(reference, output)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import qwen4exp_partial_load as partial
    import torch

    started = time.perf_counter()
    model, load_receipt = partial.load_partial(
        args.model,
        args.layer + 1,
        device=args.device,
        disk_ple=args.layer >= 1,
    )
    calibration = np.load(args.calibration, mmap_mode="r")[: args.num_samples]
    tokens = torch.tensor(np.asarray(calibration), dtype=torch.long, device=args.device)
    captured = capture_layer_inputs(model, tokens, args.layer)
    reference = call_layer(model.model.layers[args.layer], captured)
    experts = model.model.layers[args.layer].mlp.experts
    per_expert_bytes = {
        str(bits): candidate_bytes(
            int(experts.gate_up_proj.shape[1]), int(experts.gate_up_proj.shape[2]), bits
        )
        + candidate_bytes(int(experts.down_proj.shape[1]), int(experts.down_proj.shape[2]), bits)
        for bits in CANDIDATE_BITS
    }
    if args.assignment_mode == "greedy":
        measured = measure_candidates(
            model,
            captured,
            args.layer,
            reference,
            CANDIDATE_BITS,
            limit_experts=args.limit_experts,
        )
        budget_per_expert = args.byte_budget_per_expert
        if budget_per_expert <= 0:
            budget_per_expert = (per_expert_bytes["3"] + per_expert_bytes["4"]) / 2
        assignment = greedy_assignment(
            measured["per_expert_relative_error"],
            bits_list=CANDIDATE_BITS,
            byte_budget_total=budget_per_expert * measured["experts_measured"],
            per_expert_bytes=per_expert_bytes,
        )
        expert_count = measured["experts_measured"]
        mean_errors = {
            bits: sum(values) / len(values)
            for bits, values in measured["per_expert_relative_error"].items()
        }
    else:
        expert_count = int(experts.gate_up_proj.shape[0])
        assignment = all_width_assignment(
            expert_count,
            bits=args.all_width_bits,
            per_expert_bytes=per_expert_bytes,
        )
        budget_per_expert = per_expert_bytes[str(args.all_width_bits)]
        mean_errors = {}
    assembled = measure_assignment(model, captured, args.layer, reference, assignment)
    heldout = None
    controls: dict[str, Any] = {}
    heldout_assembled: dict[str, Any] = {}
    if args.heldout is not None:
        heldout_matrix = np.load(args.heldout, mmap_mode="r")[: args.num_heldout]
        heldout_tokens = torch.tensor(
            np.asarray(heldout_matrix), dtype=torch.long, device=args.device
        )
        heldout = capture_layer_inputs(model, heldout_tokens, args.layer)
        heldout_reference = call_layer(model.model.layers[args.layer], heldout)
        controls = {
            str(bits): measure_all_width(model, heldout, args.layer, heldout_reference, bits)
            for bits in CANDIDATE_BITS
        }
        heldout_assembled = measure_assignment(
            model, heldout, args.layer, heldout_reference, assignment
        )
    ple_stats = [
        {"module": name, **module.stats}
        for name, module in model.named_modules()
        if isinstance(getattr(module, "stats", None), dict)
    ]
    return {
        "format": RCO_FORMAT,
        "ple_stream": ple_stats,
        "scope": (
            f"layer {args.layer} only, {args.num_samples} calibration samples, "
            f"{expert_count} experts assigned by {args.assignment_mode}"
        ),
        "model": str(args.model),
        "calibration": str(args.calibration),
        "num_samples": args.num_samples,
        "sequence_length": int(calibration.shape[1]),
        "candidate_bits": list(CANDIDATE_BITS),
        "candidate_bytes_per_expert": per_expert_bytes,
        "assignment_mode": args.assignment_mode,
        "all_width_bits": args.all_width_bits if args.assignment_mode == "all-width" else None,
        "mean_relative_error_by_bits": mean_errors,
        "experts_assigned": expert_count,
        "assignment": {
            "histogram": assignment["histogram"],
            "total_bytes": assignment["total_bytes"],
            "mean_bytes_per_expert": assignment["mean_bytes_per_expert"],
            "byte_budget_per_expert": budget_per_expert,
        },
        "assembled": assembled,
        "heldout": {
            "path": str(args.heldout) if args.heldout else None,
            "num_samples": args.num_heldout if args.heldout else 0,
            "all_width_controls": controls,
            "assembled": heldout_assembled,
        },
        "load_receipt": {
            "loaded_tensors": load_receipt["loaded_tensors"],
            "loaded_bytes": load_receipt["loaded_bytes"],
            "shards_read": load_receipt["shards_read"],
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "non_claims": [
            "This is a layer-scope reconstruction measurement, not end-to-end model quality.",
            "Candidates are group-32 symmetric RTN; no GPTQ, activation-aware, "
            "or healing pass is included.",
            "No serving path, throughput, or full-model artifact was produced.",
            "The assignment rule is a prototype, not the stock RCO optimizer.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--num-heldout", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--limit-experts", type=int)
    parser.add_argument("--byte-budget-per-expert", type=float, default=0.0)
    parser.add_argument("--assignment-mode", choices=("greedy", "all-width"), default="greedy")
    parser.add_argument("--all-width-bits", type=int, choices=CANDIDATE_BITS, default=4)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if args.assignment_mode == "all-width" and args.limit_experts is not None:
        parser.error(
            "--limit-experts applies to the greedy candidate sweep, not --assignment-mode all-width"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run(args)
    except Exception as error:  # noqa: BLE001 - the prototype reports every failure
        print(
            json.dumps(
                {
                    "format": RCO_FORMAT,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
