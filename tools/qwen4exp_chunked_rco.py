#!/usr/bin/env python3
"""Full-depth quantization-aware RCO sweep for the own-Coder experts.

RCO replaces each retained expert one for one; it does not drop experts.  This
tool therefore keeps the 288-expert layer geometry fixed and chooses only the
per-expert width, from exact measured candidates.

The 231 GiB checkpoint cannot be resident, so the model is materialized three
layers at a time.  A chunk runs the real forward once to capture each layer's
inputs and BF16 reference output, then each expert is measured at 2, 3, and 4
bits against that reference.  The boundary hidden state is carried into the
next chunk through a pre-hook on the chunk's first layer, so every layer sees
the same inputs it would see in a BF16 full-model forward.

Output is a per-layer assignment plus an own-Coder allocation in
``strata-expert-width-allocation-v1`` form, aligned to the pruned checkpoint's
expert order.  It makes no end-to-end quality claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import qwen4exp_partial_load as partial
import qwen4exp_quant_aware_rco as rco

SWEEP_FORMAT = "strata-qwen4exp-chunked-rco-sweep-v1"
ALLOCATION_FORMAT = "strata-expert-width-allocation-v1"


def capture_chunk(
    model: Any,
    tokens: Any,
    *,
    start_layer: int,
    boundary: Any | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    """Run one real forward through the chunk, capturing per-layer inputs."""
    import torch

    layers = list(model.model.layers)[start_layer:]
    captured: list[dict[str, Any]] = []
    references: list[Any] = []
    handles = []

    def make_pre(index: int):
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            captured[index] = {"args": args, "kwargs": dict(kwargs)}

        return hook

    def make_post(index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            references[index] = output

        return hook

    if boundary is not None:

        def inject(
            _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return (boundary, *args[1:]), kwargs

        handles.append(layers[0].register_forward_pre_hook(inject, with_kwargs=True))

    for index, layer in enumerate(layers):
        captured.append({})
        references.append(None)
        handles.append(layer.register_forward_pre_hook(make_pre(index), with_kwargs=True))
        handles.append(layer.register_forward_hook(make_post(index)))

    try:
        with torch.no_grad():
            model(input_ids=tokens, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(not item for item in captured):
        raise rco.RcoError("chunk inputs were not captured for every layer")
    if any(reference is None for reference in references):
        raise rco.RcoError("chunk references were not captured for every layer")
    return captured, references[-1], references


def measure_all_width_layer(
    layer: Any, captured: dict[str, Any], reference: Any, bits: int
) -> dict[str, Any]:
    experts = layer.mlp.experts
    gate_up = experts.gate_up_proj.detach()
    down = experts.down_proj.detach()
    original_gate = gate_up.clone()
    original_down = down.clone()
    try:
        for index in range(int(gate_up.shape[0])):
            gate, down_deq = rco.quantize_expert(original_gate[index], original_down[index], bits)
            gate_up[index].copy_(gate.to(gate_up.dtype))
            down[index].copy_(down_deq.to(down.dtype))
        output = rco.call_layer(layer, captured)
    finally:
        gate_up.copy_(original_gate)
        down.copy_(original_down)
    return {"bits": bits, "relative_error": rco.relative_error(reference, output)}


def measure_assignment_layer(
    layer: Any, captured: dict[str, Any], reference: Any, assignment: dict[str, Any]
) -> dict[str, Any]:
    experts = layer.mlp.experts
    gate_up = experts.gate_up_proj.detach()
    down = experts.down_proj.detach()
    original_gate = gate_up.clone()
    original_down = down.clone()
    try:
        for index, bits in enumerate(assignment["bits"]):
            gate, down_deq = rco.quantize_expert(original_gate[index], original_down[index], bits)
            gate_up[index].copy_(gate.to(gate_up.dtype))
            down[index].copy_(down_deq.to(down.dtype))
        output = rco.call_layer(layer, captured)
    finally:
        gate_up.copy_(original_gate)
        down.copy_(original_down)
    return {"assembled_relative_error": rco.relative_error(reference, output)}


def weight_relative_error(original: Any, quantized: Any) -> float:
    """Exact per-expert weight reconstruction error."""

    numerator = (quantized.float() - original.float()).pow(2).sum().sqrt()
    denominator = original.float().pow(2).sum().sqrt().clamp_min(1e-12)
    return float((numerator / denominator).item())


def routed_token_counts(layer: Any, captured: dict[str, Any]) -> list[int]:
    """Count tokens routed to each expert for the captured layer input."""
    import torch

    kwargs = dict(captured["kwargs"])
    kwargs["past_key_values"] = None
    args = list(captured["args"])
    hidden = args[0]
    with torch.no_grad():
        mixed, _hyper, _injection = layer.mlp_hyper_connection(hidden)
        router = layer.mlp.gate
        _logits, _scores, indices = router(mixed.reshape(-1, mixed.shape[-1]))
        counts = torch.bincount(indices.reshape(-1), minlength=288).cpu().tolist()
    return [int(value) for value in counts]


def measure_layer(
    model: Any,
    layer_index: int,
    captured: dict[str, Any],
    reference: Any,
    *,
    byte_budget_per_expert: float,
    expert_sample: int = 0,
) -> dict[str, Any]:

    layer = list(model.model.layers)[layer_index]
    experts = layer.mlp.experts
    gate_up = experts.gate_up_proj.detach()
    down = experts.down_proj.detach()
    per_expert_bytes = {
        str(bits): rco.candidate_bytes(int(gate_up.shape[1]), int(gate_up.shape[2]), bits)
        + rco.candidate_bytes(int(down.shape[1]), int(down.shape[2]), bits)
        for bits in rco.CANDIDATE_BITS
    }
    # Weight-space error for every expert is exact and needs no forward, so the
    # full 288-expert sweep stays cheap even on full-attention layers.
    weight_errors: dict[str, list[float]] = {str(bits): [] for bits in rco.CANDIDATE_BITS}
    for index in range(int(gate_up.shape[0])):
        original_gate = gate_up[index]
        original_down = down[index]
        for bits in rco.CANDIDATE_BITS:
            gate, down_deq = rco.quantize_expert(original_gate, original_down, bits)
            weight_errors[str(bits)].append(
                0.5
                * (
                    weight_relative_error(original_gate, gate)
                    + weight_relative_error(original_down, down_deq)
                )
            )
    # Router load is the measured signal that tracks output impact: experts
    # with no routed tokens cannot change the layer output.  Weighting the
    # exact weight-space error by routed token count keeps the selection
    # activation-aware without one forward per expert.
    counts = routed_token_counts(layer, captured)
    weighted_errors = {
        bits: [error * (count + 1) ** 0.5 for error, count in zip(values, counts, strict=True)]
        for bits, values in weight_errors.items()
    }
    assignment = rco.greedy_assignment(
        weighted_errors,
        bits_list=rco.CANDIDATE_BITS,
        byte_budget_total=byte_budget_per_expert * int(gate_up.shape[0]),
        per_expert_bytes=per_expert_bytes,
    )
    assembled = measure_assignment_layer(layer, captured, reference, assignment)
    controls = {
        str(bits): measure_all_width_layer(layer, captured, reference, bits)
        for bits in rco.CANDIDATE_BITS
    }
    sample: dict[str, Any] = {}
    if expert_sample:
        indexes = list(
            range(0, int(gate_up.shape[0]), max(1, int(gate_up.shape[0]) // expert_sample))
        )[:expert_sample]
        output_errors: dict[str, list[float]] = {str(bits): [] for bits in rco.CANDIDATE_BITS}
        for index in indexes:
            original_gate = gate_up[index].detach().clone()
            original_down = down[index].detach().clone()
            try:
                for bits in rco.CANDIDATE_BITS:
                    gate, down_deq = rco.quantize_expert(original_gate, original_down, bits)
                    gate_up[index].copy_(gate.to(gate_up.dtype))
                    down[index].copy_(down_deq.to(down.dtype))
                    output = rco.call_layer(layer, captured)
                    output_errors[str(bits)].append(rco.relative_error(reference, output))
            finally:
                gate_up[index].copy_(original_gate)
                down[index].copy_(original_down)
        sample = {"experts": len(indexes), "output_errors": output_errors}
    return {
        "assignment": assignment,
        "assembled": assembled,
        "controls": controls,
        "mean_weight_error_by_bits": {
            bits: sum(values) / len(values) for bits, values in weight_errors.items()
        },
        "per_expert_bytes": per_expert_bytes,
        "weight_errors": weight_errors,
        "routed_token_counts": counts,
        "output_sample": sample,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    started = time.perf_counter()
    calibration = np.load(args.calibration, mmap_mode="r")[: args.num_samples]
    tokens = torch.tensor(np.asarray(calibration), dtype=torch.long, device=args.device)
    num_layers = int(args.num_layers)
    boundary = None
    layers: list[dict[str, Any]] = []
    for start in range(0, num_layers, args.chunk_layers):
        end = min(start + args.chunk_layers, num_layers)
        model, load_receipt = partial.load_partial(
            args.model,
            end,
            start_layer=start,
            device=args.device,
            disk_ple=start <= 1 < end,
        )
        captured, boundary, references = capture_chunk(
            model, tokens, start_layer=start, boundary=boundary
        )
        for position in range(end - start):
            result = measure_layer(
                model,
                start + position,
                captured[position],
                references[position],
                byte_budget_per_expert=args.byte_budget_per_expert,
                expert_sample=args.expert_sample,
            )
            layers.append(
                {
                    "layer": start + position,
                    "histogram": result["assignment"]["histogram"],
                    "total_bytes": result["assignment"]["total_bytes"],
                    "assembled_relative_error": result["assembled"]["assembled_relative_error"],
                    "controls": {
                        bits: value["relative_error"] for bits, value in result["controls"].items()
                    },
                    "mean_weight_error_by_bits": result["mean_weight_error_by_bits"],
                    "output_sample": result["output_sample"],
                    "widths": result["assignment"]["bits"],
                    "per_expert_bytes": result["per_expert_bytes"],
                    "per_expert_weight_errors": result["weight_errors"],
                }
            )
            print(
                f"layer {start + position}: {result['assignment']['histogram']} "
                f"assembled {result['assembled']['assembled_relative_error']:.6f}",
                file=sys.stderr,
                flush=True,
            )
        ple_stats = [
            {"module": name, **module.stats}
            for name, module in model.named_modules()
            if isinstance(getattr(module, "stats", None), dict)
        ]
        if layers and ple_stats:
            layers[-1].setdefault("ple_stream", ple_stats)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    histogram: dict[str, int] = {}
    for layer in layers:
        for bits, count in layer["histogram"].items():
            histogram[bits] = histogram.get(bits, 0) + count
    mean_assembled = sum(layer["assembled_relative_error"] for layer in layers) / len(layers)
    return {
        "format": SWEEP_FORMAT,
        "scope": (
            f"layers 0-{num_layers - 1}, {args.num_samples} calibration sequence, "
            "one-for-one expert replacement, no experts dropped"
        ),
        "model": str(args.model),
        "calibration": str(args.calibration),
        "num_samples": args.num_samples,
        "sequence_length": int(calibration.shape[1]),
        "byte_budget_per_expert": args.byte_budget_per_expert,
        "layers": layers,
        "summary": {
            "layers": len(layers),
            "histogram": histogram,
            "total_expert_bytes": sum(layer["total_bytes"] for layer in layers),
            "mean_assembled_relative_error": mean_assembled,
            "experts_per_layer": 288,
        },
        "allocation": {
            "format": ALLOCATION_FORMAT,
            "one_for_one": True,
            "expert_count": 288,
            "note": "widths align to the pruned checkpoint's expert order; no expert is removed",
            "layers": [{"layer": layer["layer"], "widths": layer["widths"]} for layer in layers],
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "non_claims": [
            "Per-layer reconstruction uses BF16 chunk inputs, not a fully cascaded "
            "quantized prefix.",
            "Per-expert selection uses weight-space reconstruction error; output-space "
            "error is measured for the assembled mix and for all-width controls.",
            "Candidates are group-32 RTN at 2, 3, and 4 bits; no GPTQ or stock RCO "
            "optimizer is used.",
            "No end-to-end logits, served throughput, or artifact was produced.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--chunk-layers", type=int, default=3)
    parser.add_argument("--byte-budget-per-expert", type=float, default=0.0)
    parser.add_argument("--expert-sample", type=int, default=0)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.byte_budget_per_expert <= 0:
        args.byte_budget_per_expert = 2_457_600
    try:
        payload = run(args)
    except Exception as error:  # noqa: BLE001 - the sweep reports every failure
        print(
            json.dumps(
                {
                    "format": SWEEP_FORMAT,
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
    print(
        json.dumps(
            {
                "format": SWEEP_FORMAT,
                "layers": payload["summary"]["layers"],
                "histogram": payload["summary"]["histogram"],
                "total_expert_bytes": payload["summary"]["total_expert_bytes"],
                "mean_assembled_relative_error": payload["summary"][
                    "mean_assembled_relative_error"
                ],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
