#!/usr/bin/env python3
"""Compare QDQ selection methodologies at the MoE block on real layers.

Three selection signals are scored against the same measured objective, the
MoE-block output error after the chosen widths are applied:

* ``weight``  - per-expert weight-space quantization error.
* ``router``  - weight error multiplied by sqrt(routed token count).
* ``qdq``     - measured per-expert block-output error of the quantized expert.

The width candidates use the number format each width would deploy with:
4 bits uses MXFP4 (E2M1 values, one E8M0 power-of-two scale per 32 weights),
while 3 and 2 bits use group-32 symmetric round-to-nearest.  The comparison is
layer-scope and makes no end-to-end quality claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import qwen4exp_partial_load as partial
import qwen4exp_quant_aware_rco as rco
from holo4_mxfp4_quant_aware_rco import mxfp4_dequant

COMPARE_FORMAT = "strata-qwen4exp-qdq-method-compare-v1"
SIGNALS = ("weight", "router", "qdq")


def candidate(width: int, gate_up: Any, down: Any) -> tuple[Any, Any]:
    if width == 4:
        return mxfp4_dequant(gate_up), mxfp4_dequant(down)
    return rco.quantize_expert(gate_up, down, width)


def run_block(block: Any, captured: dict[str, Any]) -> Any:
    import torch

    args = list(captured["args"])
    kwargs = dict(captured["kwargs"])
    with torch.no_grad():
        output = block(*args, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--byte-budget-per-expert", type=float, default=2_457_600)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    import numpy as np
    import torch

    started = time.perf_counter()
    matrix = np.load(args.calibration, mmap_mode="r")[:1]
    tokens = torch.tensor(np.asarray(matrix), dtype=torch.long, device=args.device)
    model, load = partial.load_partial(
        args.model, args.num_layers, device=args.device, disk_ple=True
    )
    layers = list(model.model.layers)
    # One real forward, capturing the MoE block inputs and references.
    inputs: list[dict[str, Any]] = [{} for _ in layers]
    outputs: list[Any] = [None for _ in layers]
    handles = []
    for index, layer in enumerate(layers):
        block = layer.mlp

        def pre(_m: Any, a: tuple[Any, ...], k: dict[str, Any], i: int = index) -> None:
            inputs[i] = {"args": a, "kwargs": dict(k)}

        def post(_m: Any, _a: tuple[Any, ...], out: Any, i: int = index) -> None:
            outputs[i] = out

        handles.append(block.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(block.register_forward_hook(post))
    try:
        with torch.no_grad():
            model(input_ids=tokens, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    rows = []
    for index, layer in enumerate(layers):
        block = layer.mlp
        experts = block.experts
        gate_up = experts.gate_up_proj.detach()
        down = experts.down_proj.detach()
        hidden_in = inputs[index]["args"][0]
        _logits, _scores, routed = block.gate(hidden_in.reshape(-1, hidden_in.shape[-1]))
        counts = torch.bincount(routed.reshape(-1), minlength=288).cpu().tolist()
        reference = outputs[index]
        weight_errors: dict[str, list[float]] = {str(w): [] for w in rco.CANDIDATE_BITS}
        qdq_errors: dict[str, list[float]] = {str(w): [] for w in rco.CANDIDATE_BITS}
        for expert in range(288):
            original_gate = gate_up[expert].detach().clone()
            original_down = down[expert].detach().clone()
            try:
                for width in rco.CANDIDATE_BITS:
                    gate, down_deq = candidate(width, original_gate, original_down)
                    weight_errors[str(width)].append(
                        0.5
                        * (
                            rco.relative_error(original_gate, gate)
                            + rco.relative_error(original_down, down_deq)
                        )
                    )
                    gate_up[expert].copy_(gate.to(gate_up.dtype))
                    down[expert].copy_(down_deq.to(down.dtype))
                    qdq_errors[str(width)].append(
                        rco.relative_error(reference, run_block(block, inputs[index]))
                    )
            finally:
                gate_up[expert].copy_(original_gate)
                down[expert].copy_(original_down)
        signals = {
            "weight": weight_errors,
            "router": {
                w: [e * (c + 1) ** 0.5 for e, c in zip(values, counts, strict=True)]
                for w, values in weight_errors.items()
            },
            "qdq": qdq_errors,
        }
        layer_rows = {"layer": index, "signals": {}}
        for name, errors in signals.items():
            assignment = rco.greedy_assignment(
                errors,
                bits_list=rco.CANDIDATE_BITS,
                byte_budget_total=args.byte_budget_per_expert * 288,
                per_expert_bytes={
                    str(w): rco.candidate_bytes(1280, 2560, w) + rco.candidate_bytes(2560, 640, w)
                    for w in rco.CANDIDATE_BITS
                },
            )
            original_gate = gate_up.clone()
            original_down = down.clone()
            try:
                for expert, width in enumerate(assignment["bits"]):
                    gate, down_deq = candidate(width, original_gate[expert], original_down[expert])
                    gate_up[expert].copy_(gate.to(gate_up.dtype))
                    down[expert].copy_(down_deq.to(down.dtype))
                assembled = rco.relative_error(reference, run_block(block, inputs[index]))
            finally:
                gate_up.copy_(original_gate)
                down.copy_(original_down)
            layer_rows["signals"][name] = {
                "histogram": assignment["histogram"],
                "assembled_block_relative_error": assembled,
            }
        rows.append(layer_rows)
    payload = {
        "format": COMPARE_FORMAT,
        "model": str(args.model),
        "calibration": str(args.calibration),
        "num_layers": args.num_layers,
        "byte_budget_per_expert": args.byte_budget_per_expert,
        "layers": rows,
        "mean_assembled_by_signal": {
            name: sum(row["signals"][name]["assembled_block_relative_error"] for row in rows)
            / len(rows)
            for name in SIGNALS
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "non_claims": [
            "MoE-block scope, one 512-token calibration sequence.",
            "4-bit uses MXFP4 QDQ; 3 and 2 bits use int group-32 RTN.",
            "No end-to-end logits, artifact, or serving measurement.",
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "format": COMPARE_FORMAT,
                "mean": payload["mean_assembled_by_signal"],
                "elapsed": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
