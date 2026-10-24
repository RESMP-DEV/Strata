#!/usr/bin/env python3
"""Measure the IQ arms Strata can actually serve on real own-Coder layers.

The native pack stores one gate/up type and one down type per layer, so today's
servable choices are uniform arms: IQ3_S, IQ3_XXS, or IQ4_XS gate/up with IQ4_NL
down.  Each arm is applied to real layer weights with the pinned ggml reference
quantizers, and the MoE block output is measured against BF16.  The per-expert
allocation from the QDQ sweep is measured for reference even though this engine
format cannot represent it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import iq3_qdq
import qwen4exp_chunked_rco as chunked
import qwen4exp_partial_load as partial

ARM_FORMAT = "strata-qwen4exp-iq-arm-measure-v1"
ARMS = {
    "3": {
        "gate_up": iq3_qdq.IQ3_S,
        "down": iq3_qdq.IQ4_NL,
        "gate_up_type": "IQ3_S",
        "down_type": "IQ4_NL",
    },
    "3x": {
        "gate_up": iq3_qdq.IQ3_XXS,
        "down": iq3_qdq.IQ4_NL,
        "gate_up_type": "IQ3_XXS",
        "down_type": "IQ4_NL",
    },
    "4": {
        "gate_up": iq3_qdq.IQ4_XS,
        "down": iq3_qdq.IQ4_NL,
        "gate_up_type": "IQ4_XS",
        "down_type": "IQ4_NL",
    },
}


def role_rows(experts: Any, role: str) -> tuple[Any, int, int]:
    """Return (flat rows, rows per expert, row count) for one expert role."""
    import numpy as np

    tensor = getattr(experts, role).detach()
    rows_per_expert = tensor.shape[1]
    flat = tensor.reshape(-1, tensor.shape[-1]).float().cpu().numpy()
    return np.ascontiguousarray(flat), rows_per_expert, flat.shape[0]


def apply_arm(layer: Any, arm: dict[str, Any], workers: int) -> tuple[int, Any, Any]:
    """Apply one uniform arm; return bytes and the original tensors."""
    import torch

    experts = layer.mlp.experts
    originals = (
        experts.gate_up_proj.detach().clone(),
        experts.down_proj.detach().clone(),
    )
    gate, gate_rows, gate_total = role_rows(experts, "gate_up_proj")
    down, down_rows, down_total = role_rows(experts, "down_proj")
    gate_out, gate_bytes = iq3_qdq.qdq_parallel(gate, kind=arm["gate_up"], workers=workers)
    down_out, down_bytes = iq3_qdq.qdq_parallel(down, kind=arm["down"], workers=workers)
    with torch.no_grad():
        experts.gate_up_proj.copy_(
            torch.from_numpy(gate_out.reshape(originals[0].shape)).to(originals[0].dtype)
        )
        experts.down_proj.copy_(
            torch.from_numpy(down_out.reshape(originals[1].shape)).to(originals[1].dtype)
        )
    del gate_rows, gate_total, down_rows, down_total
    return gate_bytes + down_bytes, originals


def apply_mixed(layer: Any, arms: list[str], workers: int) -> tuple[int, Any, Any]:
    """Apply the per-expert allocation; return bytes and the originals."""
    import numpy as np
    import torch

    experts = layer.mlp.experts
    originals = (
        experts.gate_up_proj.detach().clone(),
        experts.down_proj.detach().clone(),
    )
    gate, gate_rows, _ = role_rows(experts, "gate_up_proj")
    down, down_rows, _ = role_rows(experts, "down_proj")
    gate_out = np.empty_like(gate)
    down_out = np.empty_like(down)
    total_bytes = 0
    for arm_name in sorted(set(arms)):
        selected = [index for index in range(len(arms)) if arms[index] == arm_name]
        if not selected:
            continue
        arm = ARMS[arm_name]
        gate_selected = np.concatenate(
            [gate[index * gate_rows : (index + 1) * gate_rows] for index in selected]
        )
        down_selected = np.concatenate(
            [down[index * down_rows : (index + 1) * down_rows] for index in selected]
        )
        gate_out_selected, gate_bytes = iq3_qdq.qdq_parallel(
            gate_selected, kind=arm["gate_up"], workers=workers
        )
        down_out_selected, down_bytes = iq3_qdq.qdq_parallel(
            down_selected, kind=arm["down"], workers=workers
        )
        total_bytes += gate_bytes + down_bytes
        for slot, index in enumerate(selected):
            gate_out[index * gate_rows : (index + 1) * gate_rows] = gate_out_selected[
                slot * gate_rows : (slot + 1) * gate_rows
            ]
            down_out[index * down_rows : (index + 1) * down_rows] = down_out_selected[
                slot * down_rows : (slot + 1) * down_rows
            ]
    with torch.no_grad():
        experts.gate_up_proj.copy_(
            torch.from_numpy(gate_out.reshape(originals[0].shape)).to(originals[0].dtype)
        )
        experts.down_proj.copy_(
            torch.from_numpy(down_out.reshape(originals[1].shape)).to(originals[1].dtype)
        )
    return total_bytes, originals


def restore(layer: Any, originals: tuple[Any, Any]) -> None:
    import torch

    with torch.no_grad():
        layer.mlp.experts.gate_up_proj.copy_(originals[0])
        layer.mlp.experts.down_proj.copy_(originals[1])


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    started = time.perf_counter()
    matrix = np.load(args.calibration, mmap_mode="r")[: args.num_samples]
    tokens = torch.tensor(np.asarray(matrix), dtype=torch.long, device=args.device)
    layers_wanted = [int(item) for item in args.layers.split(",") if item.strip()]
    model, _load = partial.load_partial(
        args.model, max(layers_wanted) + 1, device=args.device, disk_ple=True
    )
    captured, _boundary, references, block_inputs, block_references = chunked.capture_chunk(
        model, tokens, start_layer=0, boundary=None
    )
    allocation = json.loads(args.allocation.read_text()) if args.allocation else None
    rows = []
    for position, layer_index in enumerate(layers_wanted):
        layer = list(model.model.layers)[layer_index]
        block = layer.mlp
        experts = layer.mlp.experts
        entry: dict[str, Any] = {"layer": layer_index}
        for name, arm in ARMS.items():
            started_arm = time.perf_counter()
            bytes_total, originals = apply_arm(layer, arm, args.workers)
            with torch.no_grad():
                output = block(*block_inputs[position]["args"], **block_inputs[position]["kwargs"])
            error = chunked.rco.relative_error(block_references[position], output)
            restore(layer, originals)
            entry[name] = {
                "gate_up_type": arm["gate_up_type"],
                "down_type": arm["down_type"],
                "block_relative_error": error,
                "bytes_per_expert": bytes_total // experts.gate_up_proj.shape[0],
                "elapsed_seconds": round(time.perf_counter() - started_arm, 3),
            }
        if allocation is not None:
            started_mixed = time.perf_counter()
            bytes_total, originals = apply_mixed(
                layer, allocation["layers"][layer_index]["arms"], args.workers
            )
            with torch.no_grad():
                output = block(*block_inputs[position]["args"], **block_inputs[position]["kwargs"])
            error = chunked.rco.relative_error(block_references[position], output)
            restore(layer, originals)
            entry["mixed"] = {
                "block_relative_error": error,
                "bytes_per_expert": bytes_total // experts.gate_up_proj.shape[0],
                "elapsed_seconds": round(time.perf_counter() - started_mixed, 3),
                "representable_in_current_pack": False,
            }
        rows.append(entry)
        print(json.dumps(entry), file=sys.stderr, flush=True)
    summary = {}
    names = list(ARMS) + (["mixed"] if allocation is not None else [])
    for name in names:
        values = [row[name]["block_relative_error"] for row in rows]
        byte_values = [row[name]["bytes_per_expert"] for row in rows]
        summary[name] = {
            "mean_block_relative_error": sum(values) / len(values),
            "bytes_per_expert": byte_values[0],
            "total_expert_bytes": byte_values[0] * 288 * 48,
        }
    return {
        "format": ARM_FORMAT,
        "model": str(args.model),
        "calibration": str(args.calibration),
        "num_samples": args.num_samples,
        "sequence_length": int(matrix.shape[1]),
        "layers": layers_wanted,
        "rows": rows,
        "summary": summary,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "workers": args.workers,
            "cpu_count": os.cpu_count(),
        },
        "non_claims": [
            "MoE-block scope on real layer weights, not end-to-end logits.",
            "One calibration sequence of 512 tokens.",
            "The mixed arm is not representable in the current per-layer pack format.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--layers", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run(args)
    except Exception as error:  # noqa: BLE001 - the measurement reports every failure
        print(
            json.dumps(
                {
                    "format": ARM_FORMAT,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
            )
        )
        return 2
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(
        json.dumps({"format": ARM_FORMAT, "summary": payload["summary"]}, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
