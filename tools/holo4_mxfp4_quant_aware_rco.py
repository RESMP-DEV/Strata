#!/usr/bin/env python3
"""Layer-scope MXFP4 W4A16 quant-aware RCO expert pruning for Holo4.

The experiment is one real Holo4 decoder prefix, loaded from a portable
``omni-use`` layer package. It measures the MoE block on real calibration
hidden states with:

* BF16 expert weights as the reference;
* MXFP4 expert weights (E2M1 values, one E8M0 power-of-two scale per 32
  weights) with BF16 activations, the W4A16 shape;
* expert removal with router renormalization, so pruning removes experts from
  routing rather than leaving a zero-weight hole.

RCO here means measured router-aware expert removal: the impact-aware arm
prunes the experts whose removal costs least in the W4A16 model, with
router-mass and seeded-random arms as controls. This is a layer-scope
prototype, not the full-model RCO optimizer or a deployed artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

FORMAT = "strata-holo4-mxfp4-quant-aware-rco-layer-v1"
GROUP_SIZE = 32
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


class RcoError(ValueError):
    """Raised when the package or experiment contract is not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mxfp4_bytes(shape: tuple[int, int]) -> int:
    """Exact bytes for MXFP4 weights plus one E8M0 scale per 32 values."""
    out_features, in_features = shape
    if out_features <= 0 or in_features <= 0:
        raise RcoError(f"invalid weight shape: {shape}")
    words = (in_features + 7) // 8
    return out_features * words * 4 + out_features * ((in_features + 31) // 32)


def expert_bytes(gate_up_shape: tuple[int, int], down_shape: tuple[int, int]) -> int:
    return mxfp4_bytes(gate_up_shape) + mxfp4_bytes(down_shape)


def choose_prune(
    router_mass: list[float],
    removal_error: list[float],
    count: int,
    *,
    mode: str,
    seed: int = 17,
) -> list[int]:
    """Pick experts to remove: impact-aware, router-only, or seeded random."""
    if len(router_mass) != len(removal_error):
        raise RcoError("router_mass and removal_error must have equal length")
    if not 0 <= count <= len(router_mass):
        raise RcoError(f"prune count {count} outside 0..{len(router_mass)}")
    indices = list(range(len(router_mass)))
    if count == 0:
        return []
    if mode == "impact":
        order = sorted(indices, key=lambda i: (removal_error[i], router_mass[i], i))
        return sorted(order[:count])
    if mode == "router":
        order = sorted(indices, key=lambda i: (router_mass[i], removal_error[i], i))
        return sorted(order[:count])
    if mode == "random":
        return sorted(random.Random(seed).sample(indices, count))
    raise RcoError(f"unknown prune mode: {mode}")


def relative_l2(reference: Any, observed: Any) -> float:
    numerator = (reference.float() - observed.float()).pow(2).sum().sqrt()
    denominator = reference.float().pow(2).sum().sqrt().clamp_min(1e-12)
    return float((numerator / denominator).item())


def e2m1_index(magnitudes: Any) -> Any:
    """Nearest E2M1 magnitude index, ties to even index (round-to-nearest-even)."""
    import torch

    midpoints = torch.tensor(E2M1_MIDPOINTS, dtype=magnitudes.dtype, device=magnitudes.device)
    index = torch.bucketize(magnitudes.contiguous(), midpoints, right=False)
    last = len(E2M1_MIDPOINTS) - 1
    clamped = index.clamp(max=last)
    tie = (index <= last) & (magnitudes == midpoints[clamped])
    # At an exact midpoint the lower index is the nearer-lower value; if that
    # index is odd, the even neighbour is the upper one.
    index = torch.where(tie & (index % 2 == 1), index + 1, index)
    return index.clamp(max=len(E2M1_VALUES) - 1)


def mxfp4_dequant(weight: Any) -> Any:
    """Fake-quantize a 2D weight to MXFP4 and return BF16 values."""
    import torch
    import torch.nn.functional as F

    if weight.ndim != 2:
        raise RcoError(f"expected a 2D weight, got {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    groups = (in_features + GROUP_SIZE - 1) // GROUP_SIZE
    padded = groups * GROUP_SIZE
    values = weight.float()
    if padded != in_features:
        values = F.pad(values, (0, padded - in_features))
    blocks = values.reshape(out_features, groups, GROUP_SIZE)
    max_abs = blocks.abs().amax(dim=-1)
    # Choose the smallest power-of-two scale that keeps the block maximum
    # inside E2M1's largest value, 6. A floor-based exponent can produce a
    # normalized maximum above 6 and silently clamp the largest weight.
    exponents = torch.ceil(torch.log2(max_abs.clamp_min(1e-30) / 6.0))
    exponents = torch.where(max_abs > 0, exponents, torch.zeros_like(exponents))
    exponents = exponents.clamp(-127, 127)
    scales = torch.pow(2.0, exponents)
    magnitudes = blocks.abs() / scales.unsqueeze(-1)
    index = e2m1_index(magnitudes)
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=weight.device)
    quantized = torch.sign(blocks) * table[index] * scales.unsqueeze(-1)
    return quantized.reshape(out_features, padded)[:, :in_features].to(weight.dtype).contiguous()


def load_package(package: Path) -> tuple[Any, dict[str, Any], Any]:
    import torch
    from safetensors.torch import load_file
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTextModel,
    )

    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    config = Qwen3_5MoeTextConfig.from_dict(
        json.loads((package / "text_config.json").read_text(encoding="utf-8"))
    )
    with torch.device("meta"):
        model = Qwen3_5MoeTextModel(config)
    tensors = load_file(str(package / "layer.safetensors"), device="cpu")
    params = dict(model.named_parameters())
    missing = [name for name in params if name not in tensors]
    unexpected = [name for name in tensors if name not in params]
    if missing or unexpected:
        raise RcoError(f"package mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    for name, tensor in tensors.items():
        module_name, _, parameter_name = name.rpartition(".")
        module = model.get_submodule(module_name) if module_name else model
        module._parameters[parameter_name] = torch.nn.Parameter(tensor, requires_grad=False)
    model.eval()
    return model, manifest, config


def capture_mlp_input(model: Any, tokens: Any, layer: int) -> Any:
    import torch

    captured: dict[str, Any] = {}

    def hook(_module: Any, args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        captured["hidden"] = args[0].detach()

    handle = model.layers[layer].mlp.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=tokens, use_cache=False)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RcoError("MLP input was not captured")
    return captured["hidden"]


def block_forward(
    block: Any,
    hidden_states: Any,
    removed: set[int] | None = None,
) -> Any:
    """The exact MoE block math, with optional experts removed from routing."""
    import torch
    import torch.nn.functional as F

    batch, sequence, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim)
    shared = block.shared_expert(flat)
    logits = F.linear(flat, block.gate.weight)
    if removed:
        logits = logits.clone()
        logits[:, sorted(removed)] = float("-inf")
    probabilities = F.softmax(logits, dtype=torch.float, dim=-1)
    top_values, top_indices = torch.topk(probabilities, block.gate.top_k, dim=-1)
    top_values = top_values / top_values.sum(dim=-1, keepdim=True)
    top_values = top_values.to(logits.dtype)
    routed = block.experts(flat, top_indices, top_values)
    shared = torch.sigmoid(block.shared_expert_gate(flat)) * shared
    return (routed + shared).reshape(batch, sequence, hidden_dim)


def router_stats(block: Any, hidden_states: Any) -> tuple[list[float], list[int]]:
    import torch
    import torch.nn.functional as F

    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = F.linear(flat, block.gate.weight)
    probabilities = F.softmax(logits, dtype=torch.float, dim=-1)
    mass = probabilities.sum(dim=0)
    _, indices = torch.topk(probabilities, block.gate.top_k, dim=-1)
    counts = torch.bincount(indices.flatten(), minlength=logits.shape[-1])
    return mass.tolist(), counts.tolist()


def quantize_block_experts(block: Any) -> dict[str, Any]:
    import torch

    experts = block.experts
    gate_up = experts.gate_up_proj.detach()
    down = experts.down_proj.detach()
    quantized_gate = torch.empty_like(gate_up)
    quantized_down = torch.empty_like(down)
    for index in range(int(gate_up.shape[0])):
        quantized_gate[index] = mxfp4_dequant(gate_up[index])
        quantized_down[index] = mxfp4_dequant(down[index])
    with torch.no_grad():
        experts.gate_up_proj.copy_(quantized_gate)
        experts.down_proj.copy_(quantized_down)
    return {
        "experts": int(gate_up.shape[0]),
        "gate_up_shape": list(gate_up.shape[1:]),
        "down_shape": list(down.shape[1:]),
        "bytes_per_expert": expert_bytes(tuple(gate_up.shape[1:]), tuple(down.shape[1:])),
        "total_bytes": expert_bytes(tuple(gate_up.shape[1:]), tuple(down.shape[1:]))
        * int(gate_up.shape[0]),
        "layout": "E2M1 values, one E8M0 power-of-two scale per 32 weights, BF16 activations",
    }


def evaluate_prune_set(
    block: Any,
    hidden: Any,
    bf16_reference: Any,
    w4a16_reference: Any,
    removed: list[int],
) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        output = block_forward(block, hidden, set(removed))
    return {
        "removed": removed,
        "removed_count": len(removed),
        "relative_error_vs_bf16": relative_l2(bf16_reference, output),
        "relative_error_vs_w4a16": relative_l2(w4a16_reference, output),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    torch.set_num_threads(max(1, int(os.environ.get("HOLO4_RCO_THREADS", os.cpu_count() or 1))))
    started = time.perf_counter()
    model, manifest, config = load_package(args.package)
    if args.device != "cpu":
        # The package is materialized as CPU parameters on a meta-built module;
        # non-parameter buffers have no data, so a blanket .to() is invalid.
        raise RcoError(
            "this layer-scope pilot runs on CPU; cuda needs explicit buffer materialization"
        )
    block = model.layers[args.layer].mlp
    calibration = np.load(args.calibration, mmap_mode="r")
    if args.num_samples + args.heldout_samples > calibration.shape[0]:
        raise RcoError("not enough calibration rows for calibration + heldout")
    calibration_rows = np.asarray(calibration[: args.num_samples, : args.sequence_length])
    heldout_rows = np.asarray(
        calibration[
            args.num_samples : args.num_samples + args.heldout_samples,
            : args.sequence_length,
        ]
    )
    if calibration_rows.size == 0 or heldout_rows.size == 0:
        raise RcoError("calibration and heldout slices must be non-empty")
    calibration_tokens = torch.tensor(calibration_rows, dtype=torch.long, device=args.device)
    heldout_tokens = torch.tensor(heldout_rows, dtype=torch.long, device=args.device)

    calibration_hidden = capture_mlp_input(model, calibration_tokens, args.layer)
    heldout_hidden = capture_mlp_input(model, heldout_tokens, args.layer)
    with torch.no_grad():
        calibration_bf16 = block(calibration_hidden)
        heldout_bf16 = block(heldout_hidden)

    quantized = quantize_block_experts(block)
    with torch.no_grad():
        calibration_w4a16 = block_forward(block, calibration_hidden)
        heldout_w4a16 = block_forward(block, heldout_hidden)

    mass, counts = router_stats(block, calibration_hidden)
    expert_count = int(block.experts.gate_up_proj.shape[0])
    if args.limit_experts and args.limit_experts < expert_count:
        # Measure the lowest-mass experts: that is the RCO-relevant candidate
        # pool, while keeping the first Strix run bounded.
        candidate_order = sorted(range(expert_count), key=lambda i: (mass[i], i))
        candidates = sorted(candidate_order[: args.limit_experts])
    else:
        candidates = list(range(expert_count))
    removal_errors: dict[int, float] = {}
    with torch.no_grad():
        for index in candidates:
            output = block_forward(block, calibration_hidden, {index})
            removal_errors[index] = relative_l2(calibration_w4a16, output)

    top_k = int(block.gate.top_k)
    prune_count = int(round(len(candidates) * args.prune_fraction))
    # Pruning a candidate only reduces the candidate pool; the full layer keeps
    # expert_count - prune_count experts, which must still cover top_k.
    max_prune = min(len(candidates), expert_count - top_k)
    prune_count = min(prune_count, max_prune)
    if prune_count < 0:
        raise RcoError("prune fraction leaves fewer experts than top_k")
    target = list(candidates)
    arms: dict[str, Any] = {}
    for mode in ("impact", "router", "random"):
        selected = choose_prune(
            [mass[i] for i in target],
            [removal_errors[i] for i in target],
            prune_count,
            mode=mode,
            seed=args.seed,
        )
        removed = [target[i] for i in selected]
        calibration_metrics = evaluate_prune_set(
            block,
            calibration_hidden,
            calibration_bf16,
            calibration_w4a16,
            removed,
        )
        heldout_metrics = evaluate_prune_set(
            block,
            heldout_hidden,
            heldout_bf16,
            heldout_w4a16,
            removed,
        )
        arms[mode] = {
            "removed": removed,
            "removed_count": len(removed),
            "removed_router_mass": sum(mass[i] for i in removed),
            "calibration": calibration_metrics,
            "heldout": heldout_metrics,
        }

    table = [
        {
            "expert": index,
            "router_mass": mass[index],
            "router_selected_tokens": counts[index],
            "removal_relative_error": removal_errors.get(index),
        }
        for index in range(expert_count)
    ]
    return {
        "format": FORMAT,
        "recorded_at_unix": time.time(),
        "scope": (
            f"Holo4 layer {args.layer} MoE block only, calibration "
            f"{calibration_rows.shape}, heldout {heldout_rows.shape}, "
            f"{len(candidates)} candidate experts"
        ),
        "package": {
            "path": str(args.package),
            "manifest_sha256": hashlib.sha256(
                (args.package / "manifest.json").read_bytes()
            ).hexdigest(),
            "layer_safetensors_sha256": manifest["layer_safetensors_sha256"],
            "source_revision": manifest["source_revision"],
            "prefix_layers": manifest["prefix_layers"],
        },
        "calibration": {
            "path": str(args.calibration),
            "sha256": sha256_file(args.calibration),
            "num_samples": int(calibration_rows.shape[0]),
            "heldout_samples": int(heldout_rows.shape[0]),
            "sequence_length": int(calibration_rows.shape[1]),
            "tokens": int(calibration_rows.size + heldout_rows.size),
        },
        "w4a16": {
            **quantized,
            "calibration_relative_error_vs_bf16": relative_l2(calibration_bf16, calibration_w4a16),
            "heldout_relative_error_vs_bf16": relative_l2(heldout_bf16, heldout_w4a16),
        },
        "router": {
            "top_k": int(block.gate.top_k),
            "mass_sum": float(sum(mass)),
            "mass_min": min(mass),
            "mass_max": max(mass),
            "selected_tokens_min": min(counts),
            "selected_tokens_max": max(counts),
        },
        "prune": {
            "fraction": args.prune_fraction,
            "count": prune_count,
            "candidate_experts": len(candidates),
            "bytes_saved_per_arm": prune_count * quantized["bytes_per_expert"],
            "arms": arms,
        },
        "expert_table": table,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "threads": torch.get_num_threads(),
            "device": args.device,
        },
        "non_claims": [
            "Layer-scope MoE-block reconstruction, not full-model RCO or a serving artifact.",
            "MXFP4 W4A16 is measured as fake-quantized BF16 compute, not a "
            "kernel throughput result.",
            "The impact arm measures removal on calibration hidden states; "
            "heldout rows are a separate check, not a benchmark.",
            "Only the exported prefix is loaded; this does not prove "
            "later-layer or vision behavior.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--heldout-samples", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--limit-experts", type=int)
    parser.add_argument("--prune-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run(args)
    except Exception as error:  # noqa: BLE001 - receipt is the failure surface.
        print(
            json.dumps(
                {
                    "format": FORMAT,
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
