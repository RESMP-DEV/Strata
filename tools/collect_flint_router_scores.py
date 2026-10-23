#!/usr/bin/env python3
"""Stream Flint BF16 layers and collect router statistics for RCO controls."""

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


def keep_rows(router_scores: Any, keep: int) -> list[list[int]]:
    """Select experts by descending score with source-ID tie-break."""
    if router_scores.shape != (ROUTED_LAYERS, SOURCE_EXPERTS):
        raise ValueError(
            f"router score shape {tuple(router_scores.shape)} does not match "
            f"({ROUTED_LAYERS}, {SOURCE_EXPERTS})"
        )
    if not 0 < keep < SOURCE_EXPERTS:
        raise ValueError(f"keep must be in (0, {SOURCE_EXPERTS})")
    rows = []
    for scores in router_scores:
        order = sorted(range(SOURCE_EXPERTS), key=lambda expert: (-float(scores[expert]), expert))
        rows.append(order[:keep])
    return rows


def fractions(values: Any, selected: list[list[int]]) -> list[float]:
    """Return each layer's retained fraction of a per-expert statistic."""
    return [
        sum(float(row[expert]) for expert in keep) / max(float(row.sum()), 1e-30)
        for row, keep in zip(values, selected, strict=True)
    ]


def build_model(config_path: Path, model_type_module: Any):
    from accelerate import init_empty_weights

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "flint":
        raise ValueError("model_type is not flint")
    flint_config = model_type_module.FlintConfig.from_dict(config)
    with init_empty_weights(include_buffers=False):
        return model_type_module.FlintForCausalLM(flint_config)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from mlx_gptq.flint_torch import FlintConfig, FlintForCausalLM  # noqa: F401
    from mlx_gptq.sequential import Pipeline, RawShardIndex, free_module, materialize_module

    if args.samples <= 0 or args.sequence_length <= 0 or args.batch_size <= 0:
        raise ValueError("samples, sequence length, and batch size must be positive")
    if args.batch_size > args.samples:
        raise ValueError("batch size exceeds sample count")

    tokens = torch.from_numpy(np.load(args.calibration_tokens))
    if tokens.ndim != 2:
        raise ValueError("calibration tokens must have shape [samples, sequence]")
    tokens = tokens[: args.samples, : args.sequence_length].contiguous()
    if tokens.shape[0] != args.samples or tokens.shape[1] != args.sequence_length:
        raise ValueError("calibration token file is smaller than the requested slice")

    model = build_model(
        args.model / "config.json",
        SimpleNamespace(FlintConfig=FlintConfig, FlintForCausalLM=FlintForCausalLM),
    )
    pipeline_args = SimpleNamespace(
        devices=[args.device],
        dtype="bfloat16",
        layers_attr="model.layers",
        checkpoint_model_prefix=None,
        artifact_layers_prefix=None,
    )
    pipeline = Pipeline(model, RawShardIndex(str(args.model)), pipeline_args)
    batches = pipeline.catch_layer0_inputs(tokens, args.batch_size)

    router_scores = torch.zeros(ROUTED_LAYERS, SOURCE_EXPERTS, dtype=torch.float64)
    frequencies = torch.zeros(ROUTED_LAYERS, SOURCE_EXPERTS, dtype=torch.float64)
    dense_layers = set(model.config.mlp_only_layers or [])
    layer_rows: list[dict[str, Any]] = []
    started = time.monotonic()

    for physical_layer in range(model.config.num_hidden_layers):
        layer_started = time.monotonic()
        layer = pipeline.layers[physical_layer]
        prefix = f"model.layers.{physical_layer}"
        materialize_module(layer, prefix, pipeline.shards, args.device, torch.bfloat16)
        hook = None
        if physical_layer not in dense_layers:
            logical_layer = len(layer_rows)

            def make_hook(target_layer: int):
                def hook_fn(_module, _inputs, output):
                    scores = torch.sigmoid(output.detach().float())
                    flat = scores.reshape(-1, SOURCE_EXPERTS)
                    router_scores[target_layer] += flat.sum(dim=0).double().cpu()
                    selected = flat.topk(6, dim=-1).indices.reshape(-1)
                    frequencies[target_layer] += (
                        torch.bincount(selected, minlength=SOURCE_EXPERTS).double().cpu()
                    )

                return hook_fn

            hook = layer.mlp.gate.register_forward_hook(make_hook(logical_layer))
        old_contexts = [item[1] for item in batches]
        try:
            outputs = pipeline.forward_layer(layer, batches, physical_layer)
        finally:
            if hook is not None:
                hook.remove()
            free_module(layer)
        batches = list(zip(outputs, old_contexts, strict=True))
        row = {
            "physical_layer": physical_layer,
            "kind": "dense" if physical_layer in dense_layers else "routed",
            "elapsed_seconds": round(time.monotonic() - layer_started, 3),
        }
        if row["kind"] == "routed":
            row["logical_layer"] = len(layer_rows)
            layer_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if len(layer_rows) != ROUTED_LAYERS:
        raise RuntimeError(f"observed {len(layer_rows)} routed layers, expected {ROUTED_LAYERS}")
    selected = keep_rows(router_scores, args.keep)
    retained_mass = fractions(router_scores, selected)
    retained_frequency = fractions(frequencies, selected)
    mask_document = {
        "schema": "strata-flint-expert-mask-v1",
        "source_experts": SOURCE_EXPERTS,
        "keep_ids": selected,
    }
    mask_bytes = (json.dumps(mask_document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    receipt = {
        "schema": "strata-flint-router-control-v1",
        "model": str(args.model),
        "calibration_tokens": str(args.calibration_tokens),
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "device": args.device,
        "keep_per_layer": args.keep,
        "selection": "sigmoid-router-mass-desc",
        "tie_break": "source-id-asc",
        "router_scores": router_scores.tolist(),
        "top6_frequencies": frequencies.tolist(),
        "keep_ids": selected,
        "retained_router_mass_fraction": retained_mass,
        "retained_top6_frequency_fraction": retained_frequency,
        "retained_router_mass_summary": {
            "mean": float(torch.tensor(retained_mass).mean()),
            "min": min(retained_mass),
            "max": max(retained_mass),
        },
        "retained_top6_frequency_summary": {
            "mean": float(torch.tensor(retained_frequency).mean()),
            "min": min(retained_frequency),
            "max": max(retained_frequency),
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "non_claims": [
            "This is a router-score control, not a Gumbel-STE RCO mask.",
            "No full-model masked KL, materialization, calibration, or pack ran.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    args.mask_output.parent.mkdir(parents=True, exist_ok=True)
    args.mask_output.write_bytes(mask_bytes)
    receipt["mask_output"] = str(args.mask_output)
    receipt["mask_sha256"] = hashlib.sha256(mask_bytes).hexdigest()
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--keep", type=int, default=192)
    args = parser.parse_args()
    receipt = run(args)
    summary = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "router_scores",
            "top6_frequencies",
            "keep_ids",
            "retained_router_mass_fraction",
            "retained_top6_frequency_fraction",
        }
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
