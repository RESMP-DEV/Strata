#!/usr/bin/env python3
"""Stream every pruned Flint layer and verify finite final logits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


def build_model(config_path: Path, module: Any):
    from accelerate import init_empty_weights

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "flint":
        raise ValueError("model_type is not flint")
    if config.get("num_experts") != 192 or config.get("original_num_experts") != 256:
        raise ValueError("expected a physical 192-of-256 expert checkpoint")
    flint_config = module.FlintConfig.from_dict(config)
    with init_empty_weights(include_buffers=False):
        return module.FlintForCausalLM(flint_config)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx_gptq.sequential as sequential
    from mlx_gptq.flint_torch import FlintConfig, FlintForCausalLM

    tokens = torch.from_numpy(np.load(args.calibration_tokens))
    if tokens.ndim != 2:
        raise ValueError("calibration tokens must be rank 2")
    tokens = tokens[: args.samples, : args.sequence_length].contiguous()
    model = build_model(
        args.model / "config.json",
        SimpleNamespace(FlintConfig=FlintConfig, FlintForCausalLM=FlintForCausalLM),
    )
    pipeline = sequential.Pipeline(
        model,
        sequential.RawShardIndex(str(args.model)),
        SimpleNamespace(
            devices=[args.device],
            dtype="bfloat16",
            layers_attr="model.layers",
            checkpoint_model_prefix=None,
            artifact_layers_prefix=None,
        ),
    )
    batches = pipeline.catch_layer0_inputs(tokens, args.batch_size)
    started = time.monotonic()
    for physical_layer in range(model.config.num_hidden_layers):
        layer = pipeline.layers[physical_layer]
        sequential.materialize_module(
            layer,
            f"model.layers.{physical_layer}",
            pipeline.shards,
            args.device,
            torch.bfloat16,
        )
        try:
            outputs = pipeline.forward_layer(layer, batches, physical_layer)
        finally:
            sequential.free_module(layer)
        contexts = [row[1] for row in batches]
        batches = list(zip(outputs, contexts, strict=True))
        print(
            json.dumps({"physical_layer": physical_layer, "status": "loaded"}),
            flush=True,
        )

    sequential.materialize_module(
        model.model.norm,
        pipeline._checkpoint_name("model.norm"),
        pipeline.shards,
        args.device,
        torch.bfloat16,
    )
    sequential.materialize_module(
        model.lm_head,
        pipeline._checkpoint_name("lm_head"),
        pipeline.shards,
        args.device,
        torch.bfloat16,
    )
    hidden = batches[-1][0].to(args.device, torch.bfloat16)
    with torch.no_grad():
        logits = model.lm_head(model.model.norm(hidden)).float()
    finite = bool(torch.isfinite(logits).all().item())
    if not finite:
        raise RuntimeError("pruned checkpoint produced non-finite logits")
    top_ids = logits[0, -1].topk(8).indices.tolist()
    return {
        "schema": "strata-pruned-flint-load-v1",
        "model": str(args.model),
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "device": args.device,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "all_layers_loaded": model.config.num_hidden_layers,
        "logits_finite": finite,
        "top8_last_token_ids": top_ids,
        "non_claims": [
            "Finite teacher-forced logits do not establish generation quality.",
            "No Gumbel-STE search, calibration, quantization, or benchmark ran.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calibration-tokens", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
