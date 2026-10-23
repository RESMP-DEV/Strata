#!/usr/bin/env python3
"""Evaluate a full or runtime-pruned Flint MLX model on token windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


def load_mask(path: Path) -> list[list[int]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "strata-flint-expert-mask-v1":
        raise ValueError("mask schema must be strata-flint-expert-mask-v1")
    source_experts = int(document.get("source_experts", 0))
    rows = document.get("keep_ids")
    if source_experts <= 0 or not isinstance(rows, list) or not rows:
        raise ValueError("mask lacks source_experts or keep_ids")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError("mask keep rows have different widths")
    keep = widths.pop()
    if not 0 < keep <= source_experts:
        raise ValueError("mask keep count is outside the source expert range")
    for row in rows:
        if len(set(row)) != keep or min(row) < 0 or max(row) >= source_experts:
            raise ValueError("mask keep row has a duplicate or out-of-range expert")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_bytes(model: Any) -> int:
    from mlx.utils import tree_flatten

    return sum(value.nbytes for _, value in tree_flatten(model.parameters()))


def apply_mask(model: Any, keep_rows: list[list[int]]) -> int:
    import mlx.core as mx

    routed = [layer for layer in model.model.layers if hasattr(layer.mlp, "switch_mlp")]
    if len(routed) != len(keep_rows):
        raise ValueError(f"model has {len(routed)} routed layers but mask has {len(keep_rows)}")
    for layer, keep in zip(routed, keep_rows, strict=True):
        indices = mx.array(keep, dtype=mx.int32)
        gate = layer.mlp.gate
        gate.weight = mx.take(gate.weight, indices, axis=0)
        projections = (
            layer.mlp.switch_mlp.gate_proj,
            layer.mlp.switch_mlp.up_proj,
            layer.mlp.switch_mlp.down_proj,
        )
        for projection in (item for item in projections if item is not None):
            projection.weight = mx.take(projection.weight, indices, axis=0)
            scales = getattr(projection, "scales", None)
            if scales is not None:
                projection.scales = mx.take(scales, indices, axis=0)
            biases = getattr(projection, "biases", None)
            if biases is not None:
                projection.biases = mx.take(biases, indices, axis=0)
            bias = getattr(projection, "bias", None)
            if bias is not None:
                projection.bias = mx.take(bias, indices, axis=0)
        mx.eval(gate.weight, layer.mlp.switch_mlp.gate_proj.weight)
    model.args.num_experts = len(keep_rows[0])
    return len(routed)


def score(model: Any, tokens: Any, offset: int, samples: int, seq_len: int) -> dict[str, Any]:
    import mlx.core as mx

    if tokens.ndim != 2:
        raise ValueError("tokens must be rank 2")
    selected = tokens[offset : offset + samples, : seq_len + 1]
    if selected.shape != (samples, seq_len + 1):
        raise ValueError("token file is smaller than the requested windows")
    per_window_nll = []
    per_window_top1 = []
    started = time.monotonic()
    for row in selected:
        inputs = mx.array(row[:seq_len])[None]
        targets = mx.array(row[1:])
        logits = model(inputs)[0].astype(mx.float32)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nll = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
        top1 = mx.argmax(logits, axis=-1) == targets
        mx.eval(nll, top1)
        per_window_nll.append(float(nll.mean()))
        per_window_top1.append(float(top1.mean()))
    mean_nll = sum(per_window_nll) / len(per_window_nll)
    return {
        "schema": "strata-flint-mlx-runtime-quality-v1",
        "samples": samples,
        "sample_offset": offset,
        "sequence_length": seq_len,
        "mean_nll_nats": mean_nll,
        "perplexity": math.exp(mean_nll),
        "top1_accuracy": sum(per_window_top1) / len(per_window_top1),
        "per_window_nll": per_window_nll,
        "per_window_top1": per_window_top1,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from mlx_lm import load

    model, _tokenizer = load(str(args.model))
    bytes_before = parameter_bytes(model)
    keep_rows = load_mask(args.mask) if args.mask is not None else None
    applied = apply_mask(model, keep_rows) if keep_rows is not None else 0
    bytes_after = parameter_bytes(model)
    tokens = np.load(args.tokens)
    result = score(model, tokens, args.sample_offset, args.samples, args.sequence_length)
    result.update(
        {
            "model": str(args.model),
            "mask": str(args.mask) if args.mask is not None else None,
            "mask_sha256": sha256(args.mask) if args.mask is not None else None,
            "routed_layers_pruned": applied,
            "num_experts": model.args.num_experts,
            "model_parameter_bytes_before": bytes_before,
            "model_parameter_bytes_after": bytes_after,
            "model_parameter_bytes_saved": bytes_before - bytes_after,
            "tokens": str(args.tokens),
            "tokens_sha256": sha256(args.tokens),
            "non_claims": [
                "This is teacher-forced token quality, not a task-success benchmark.",
                "Runtime pruning uses the already-calibrated MXFP4 weights.",
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.npz_output,
        nll=np.asarray(result["per_window_nll"], dtype=np.float64),
        top1=np.asarray(result["per_window_top1"], dtype=np.float64),
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--npz-output", required=True, type=Path)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    args = parser.parse_args()
    result = run(args)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"per_window_nll", "per_window_top1"}
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
