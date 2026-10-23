#!/usr/bin/env python3
"""Teacher-forced quality comparison for full and pruned Flint checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": dirty}


def select_windows(
    tokens: Any, sample_offset: int, samples: int, sequence_length: int
) -> tuple[Any, Any]:
    """Return aligned inputs and targets from disjoint calibration rows."""
    if tokens.ndim != 2:
        raise ValueError("token file must have shape [windows, source_sequence]")
    if sample_offset < 0 or samples <= 0 or sequence_length <= 0:
        raise ValueError("offset, samples, and sequence length must be nonnegative/positive")
    if sample_offset + samples > tokens.shape[0] or sequence_length + 1 > tokens.shape[1]:
        raise ValueError("requested held-out token windows exceed the token file")
    windows = tokens[sample_offset : sample_offset + samples, : sequence_length + 1]
    inputs = windows[:, :-1]
    targets = windows[:, 1:]
    if hasattr(inputs, "contiguous"):
        inputs, targets = inputs.contiguous(), targets.contiguous()
    else:
        inputs, targets = inputs.copy(), targets.copy()
    return inputs, targets


def summarize_token_metrics(nll: Any, top1: Any) -> dict[str, float | int]:
    flat_nll = nll.reshape(-1).astype("float64", copy=False)
    mean_nll = float(flat_nll.mean())
    return {
        "tokens": int(flat_nll.size),
        "mean_nll_nats": mean_nll,
        "mean_nll_bits": mean_nll / math.log(2.0),
        "perplexity": math.exp(mean_nll),
        "top1_correct": int(top1.sum()),
        "top1_accuracy": float(top1.astype("float64", copy=False).mean()),
    }


def paired_bootstrap(
    baseline_nll: Any,
    candidate_nll: Any,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the paired NLL delta by resampling complete windows."""
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if baseline_nll.shape != candidate_nll.shape:
        raise ValueError("baseline and candidate token metrics have different shapes")
    baseline_means = baseline_nll.mean(axis=1, dtype="float64")
    candidate_means = candidate_nll.mean(axis=1, dtype="float64")
    deltas = candidate_means - baseline_means
    rng = np_random(seed)
    sampled = deltas[rng.integers(0, len(deltas), size=(iterations, len(deltas)))].mean(axis=1)
    return {
        "mean_delta_nats": float(deltas.mean()),
        "mean_delta_bits": float(deltas.mean() / math.log(2.0)),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "delta_nats_ci95": [float(x) for x in np_percentile(sampled, [2.5, 97.5])],
    }


def compare_metrics(
    baseline_nll: Any,
    candidate_nll: Any,
    baseline_top1: Any,
    candidate_top1: Any,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = paired_bootstrap(baseline_nll, candidate_nll, iterations, seed)
    result.update(
        {
            "delta_perplexity": math.exp(float(candidate_nll.mean(dtype="float64")))
            - math.exp(float(baseline_nll.mean(dtype="float64"))),
            "top1_agreement": float((baseline_top1 == candidate_top1).mean()),
            "both_correct": int((baseline_top1 & candidate_top1).sum()),
            "baseline_only_correct": int((baseline_top1 & ~candidate_top1).sum()),
            "candidate_only_correct": int((~baseline_top1 & candidate_top1).sum()),
        }
    )
    return result


def stack_batch_rows(parts: list[Any]) -> Any:
    """Stack per-batch token metric rows into the original sample order."""
    import torch

    return torch.cat(parts, dim=0)


def np_random(seed: int) -> Any:
    import numpy as np

    return np.random.default_rng(seed)


def np_percentile(values: Any, percentiles: list[float]) -> Any:
    import numpy as np

    return np.percentile(values, percentiles)


def build_model(config_path: Path, module: Any):
    import json

    from accelerate import init_empty_weights

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "flint":
        raise ValueError("model_type is not flint")
    flint_config = module.FlintConfig.from_dict(config)
    with init_empty_weights(include_buffers=False):
        return module.FlintForCausalLM(flint_config)


def evaluate_checkpoint(
    label: str,
    model_path: Path,
    inputs: Any,
    targets: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch
    from mlx_gptq.flint_torch import FlintConfig, FlintForCausalLM  # noqa: F401
    from mlx_gptq.sequential import Pipeline, RawShardIndex, free_module, materialize_module

    started = time.monotonic()
    model = build_model(
        model_path / "config.json",
        SimpleNamespace(FlintConfig=FlintConfig, FlintForCausalLM=FlintForCausalLM),
    )
    pipeline = Pipeline(
        model,
        RawShardIndex(str(model_path)),
        SimpleNamespace(
            devices=[args.device],
            dtype="bfloat16",
            layers_attr="model.layers",
            checkpoint_model_prefix=None,
            artifact_layers_prefix=None,
        ),
    )
    batches = pipeline.catch_layer0_inputs(inputs, args.batch_size)
    layer_rows: list[dict[str, Any]] = []
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
        try:
            outputs = pipeline.forward_layer(layer, batches, physical_layer)
        finally:
            free_module(layer)
        batches = list(zip(outputs, [row[1] for row in batches], strict=True))
        row = {
            "physical_layer": physical_layer,
            "elapsed_seconds": round(time.monotonic() - layer_started, 3),
        }
        layer_rows.append(row)
        print(json.dumps({label: row}, sort_keys=True), flush=True)

    materialize_module(
        model.model.norm,
        pipeline._checkpoint_name("model.norm"),
        pipeline.shards,
        args.device,
        torch.bfloat16,
    )
    materialize_module(
        model.lm_head,
        pipeline._checkpoint_name("lm_head"),
        pipeline.shards,
        args.device,
        torch.bfloat16,
    )

    nll_rows: list[Any] = []
    top1_rows: list[Any] = []
    for batch_index, (hid_cpu, _context) in enumerate(batches):
        hidden = hid_cpu.to(args.device, torch.bfloat16)
        sample_start = batch_index * args.batch_size
        batch_nll_parts: list[Any] = []
        batch_top1_parts: list[Any] = []
        for start in range(0, hidden.shape[1], args.logit_batch_tokens):
            stop = min(start + args.logit_batch_tokens, hidden.shape[1])
            with torch.no_grad():
                normalized = model.model.norm(hidden[:, start:stop])
                logits = model.lm_head(normalized).float()
                target = targets[sample_start : sample_start + hidden.shape[0], start:stop].to(
                    args.device
                )
                log_probs = torch.log_softmax(logits, dim=-1)
                token_nll = -log_probs.gather(-1, target[:, :, None]).squeeze(-1)
            batch_nll_parts.append(token_nll.detach().cpu())
            batch_top1_parts.append((logits.argmax(-1) == target).detach().cpu())
            del logits, log_probs, token_nll
        nll_rows.append(torch.cat(batch_nll_parts, dim=1))
        top1_rows.append(torch.cat(batch_top1_parts, dim=1))

    import numpy as np

    nll = stack_batch_rows(nll_rows).numpy().astype("float32", copy=False)
    top1 = stack_batch_rows(top1_rows).numpy().astype(bool, copy=False)
    if not np.isfinite(nll).all():
        raise RuntimeError(f"{label} produced non-finite NLL")
    free_module(model.model.norm)
    free_module(model.lm_head)
    expert_map = model_path / "expert-id-map.json"
    return {
        "label": label,
        "model": str(model_path),
        "config_sha256": sha256(model_path / "config.json"),
        "expert_id_map_sha256": sha256(expert_map) if expert_map.exists() else None,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_experts": model.config.num_experts,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "layer_rows": layer_rows,
        **summarize_token_metrics(nll, top1),
        "nll": nll,
        "top1": top1,
    }


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    tokens = torch.from_numpy(np.load(args.tokens))
    inputs, targets = select_windows(tokens, args.sample_offset, args.samples, args.sequence_length)
    raw_output = args.output.with_suffix(".npz")
    args.output.unlink(missing_ok=True)
    raw_output.unlink(missing_ok=True)
    receipt: dict[str, Any] = {
        "schema": "strata-flint-quality-v1",
        "status": "running",
        "started_epoch": time.time(),
        "git": git_state(),
        "tokens": str(args.tokens),
        "tokens_sha256": sha256(args.tokens),
        "sample_offset": args.sample_offset,
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "logit_batch_tokens": args.logit_batch_tokens,
        "device": args.device,
        "non_claims": [
            "Held-out calibration-token NLL is not a broad benchmark suite.",
            "No generation, quantization, Gumbel-STE optimization, or serving claim is made.",
        ],
    }
    write_receipt(args.output, receipt)
    try:
        baseline = evaluate_checkpoint("baseline", args.baseline_model, inputs, targets, args)
        candidate = evaluate_checkpoint("candidate", args.candidate_model, inputs, targets, args)
        comparison = compare_metrics(
            baseline["nll"],
            candidate["nll"],
            baseline["top1"],
            candidate["top1"],
            args.bootstrap_iterations,
            args.bootstrap_seed,
        )
        np.savez_compressed(
            raw_output,
            baseline_nll=baseline["nll"],
            candidate_nll=candidate["nll"],
            baseline_top1=baseline["top1"],
            candidate_top1=candidate["top1"],
        )
        receipt.update(
            {
                "status": "passed",
                "elapsed_seconds": round(time.time() - receipt["started_epoch"], 3),
                "baseline": {
                    key: value
                    for key, value in baseline.items()
                    if key not in {"nll", "top1", "layer_rows"}
                },
                "candidate": {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"nll", "top1", "layer_rows"}
                },
                "comparison": comparison,
                "raw_output": str(raw_output),
                "raw_output_sha256": sha256(raw_output),
            }
        )
    except BaseException as error:
        receipt.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(time.time() - receipt["started_epoch"], 3),
            }
        )
        write_receipt(args.output, receipt)
        raise
    write_receipt(args.output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-model", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-offset", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--logit-batch-tokens", type=int, default=64)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=173)
    args = parser.parse_args()
    receipt = run(args)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "baseline_perplexity": receipt["baseline"]["perplexity"],
                "candidate_perplexity": receipt["candidate"]["perplexity"],
                "comparison": receipt["comparison"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
