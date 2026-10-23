#!/usr/bin/env python3
"""Materialize a selected prefix of a Qwen4Exp checkpoint without loading it all.

The 288-expert own-Coder checkpoint is 231 GiB while B550 has 91 GiB of RAM, so
the stock loader cannot materialize the model.  This tool builds the configured
model on the ``meta`` device, keeps only the requested number of decoder layers,
and copies just those tensors in from the safetensors shards.  Every loaded
tensor is compared against its source tensor, so a partial materialization is
never assumed correct.

It is the streaming foundation for a layer-at-a-time quantization run: a full
run materializes one decoder layer, quantizes it, writes its tensors, and moves
on, so peak host memory is one layer rather than the whole checkpoint.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

LOAD_FORMAT = "strata-qwen4exp-partial-load-v1"


class LoadError(RuntimeError):
    """Raised when the checkpoint does not match the configured model."""


def read_index(directory: Path) -> dict[str, str]:
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        raise LoadError(f"{directory}: model.safetensors.index.json is missing")
    return json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})


def checkpoint_name(model_name: str, weight_map: dict[str, str]) -> str | None:
    """Map a text-model parameter name onto the checkpoint's key layout."""
    if model_name.startswith("model."):
        candidate = "model.language_model." + model_name.removeprefix("model.")
        if candidate in weight_map:
            return candidate
    if model_name in weight_map:
        return model_name
    return None


def trim_config(config: Any, keep_layers: int) -> Any:
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    if keep_layers <= 0 or keep_layers > int(text.num_hidden_layers):
        raise LoadError(f"keep_layers must be in [1, {text.num_hidden_layers}]")
    text.num_hidden_layers = keep_layers
    text.layer_types = list(text.layer_types)[:keep_layers]
    text.ple_layer_ids = [index for index in list(text.ple_layer_ids or []) if index <= keep_layers]
    if hasattr(text, "mtp"):
        # The materialized checkpoint intentionally omits MTP tensors.
        text.mtp = None
    return config


def load_partial(
    model_dir: Path,
    keep_layers: int,
    *,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    config = trim_config(
        AutoConfig.from_pretrained(str(model_dir), local_files_only=True), keep_layers
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # Non-persistent config-derived buffers (RoPE frequencies) are meta
    # placeholders; rebuild those modules so they carry real values.
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    for module_name, module in list(model.named_modules()):
        if not any(buffer.device.type == "meta" for buffer in module.buffers(recurse=False)):
            continue
        try:
            fresh = type(module)(text_config)
        except Exception:  # noqa: BLE001 - only rebuildable modules are replaced
            continue
        parent_name, _, attribute = module_name.rpartition(".")
        setattr(model.get_submodule(parent_name), attribute, fresh)

    weight_map = read_index(model_dir)
    by_shard: dict[str, list[tuple[str, str]]] = {}
    missing: list[str] = []
    mapping: dict[str, str] = {}
    for name, _ in model.named_parameters():
        source = checkpoint_name(name, weight_map)
        if source is None:
            missing.append(name)
            continue
        mapping[name] = source
        by_shard.setdefault(weight_map[source], []).append((name, source))

    loaded: list[dict[str, Any]] = []
    handles: dict[str, Any] = {}
    try:
        for shard, pairs in sorted(by_shard.items()):
            handles[shard] = safe_open(str(model_dir / shard), framework="pt")
            handle = handles[shard]
            for name, source in pairs:
                tensor = handle.get_tensor(source)
                parameter = dict(model.named_parameters())[name]
                if tuple(parameter.shape) != tuple(tensor.shape):
                    raise LoadError(
                        f"{name}: model shape {tuple(parameter.shape)} != checkpoint "
                        f"{tuple(tensor.shape)} for {source}"
                    )
                # In-place writes into meta tensors are no-ops, so the real
                # tensor replaces the placeholder parameter outright.
                module_name, _, parameter_name = name.rpartition(".")
                module = model.get_submodule(module_name)
                module._parameters[parameter_name] = torch.nn.Parameter(tensor, requires_grad=False)
                loaded.append(
                    {
                        "parameter": name,
                        "source": source,
                        "shard": shard,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype).removeprefix("torch."),
                        "bytes": int(tensor.numel() * tensor.element_size()),
                    }
                )
    finally:
        for handle in handles.values():
            del handle

    # Params still on meta are a hard error: a partially materialized model
    # would silently produce garbage activations.
    meta_params = [
        name for name, parameter in model.named_parameters() if parameter.device.type == "meta"
    ]
    meta_buffers = [name for name, buffer in model.named_buffers() if buffer.device.type == "meta"]
    if meta_params or meta_buffers:
        raise LoadError(
            f"{len(meta_params)} parameters and {len(meta_buffers)} buffers remain on meta: "
            f"{(meta_params + meta_buffers)[:8]}"
        )
    model = model.to(device)
    model.eval()
    receipt = {
        "format": LOAD_FORMAT,
        "model_dir": str(model_dir),
        "keep_layers": keep_layers,
        "loaded_tensors": len(loaded),
        "loaded_bytes": sum(entry["bytes"] for entry in loaded),
        "shards_read": len(by_shard),
        "missing_parameters": missing,
        "tensors": loaded,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "non_claims": [
            "A partial prefix is not the full model and carries no quality claim.",
            "No quantization or measurement ran in load-only mode.",
        ],
    }
    return model, receipt


def quantize_partial(
    model: Any,
    *,
    model_dir: Path,
    tokenizer_dir: Path,
    calibration: Path,
    num_samples: int,
    output: Path,
    sequential_targets: list[str] | None = None,
    targets_per_subgraph: int = 1,
) -> dict[str, Any]:
    import numpy as np
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from transformers import AutoTokenizer

    matrix = np.load(calibration, mmap_mode="r")
    if num_samples <= 0 or num_samples > matrix.shape[0]:
        raise LoadError(f"num_samples must be in [1, {matrix.shape[0]}]")
    samples = np.asarray(matrix[:num_samples])
    dataset = Dataset.from_dict({"input_ids": samples.tolist()})
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model.config._name_or_path = str(tokenizer_dir)
    recipe = GPTQModifier(
        targets="Linear",
        scheme="MXFP4A16",
        ignore=["lm_head"],
        dampening_frac=0.01,
        actorder="static",
    )
    started = time.perf_counter()
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        pipeline="sequential" if sequential_targets else "independent",
        sequential_targets=sequential_targets,
        sequential_offload_device="cpu",
        sequential_targets_per_subgraph=targets_per_subgraph,
        num_calibration_samples=num_samples,
        max_seq_length=int(matrix.shape[1]),
        batch_size=1,
        shuffle_calibration_samples=False,
    )
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, save_compressed=True)
    files = sorted(path.name for path in output.iterdir() if path.is_file())
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    return {
        "elapsed_seconds": round(elapsed, 3),
        "calibration": {
            "path": str(calibration),
            "num_samples": num_samples,
            "sequence_length": int(matrix.shape[1]),
        },
        "output": str(output),
        "files": files,
        "quantization_format": config.get("quantization_config", {}).get("format"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--keep-layers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--sequential-targets",
        help="comma-separated module class names for the sequential pipeline",
    )
    parser.add_argument("--targets-per-subgraph", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        model, receipt = load_partial(args.model, args.keep_layers, device=args.device)
        if args.quantize:
            if args.calibration is None or args.output is None:
                raise LoadError("--quantize requires --calibration and --output")
            receipt["quantize"] = quantize_partial(
                model,
                model_dir=args.model,
                tokenizer_dir=args.tokenizer or args.model,
                calibration=args.calibration,
                num_samples=args.num_samples,
                output=args.output,
                sequential_targets=(
                    [item.strip() for item in args.sequential_targets.split(",") if item.strip()]
                    if args.sequential_targets
                    else None
                ),
                targets_per_subgraph=args.targets_per_subgraph,
            )
    except Exception as error:  # noqa: BLE001 - the loader reports every failure
        print(
            json.dumps(
                {
                    "format": LOAD_FORMAT,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    receipt["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "format": LOAD_FORMAT,
                "keep_layers": receipt["keep_layers"],
                "loaded_tensors": receipt["loaded_tensors"],
                "loaded_bytes": receipt["loaded_bytes"],
                "shards_read": receipt["shards_read"],
                "missing_parameters": receipt["missing_parameters"],
                "elapsed_seconds": receipt["elapsed_seconds"],
                "quantize": receipt.get("quantize"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
