#!/usr/bin/env python3
"""Probe the llmcompressor MoE path on a tiny random Qwen4Exp model.

This runs on the compute host inside the pinned quantization environment.  It
uses no checkpoint, no corpus, and no GPU: every weight is randomly
initialized, so the probe proves machinery, never quality.  Stages:

1. protocol  - are ``Qwen4ExpTextExperts`` instances recognized as fused experts?
2. linearize- does llmcompressor's MoE linearization split them per expert?
3. parity   - does the linearized forward match the fused forward?
4. repack   - does repacking restore the fused module and outputs?
5. quantize - does a real MXFP4A16 GPTQ pass produce compressed-tensors output?

The receipt records measured counts, tolerances, and non-claims.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROBE_FORMAT = "strata-qwen4exp-moe-probe-v1"


class ProbeError(RuntimeError):
    """Raised when a probe stage fails."""


def tiny_text_config(*, with_ple: bool, mixed_attention: bool = False) -> Any:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=64,
        indexer_compress_ratio=4,
        linear_key_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_value_head_dim=16,
        max_position_embeddings=128,
        eos_token_id=0,
        bos_token_id=0,
        layer_types=(
            # PLE is only valid on linear-attention layers, and the real
            # Flash-Next model places PLE on the second layer.
            ["linear_attention", "linear_attention"]
            if mixed_attention
            else ["full_attention", "full_attention"]
        ),
        ple_layer_ids=[2] if with_ple else [],
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=64,
        split_ngram_parts=4,
        ple_embed_dim=64,
    )
    return config


def build_model(config: Any, dtype: Any) -> Any:
    import torch
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    torch.manual_seed(17)
    # llmcompressor derives the per-expert linear dtype from ``config.dtype``;
    # the Flash-Next checkpoint declares bfloat16, so the probe must too.
    config.dtype = dtype
    model = Qwen4ExpForCausalLM(config)
    model = model.to(dtype=dtype)
    model.eval()
    return model


def expert_modules(model: Any) -> list[tuple[str, Any]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".experts") and hasattr(module, "gate_up_proj")
    ]


def expected_expert_linear_count(num_layers: int, num_experts: int) -> int:
    """Three quantized linears per routed expert per layer."""
    return num_layers * num_experts * 3


def _environment() -> dict[str, str]:
    import importlib.metadata

    import torch
    import transformers

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "llmcompressor": importlib.metadata.version("llmcompressor"),
        "compressed_tensors": importlib.metadata.version("compressed-tensors"),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
    from llmcompressor.modeling.moe.linearize import linearize_moe, repack_moe

    stages: dict[str, Any] = {}
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if args.ple_structure_only:
        return _ple_structure_stage(args, dtype)
    config = tiny_text_config(with_ple=args.with_ple, mixed_attention=args.mixed_attention)
    device = torch.device(args.device)
    model = build_model(config, dtype).to(device)
    input_ids = torch.arange(args.samples * args.sequence_length, dtype=torch.long)
    input_ids = (input_ids % config.vocab_size).reshape(args.samples, args.sequence_length)
    input_ids = input_ids.to(device)

    experts = expert_modules(model)
    if not experts:
        raise ProbeError("no Qwen4Exp expert modules were found in the tiny model")
    recognized = [
        {"name": name, "recognized": isinstance(module, FusedExpertsProtocol)}
        for name, module in experts
    ]
    if not all(entry["recognized"] for entry in recognized):
        raise ProbeError(f"expert modules are not recognized as fused experts: {recognized}")
    stages["protocol"] = {
        "expert_modules": len(experts),
        "recognized": True,
        "gate_up_shape": list(experts[0][1].gate_up_proj.shape),
        "down_shape": list(experts[0][1].down_proj.shape),
        "num_experts": int(experts[0][1].num_experts),
    }

    with torch.no_grad():
        fused_logits = model(input_ids=input_ids, use_cache=False).logits.float()

    linearize_moe(model)
    linearized = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LinearExperts2D)
    ]
    if len(linearized) != len(experts):
        raise ProbeError(
            f"linearization produced {len(linearized)} expert stacks for "
            f"{len(experts)} expert modules"
        )
    linearized_prefixes = tuple(f"{stack_name}." for stack_name, _ in linearized)
    expert_linears = sum(
        1
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.startswith(linearized_prefixes)
    )
    expected = expected_expert_linear_count(config.num_hidden_layers, config.num_experts)
    if expert_linears < expected:
        raise ProbeError(
            f"expected at least {expected} per-expert linears after linearization, "
            f"found {expert_linears}"
        )
    with torch.no_grad():
        linear_logits = model(input_ids=input_ids, use_cache=False).logits.float()
    parity = float((fused_logits - linear_logits).abs().max())
    if parity > args.tolerance:
        raise ProbeError(
            f"linearized forward differs from fused forward by {parity} > {args.tolerance}"
        )
    stages["linearize"] = {
        "linearized_stacks": len(linearized),
        "expert_linear_modules": expert_linears,
        "expected_expert_linear_modules": expected,
        "max_abs_logit_delta_vs_fused": parity,
        "tolerance": args.tolerance,
    }

    repack_moe(model)
    repacked = expert_modules(model)
    if len(repacked) != len(experts):
        raise ProbeError("repack did not restore the fused expert modules")
    with torch.no_grad():
        repacked_logits = model(input_ids=input_ids, use_cache=False).logits.float()
    repack_delta = float((fused_logits - repacked_logits).abs().max())
    if repack_delta > args.tolerance:
        raise ProbeError(
            f"repacked forward differs from fused forward by {repack_delta} > {args.tolerance}"
        )
    stages["repack"] = {
        "fused_expert_modules": len(repacked),
        "max_abs_logit_delta_vs_fused": repack_delta,
        "tolerance": args.tolerance,
    }

    if args.quantize:
        stages["quantize"] = _quantize_stage(config, dtype, args)
    return {
        "format": PROBE_FORMAT,
        "stages": stages,
        "environment": _environment(),
        "non_claims": [
            "Every weight is randomly initialized; the probe proves machinery, not quality.",
            "No Flash-Next checkpoint, calibration corpus, GPU, or benchmark was used.",
            "A passing probe does not prove that a full 288-expert checkpoint fits in RAM.",
        ],
    }


def _ple_structure_stage(args: argparse.Namespace, dtype: Any) -> dict[str, Any]:
    """Inspect the PLE n-gram embedding structure without running a forward.

    The linear-attention layers require CUDA kernels, so this stage builds the
    module only.  It reports the shard layout that a full run must stream.
    """
    import torch
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    config = tiny_text_config(with_ple=True, mixed_attention=True)
    torch.manual_seed(17)
    model = Qwen4ExpForCausalLM(config).to(dtype=dtype)
    ple_modules = [(name, module) for name, module in model.named_modules() if ".ple." in name]
    shards = [
        (name, module)
        for name, module in model.named_modules()
        if "ngram_embedding" in name and hasattr(module, "weight")
    ]
    ple_bytes = sum(
        parameter.numel() * parameter.element_size()
        for _, module in ple_modules
        for parameter in module.parameters(recurse=True)
    )
    return {
        "format": PROBE_FORMAT,
        "stages": {
            "ple_structure": {
                "ple_layer_ids": list(config.ple_layer_ids),
                "ngram_vocab_size_base": config.ngram_vocab_size_base,
                "ngram_size": config.ngram_size,
                "heads_per_ngram": config.heads_per_ngram,
                "ple_embed_dim": config.ple_embed_dim,
                "split_ngram_parts": config.split_ngram_parts,
                "ple_modules": len(ple_modules),
                "ngram_shards": len(shards),
                "shard_sample": sorted(name for name, _ in shards)[:4],
                "ple_bytes": ple_bytes,
                "dtype": str(dtype),
            }
        },
        "environment": _environment(),
        "non_claims": [
            "The PLE structure was inspected on a tiny random config; no Flash-Next "
            "checkpoint was loaded.",
            "No forward pass ran; linear-attention kernels require CUDA.",
            "This stage proves the module layout, not serving feasibility.",
        ],
    }


def _quantize_stage(config: Any, dtype: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier

    model = build_model(config, dtype).to(args.device)
    generator = torch.Generator().manual_seed(23)
    samples = torch.randint(
        0,
        config.vocab_size,
        (args.quant_samples, args.quant_sequence_length),
        generator=generator,
    )
    dataset = Dataset.from_dict({"input_ids": samples.tolist()})
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    recipe = GPTQModifier(
        targets="Linear",
        scheme="MXFP4A16",
        ignore=["lm_head"],
        dampening_frac=0.01,
        actorder="static",
    )
    started = time.perf_counter()
    if args.tokenizer:
        # llmcompressor resolves the calibration processor from the model path,
        # exactly like the pinned 27B recipe does.
        model.config._name_or_path = str(args.tokenizer)
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        pipeline=args.pipeline,
        num_calibration_samples=args.quant_samples,
        max_seq_length=args.quant_sequence_length,
        batch_size=1,
        shuffle_calibration_samples=False,
    )
    elapsed = time.perf_counter() - started
    output = Path(args.output or tempfile.mkdtemp(prefix="qwen4exp-moe-probe-"))
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, save_compressed=True)
    files = sorted(path.name for path in output.iterdir() if path.is_file())
    saved_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    quantization_config = saved_config.get("quantization_config", {})
    shard = next((path for path in output.glob("*.safetensors")), None)
    tensor_names: list[str] = []
    if shard is not None:
        from safetensors import safe_open

        with safe_open(str(shard), framework="pt") as handle:
            tensor_names = list(handle.keys())
    expert_tensors = [name for name in tensor_names if ".experts." in name]
    return {
        "output": str(output),
        "files": files,
        "scheme": quantization_config.get("scheme"),
        "quant_method": quantization_config.get("quant_method"),
        "tensor_count": len(tensor_names),
        "expert_tensor_count": len(expert_tensors),
        "expert_tensor_sample": sorted(expert_tensors)[:6],
        "elapsed_seconds": round(elapsed, 3),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--with-ple", action="store_true")
    parser.add_argument(
        "--mixed-attention",
        action="store_true",
        help="use the linear-attention layer type, which requires CUDA kernels",
    )
    parser.add_argument(
        "--ple-structure-only",
        action="store_true",
        help="inspect the PLE module layout without running a forward",
    )
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument(
        "--pipeline",
        default="independent",
        choices=("basic", "datafree", "independent", "sequential"),
    )
    parser.add_argument("--quant-samples", type=int, default=4)
    parser.add_argument("--quant-sequence-length", type=int, default=64)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="tokenizer directory used to build the oneshot processor",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_probe(args)
    except Exception as error:  # noqa: BLE001 - the probe reports every failure
        print(
            json.dumps(
                {
                    "format": PROBE_FORMAT,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if not args.keep_output and args.output:
        shutil.rmtree(args.output, ignore_errors=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
