#!/usr/bin/env python3
"""Verify that a fused MoE artifact matches its per-expert source exactly.

The check runs on the compute host: it decompresses both layouts with
compressed-tensors' own MXFP4 decompressor and compares the dequantized
values.  Equality of the dequantized tensors proves that the fused layout is a
semantics-preserving rearrangement, not merely a byte shuffle.

The tool proves nothing about quality: it never compares against an
unquantized reference and never runs a model forward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VERIFY_FORMAT = "strata-fused-moe-artifact-verification-v1"


class VerificationError(RuntimeError):
    """Raised when the fused artifact does not match its source."""


def read_index(directory: Path) -> dict[str, str]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
    single = directory / "model.safetensors"
    if not single.is_file():
        raise VerificationError(f"{directory}: no safetensors shard or index found")
    from safetensors import safe_open

    with safe_open(str(single), framework="pt") as handle:
        names = list(handle.keys())
    return {name: single.name for name in names if name != "__metadata__"}


def load_tensor(directory: Path, weight_map: dict[str, str], name: str) -> Any:
    from safetensors import safe_open

    shard = weight_map[name]
    with safe_open(str(directory / shard), framework="pt") as handle:
        return handle.get_tensor(name)


def build_scheme(config: dict[str, Any]) -> Any:
    import torch
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    groups = config.get("quantization_config", {}).get("config_groups", {})
    weights = None
    for group in groups.values():
        weights = group.get("weights", {})
        break
    weights = weights or {}
    args = QuantizationArgs(
        num_bits=int(weights.get("num_bits", 4)),
        type=weights.get("type", "float"),
        symmetric=bool(weights.get("symmetric", True)),
        strategy=weights.get("strategy", "group"),
        group_size=int(weights.get("group_size", 32)),
        dynamic=bool(weights.get("dynamic", False)),
        scale_dtype=torch.uint8 if weights.get("scale_dtype") == "torch.uint8" else None,
        block_structure=weights.get("block_structure"),
    )
    return QuantizationScheme(targets=["Linear"], weights=args)


def decompress(state: dict[str, Any], scheme: Any) -> Any:
    from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor

    return MXFP4PackedCompressor.decompress(dict(state), scheme)["weight"]


def verify(
    linearized: Path,
    fused: Path,
    *,
    max_experts: int | None = None,
) -> dict[str, Any]:
    config = json.loads((linearized / "config.json").read_text(encoding="utf-8"))
    scheme = build_scheme(config)
    linear_map = read_index(linearized)
    fused_map = read_index(fused)

    stacks: dict[str, list[str]] = {}
    for name in linear_map:
        parts = name.split(".experts.")
        if len(parts) != 2:
            continue
        prefix, tail = parts
        prefix = f"{prefix}.experts"
        index = int(tail.split(".", 1)[0])
        stacks.setdefault(prefix, [])
        if index not in stacks[prefix]:
            stacks[prefix].append(index)

    checked = 0
    compared = 0
    for prefix in sorted(stacks):
        indices = sorted(stacks[prefix])
        if max_experts is not None:
            indices = indices[:max_experts]
        gate_up = f"{prefix}.gate_up_proj"
        down = f"{prefix}.down_proj"
        if f"{gate_up}.weight_packed" not in fused_map:
            raise VerificationError(f"{fused}: missing fused tensor {gate_up}.weight_packed")
        if f"{down}.weight_packed" not in fused_map:
            raise VerificationError(f"{fused}: missing fused tensor {down}.weight_packed")
        # compressed-tensors' generic decompressor accepts 2D packed tensors
        # only, so each fused expert slice is decompressed on its own.
        fused_gate_up_packed = load_tensor(fused, fused_map, f"{gate_up}.weight_packed")
        fused_gate_up_scale = load_tensor(fused, fused_map, f"{gate_up}.weight_scale")
        fused_down_packed = load_tensor(fused, fused_map, f"{down}.weight_packed")
        fused_down_scale = load_tensor(fused, fused_map, f"{down}.weight_scale")
        for index in indices:
            gate = decompress(
                {
                    "weight_packed": load_tensor(
                        linearized,
                        linear_map,
                        f"{prefix}.{index}.gate_proj.weight_packed",
                    ),
                    "weight_scale": load_tensor(
                        linearized,
                        linear_map,
                        f"{prefix}.{index}.gate_proj.weight_scale",
                    ),
                },
                scheme,
            )
            up = decompress(
                {
                    "weight_packed": load_tensor(
                        linearized,
                        linear_map,
                        f"{prefix}.{index}.up_proj.weight_packed",
                    ),
                    "weight_scale": load_tensor(
                        linearized,
                        linear_map,
                        f"{prefix}.{index}.up_proj.weight_scale",
                    ),
                },
                scheme,
            )
            source_gate_up = torch_cat([gate, up])
            fused_gate_up = decompress(
                {
                    "weight_packed": fused_gate_up_packed[index],
                    "weight_scale": fused_gate_up_scale[index],
                },
                scheme,
            )
            if not torch_equal(source_gate_up, fused_gate_up):
                raise VerificationError(f"{prefix}: expert {index} gate/up mismatch after fusion")
            source_down = decompress(
                {
                    "weight_packed": load_tensor(
                        linearized,
                        linear_map,
                        f"{prefix}.{index}.down_proj.weight_packed",
                    ),
                    "weight_scale": load_tensor(
                        linearized,
                        linear_map,
                        f"{prefix}.{index}.down_proj.weight_scale",
                    ),
                },
                scheme,
            )
            fused_down = decompress(
                {
                    "weight_packed": fused_down_packed[index],
                    "weight_scale": fused_down_scale[index],
                },
                scheme,
            )
            if not torch_equal(source_down, fused_down):
                raise VerificationError(f"{prefix}: expert {index} down_proj mismatch after fusion")
            compared += 2
        checked += len(indices)
    return {
        "format": VERIFY_FORMAT,
        "linearized": str(linearized),
        "fused": str(fused),
        "expert_stacks": len(stacks),
        "experts_compared": checked,
        "tensors_compared": compared,
        "scheme": config.get("quantization_config", {}).get("format"),
        "non_claims": [
            "The comparison proves layout equivalence of the quantized tensors only.",
            "No unquantized reference, quality metric, or model forward was used.",
            "This does not prove that an inference engine can serve the fused artifact.",
        ],
    }


def torch_cat(tensors: list[Any]) -> Any:
    import torch

    return torch.cat(tensors, dim=0)


def torch_equal(left: Any, right: Any) -> bool:
    import torch

    if left.shape != right.shape:
        return False
    return bool(torch.equal(left, right))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--linearized", required=True, type=Path)
    parser.add_argument("--fused", required=True, type=Path)
    parser.add_argument("--max-experts", type=int)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = verify(args.linearized, args.fused, max_experts=args.max_experts)
    except Exception as error:  # noqa: BLE001 - the verifier reports every failure
        print(
            json.dumps(
                {
                    "format": VERIFY_FORMAT,
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
