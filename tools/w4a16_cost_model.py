#!/usr/bin/env python3
"""Compute exact MXFP4A16 deployment bytes for a Qwen4Exp checkpoint.

The cost model is derived from the model's real module geometry, not from a
nominal "4 bits per parameter" guess: every quantized ``nn.Linear`` contributes
packed int32 words plus one e8m0 scale byte per 32-element group, and fused
expert parameters are costed per expert.  The model is instantiated on the
``meta`` device, so no checkpoint weight is read into RAM.

The output JSON is the ``cost_artifact`` + ``tensor_costs`` input that
``tools/quant_candidate_plan.py`` requires for a W4A16-only plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

COST_FORMAT = "strata-mxfp4a16-cost-v1"
PACK_BLOCK = 8  # 4-bit values per int32 word
SCALE_GROUP = 32


class CostError(ValueError):
    """Raised when the checkpoint geometry cannot be costed exactly."""


def linear_bytes(out_features: int, in_features: int) -> tuple[int, int]:
    """Return (packed weight bytes, scale bytes) for one MXFP4 linear."""
    if out_features <= 0 or in_features <= 0:
        raise CostError(f"invalid linear shape {out_features}x{in_features}")
    packed = out_features * ((in_features + PACK_BLOCK - 1) // PACK_BLOCK) * 4
    scales = out_features * ((in_features + SCALE_GROUP - 1) // SCALE_GROUP)
    return packed, scales


def fused_expert_bytes(num_experts: int, out_features: int, in_features: int) -> tuple[int, int]:
    packed, scales = linear_bytes(out_features, in_features)
    return packed * num_experts, scales * num_experts


def ignored_module(name: str) -> bool:
    return name == "lm_head" or name.endswith(".mlp.gate")


def build_cost(model_dir: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    rows: list[dict[str, Any]] = []
    expert_modules = 0
    for name, module in model.named_modules():
        if ignored_module(name):
            continue
        if isinstance(module, torch.nn.Linear):
            out_features, in_features = module.weight.shape
            packed, scales = linear_bytes(out_features, in_features)
            rows.append(
                {
                    "tensor": f"{name}.weight",
                    "kind": "linear",
                    "shape": [out_features, in_features],
                    "packed_bytes": packed,
                    "scale_bytes": scales,
                    "bytes": packed + scales,
                }
            )
        elif name.endswith(".experts") and hasattr(module, "gate_up_proj"):
            expert_modules += 1
            gate_up = module.gate_up_proj
            down = module.down_proj
            num_experts = int(gate_up.shape[0])
            gate_packed, gate_scales = fused_expert_bytes(
                num_experts, int(gate_up.shape[1]), int(gate_up.shape[2])
            )
            down_packed, down_scales = fused_expert_bytes(
                num_experts, int(down.shape[1]), int(down.shape[2])
            )
            rows.append(
                {
                    "tensor": f"{name}.gate_up_proj",
                    "kind": "fused_experts",
                    "shape": list(gate_up.shape),
                    "packed_bytes": gate_packed,
                    "scale_bytes": gate_scales,
                    "bytes": gate_packed + gate_scales,
                }
            )
            rows.append(
                {
                    "tensor": f"{name}.down_proj",
                    "kind": "fused_experts",
                    "shape": list(down.shape),
                    "packed_bytes": down_packed,
                    "scale_bytes": down_scales,
                    "bytes": down_packed + down_scales,
                }
            )
    if not rows:
        raise CostError(f"{model_dir}: no quantizable modules found")
    weight_bytes = sum(row["bytes"] for row in rows)
    packed_bytes = sum(row["packed_bytes"] for row in rows)
    scale_bytes = sum(row["scale_bytes"] for row in rows)
    return {
        "format": COST_FORMAT,
        "model_dir": str(model_dir),
        "model_type": config.model_type,
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0)),
        "num_experts": int(getattr(config, "num_experts", 0)),
        "expert_modules": expert_modules,
        "tensor_count": len(rows),
        "deployment_weight_bytes": weight_bytes,
        "packed_weight_bytes": packed_bytes,
        "scale_bytes": scale_bytes,
        "rows": rows,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "non_claims": [
            "These are planned deployment bytes from module geometry, not a packed artifact.",
            "No weight was dequantized, no model forward ran, and no quality metric exists.",
            "Non-Linear and ignored modules (embeddings, PLE, routers, lm_head) are excluded.",
        ],
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_cost(args.model)
    except Exception as error:  # noqa: BLE001 - the planner reports the failure
        print(
            json.dumps(
                {
                    "format": COST_FORMAT,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "format": COST_FORMAT,
                "output": str(args.output),
                "sha256": sha256_text(text),
                "tensor_count": payload["tensor_count"],
                "expert_modules": payload["expert_modules"],
                "deployment_weight_bytes": payload["deployment_weight_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
