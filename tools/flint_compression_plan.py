#!/usr/bin/env python3
"""Plan exact Flint expert budgets for the separate RCO and GSQ lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PLAN_FORMAT = "strata-flint-compression-plan-v1"
ROUTED_LAYERS = 47
SOURCE_EXPERTS = 256
ACTIVE_EXPERTS = 6
MOE_INTERMEDIATE = 512
HIDDEN_SIZE = 2048
EXPERT_PARAMETERS = 3 * MOE_INTERMEDIATE * HIDDEN_SIZE
EXPECTED_TENSORS = Counter(
    {
        "routed_experts": ROUTED_LAYERS * SOURCE_EXPERTS * 3,
        "routers": ROUTED_LAYERS,
        "shared_experts": ROUTED_LAYERS * 3,
    }
)
ROUTER_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$")
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
SHARED_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.shared_expert\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(weight_map: dict[str, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    routed_layers: set[int] = set()
    for name in weight_map:
        if EXPERT_RE.fullmatch(name):
            counts["routed_experts"] += 1
            routed_layers.add(int(EXPERT_RE.fullmatch(name).group(1)))  # type: ignore[union-attr]
        elif ROUTER_RE.fullmatch(name):
            layer = int(ROUTER_RE.fullmatch(name).group(1))  # type: ignore[union-attr]
            if layer in {0, 1}:
                raise ValueError(f"dense layer has a routed router: {name}")
            routed_layers.add(layer)
            counts["routers"] += 1
        elif SHARED_RE.fullmatch(name):
            layer = int(SHARED_RE.fullmatch(name).group(1))  # type: ignore[union-attr]
            if layer in {0, 1}:
                raise ValueError(f"dense layer has a shared expert: {name}")
            routed_layers.add(layer)
            counts["shared_experts"] += 1
    if routed_layers != set(range(2, 49)):
        missing = sorted(set(range(2, 49)) - routed_layers)
        raise ValueError(f"missing routed layer metadata: {missing[:4]}")
    if counts != EXPECTED_TENSORS:
        raise ValueError(f"unexpected Flint tensor counts: {dict(counts)}")
    return counts


def lane(keep: int) -> dict[str, Any]:
    if not 0 < keep <= SOURCE_EXPERTS:
        raise ValueError(f"keep must be in (0, {SOURCE_EXPERTS}]")
    removed = ROUTED_LAYERS * (SOURCE_EXPERTS - keep)
    retained = ROUTED_LAYERS * keep
    return {
        "keep_per_layer": keep,
        "prune_per_layer": SOURCE_EXPERTS - keep,
        "retained_expert_slots": retained,
        "pruned_expert_slots": removed,
        "bf16_routed_bytes": retained * EXPERT_PARAMETERS * 2,
        "bf16_removed_bytes": removed * EXPERT_PARAMETERS * 2,
        "mxfp4_routed_bytes": retained * EXPERT_PARAMETERS * 17 // 32,
        "mxfp4_removed_bytes": removed * EXPERT_PARAMETERS * 17 // 32,
    }


def make_plan(config_path: Path, index_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "flint":
        raise ValueError("model_type is not flint")
    if config.get("num_experts") != SOURCE_EXPERTS:
        raise ValueError("source num_experts is not 256")
    if config.get("num_experts_per_tok") != ACTIVE_EXPERTS:
        raise ValueError("source num_experts_per_tok is not 6")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    counts = classify(weight_map)
    return {
        "format": PLAN_FORMAT,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "index": str(index_path),
        "index_sha256": sha256(index_path),
        "index_tensors": len(weight_map),
        "tensor_counts": dict(counts),
        "geometry": {
            "decoder_layers": 49,
            "dense_layers": 2,
            "routed_layers": ROUTED_LAYERS,
            "source_experts": SOURCE_EXPERTS,
            "active_experts": ACTIVE_EXPERTS,
            "expert_parameters": EXPERT_PARAMETERS,
        },
        "lanes": {
            "rco_primary": lane(192),
            "sensitivity": [lane(224), lane(192), lane(160)],
        },
        "method_order": {
            "rco": [
                "full-source sigmoid-router Gumbel-STE mask search",
                "hard-mask full-model KL validation",
                "physical BF16 materialization",
                "layer-sequential GPTQ-MXFP4 recalibration",
                "MLX pack and correctness/performance gates",
            ],
            "gsq": [
                "physical BF16 materialization from an already-selected mask",
                "Flint GSQ model wrapper",
                "GSQ Gumbel-Softmax scalar quantization",
                "compressed-tensors or MLX-compatible export",
                "same correctness/performance gates",
            ],
        },
        "non_claims": [
            "This plan reads metadata only and does not select a mask.",
            "GSQ replaces GPTQ optimization; it is not a stage after RCO.",
            "No quality or throughput effect is claimed before live gates run.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = make_plan(args.config, args.index)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
