#!/usr/bin/env python3
"""Stream a physically pruned Flint BF16 expert checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import flint_compression_plan as plan  # noqa: E402

ROUTER_RE = plan.ROUTER_RE
EXPERT_RE = plan.EXPERT_RE
MASK_FORMAT = "strata-flint-expert-mask-v1"


def sha256(path: Path) -> str:
    return plan.sha256(path)


def load_mask(path: Path, source_experts: int) -> tuple[int, list[list[int]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != MASK_FORMAT:
        raise ValueError(f"{path}: schema must be {MASK_FORMAT}")
    if document.get("source_experts") != source_experts:
        raise ValueError(f"{path}: source_experts must be {source_experts}")
    keep = document.get("keep_ids")
    if not isinstance(keep, list) or not keep:
        raise ValueError(f"{path}: keep_ids must be a non-empty list")
    for rows in keep:
        if not isinstance(rows, list):
            raise ValueError(f"{path}: every keep_ids row must be a list")
        if len(rows) != len(set(rows)):
            raise ValueError(f"{path}: duplicate source expert ID in a layer")
        if any(not isinstance(value, int) or not 0 <= value < source_experts for value in rows):
            raise ValueError(f"{path}: source expert ID out of range")
    keep_count = len(keep[0])
    if any(len(rows) != keep_count for rows in keep):
        raise ValueError(f"{path}: every layer must retain {keep_count} experts")
    if not 0 < keep_count < source_experts:
        raise ValueError(f"{path}: keep count must be in (0, {source_experts})")
    return keep_count, [list(rows) for rows in keep]


def validate_inventory(config: dict[str, Any], weight_map: dict[str, str]) -> Counter[str]:
    if config.get("model_type") != "flint":
        raise ValueError("source model_type is not flint")
    source_experts = config.get("num_experts")
    if not isinstance(source_experts, int) or source_experts != plan.SOURCE_EXPERTS:
        raise ValueError(f"source num_experts must be {plan.SOURCE_EXPERTS}")
    counts = plan.classify(weight_map)
    if counts != plan.EXPECTED_TENSORS:
        raise ValueError(f"unexpected tensor inventory: {dict(counts)}")
    return counts


def tokenizer_files(source: Path) -> list[str]:
    names = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(
            ("tokenizer", "generation_config", "chat_template")
        ) or path.name in {"LICENSE", "NOTICE"}:
            names.append(path.name)
    if not any(name.startswith("tokenizer") for name in names):
        raise ValueError("source has no tokenizer files")
    return names


def materialize(
    source: Path,
    mask_path: Path,
    output: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = source_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source weight_map is missing")
    validate_inventory(config, weight_map)
    keep_count, keep_rows = load_mask(mask_path, plan.SOURCE_EXPERTS)
    if len(keep_rows) != plan.ROUTED_LAYERS:
        raise ValueError(f"mask has {len(keep_rows)} layers, expected {plan.ROUTED_LAYERS}")

    dropped_experts = sum(plan.SOURCE_EXPERTS - len(rows) for rows in keep_rows)
    selected_tensors = len(weight_map) - dropped_experts * 3
    base = {
        "format": "strata-pruned-flint-bf16-v1",
        "source": str(source),
        "source_config_sha256": sha256(config_path),
        "source_index_sha256": sha256(index_path),
        "mask": str(mask_path),
        "mask_sha256": sha256(mask_path),
        "output": str(output),
        "source_experts": plan.SOURCE_EXPERTS,
        "keep_per_layer": keep_count,
        "selected_tensors": selected_tensors,
        "omitted_tensors": len(weight_map) - selected_tensors,
        "tokenizer_files": tokenizer_files(source),
        "dry_run": dry_run,
    }
    if dry_run:
        return base

    import torch  # pylint: disable=import-outside-toplevel
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel
    from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel

    output.mkdir(parents=True)
    output_map: dict[str, str] = {}
    total_size = 0
    copied = Counter()
    started = time.monotonic()
    for shard_number, shard_name in enumerate(sorted(set(weight_map.values())), 1):
        tensors: dict[str, Any] = {}
        names = sorted(name for name, target in weight_map.items() if target == shard_name)
        with safe_open(source / shard_name, framework="pt", device="cpu") as handle:
            for source_name in names:
                router = ROUTER_RE.fullmatch(source_name)
                expert = EXPERT_RE.fullmatch(source_name)
                tensor = handle.get_tensor(source_name)
                if router is not None:
                    layer = int(router.group(1))
                    rows = torch.tensor(keep_rows[layer - 2], dtype=torch.long)
                    tensor = tensor.index_select(0, rows)
                    target_name = source_name
                    copied["router"] += 1
                elif expert is not None:
                    layer = int(expert.group(1))
                    source_expert = int(expert.group(2))
                    projection = expert.group(3)
                    try:
                        target_expert = keep_rows[layer - 2].index(source_expert)
                    except ValueError:
                        del tensor
                        continue
                    target_prefix = f"model.layers.{layer}.mlp.experts.{target_expert}"
                    target_name = f"{target_prefix}.{projection}.weight"
                    copied["expert"] += 1
                else:
                    target_name = source_name
                    copied["other"] += 1
                tensors[target_name] = tensor
                total_size += tensor.numel() * tensor.element_size()
        if tensors:
            save_file(tensors, output / shard_name, metadata={"format": "pt"})
            output_map.update(dict.fromkeys(tensors, shard_name))
            del tensors
        print(
            f"[{shard_number}/{len(set(weight_map.values()))}] "
            f"{shard_name}: cumulative tensors={len(output_map)}",
            flush=True,
        )

    expected_copied = Counter(
        {
            "router": plan.ROUTED_LAYERS,
            "expert": plan.ROUTED_LAYERS * keep_count * 3,
            "other": selected_tensors - plan.ROUTED_LAYERS - plan.ROUTED_LAYERS * keep_count * 3,
        }
    )
    if copied != expected_copied:
        raise ValueError(f"unexpected materialized counts: {dict(copied)}")
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": dict(sorted(output_map.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    target_config = dict(config)
    target_config["num_experts"] = keep_count
    target_config["original_num_experts"] = plan.SOURCE_EXPERTS
    (output / "config.json").write_text(
        json.dumps(target_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    expert_map = {
        "schema": "strata-flint-expert-id-map-v1",
        "source_expert_count": plan.SOURCE_EXPERTS,
        "physical_expert_count": keep_count,
        "physical_to_source": keep_rows,
    }
    (output / "expert-id-map.json").write_text(
        json.dumps(expert_map, indent=2) + "\n", encoding="utf-8"
    )
    for name in base["tokenizer_files"]:
        (output / name).write_bytes((source / name).read_bytes())
    receipt = {
        **base,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output_tensors": len(output_map),
        "output_total_size": total_size,
        "materialized_tensor_counts": dict(copied),
        "expert_id_map_sha256": sha256(output / "expert-id-map.json"),
        "non_claims": [
            "No model forward, RCO quality search, calibration, quantization, or pack ran.",
            "The input mask is authoritative; materialization does not validate its quality.",
        ],
    }
    (output / "materialization.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = materialize(args.source, args.mask, args.output, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
