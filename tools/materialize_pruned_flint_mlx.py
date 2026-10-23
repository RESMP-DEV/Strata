#!/usr/bin/env python3
"""Slice a calibrated Flint MLX MXFP4 model to a stored expert mask."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

ROUTED_TENSOR = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.(?:gate|switch_mlp\.(?:gate_proj|up_proj|down_proj))"
    r"\.(?:weight|scales|biases|bias)$"
)
COPY_FILES = (
    "LICENSE",
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mask(path: Path) -> tuple[int, list[list[int]]]:
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
    return source_experts, rows


def map_physical_rows(
    num_layers: int, dense_layers: list[int], keep_rows: list[list[int]]
) -> dict[int, list[int]]:
    routed = [physical for physical in range(num_layers) if physical not in set(dense_layers)]
    if len(routed) != len(keep_rows):
        raise ValueError(f"model has {len(routed)} routed layers but mask has {len(keep_rows)}")
    return dict(zip(routed, keep_rows, strict=True))


def routed_layer(tensor_name: str) -> int | None:
    match = ROUTED_TENSOR.match(tensor_name)
    return int(match.group(1)) if match else None


def materialize(
    source: Path, mask_path: Path, output: Path, dry_run: bool = False
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    source_experts, keep_rows = load_mask(mask_path)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "flint":
        raise ValueError("source model_type is not flint")
    if config.get("num_experts") != source_experts:
        raise ValueError("mask source_experts does not match the source config")
    rows = map_physical_rows(
        int(config["num_hidden_layers"]),
        list(config.get("mlp_only_layers") or []),
        keep_rows,
    )
    keep = len(keep_rows[0])
    if dry_run:
        return {
            "source": str(source),
            "mask": str(mask_path),
            "keep_per_layer": keep,
            "routed_layers": len(rows),
            "output": str(output),
            "dry_run": True,
        }

    output.mkdir(parents=True)
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    shard_receipts = []
    sliced = 0
    for shard_name in shard_names:
        import mlx.core as mx

        shard_path = source / shard_name
        weights = mx.load(str(shard_path))
        for name in weights:
            physical = routed_layer(name)
            if physical is None:
                continue
            if physical not in rows:
                raise ValueError(f"tensor on non-routed layer: {name}")
            indices = mx.array(rows[physical], dtype=mx.int32)
            if weights[name].shape[0] != source_experts:
                raise ValueError(
                    f"{name}: expert axis {weights[name].shape[0]} != {source_experts}"
                )
            weights[name] = mx.take(weights[name], indices, axis=0)
            sliced += 1
        destination = output / shard_name
        tmp = destination.with_suffix(".tmp.safetensors")
        mx.save_safetensors(str(tmp), weights, metadata={"format": "mlx"})
        tmp.replace(destination)
        shard_receipts.append(
            {
                "name": shard_name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
        del weights

    config["num_experts"] = keep
    config["original_num_experts"] = source_experts
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(source / "model.safetensors.index.json", output)
    for name in COPY_FILES:
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, output / name)

    expert_map = {
        "schema": "strata-flint-expert-id-map-v1",
        "source_experts": source_experts,
        "kept_per_layer": keep,
        "layers": {
            str(physical): {
                "source_ids": keep_ids,
                "target_ids": list(range(keep)),
            }
            for physical, keep_ids in rows.items()
        },
    }
    (output / "expert-id-map.json").write_text(
        json.dumps(expert_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(mask_path, output / "pruning-mask.json")
    receipt = {
        "schema": "strata-pruned-flint-mlx-v1",
        "source": str(source),
        "mask": str(mask_path),
        "mask_sha256": sha256(mask_path),
        "source_experts": source_experts,
        "kept_per_layer": keep,
        "routed_layers": len(rows),
        "sliced_tensors": sliced,
        "shards": shard_receipts,
        "expert_id_map_sha256": sha256(output / "expert-id-map.json"),
        "payload_bytes": sum(row["bytes"] for row in shard_receipts),
        "non_claims": [
            "Slicing preserves the source GPTQ-MXFP4 q/scales; it does not recalibrate.",
            "Runtime quality and packed-model equivalence must be measured separately.",
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
    result = materialize(args.source, args.mask, args.output, args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
