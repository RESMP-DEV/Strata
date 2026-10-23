#!/usr/bin/env python3
"""Fuse per-expert compressed tensors into the HF-native fused MoE layout.

llmcompressor's MoE linearization writes one compressed tensor set per routed
expert (``mlp.experts.<index>.gate_proj.weight_packed`` and friends).  Native
Qwen4Exp checkpoints instead store fused 3D expert tensors
(``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj``).  This tool performs
that fusion at the file level, so it never needs the whole model in RAM.

Semantics mirror upstream llm-compressor ``main``
(``llmcompressor/modeling/moe/pack_helpers.py``): gate and up are concatenated
along the out-feature axis and the experts are stacked in expert order.  The
transformation is a byte-exact rearrangement.

The tool is dependency-free on purpose: it reads and writes safetensors files
directly, so the control host needs no torch, numpy, or safetensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from pathlib import Path
from typing import Any

REPACK_FORMAT = "strata-qwen4exp-moe-repack-v1"
DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<index>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)"
    r"\.(?P<suffix>[A-Za-z0-9_]+)$"
)
PASSTHROUGH_FILES = (
    "config.json",
    "generation_config.json",
    "recipe.yaml",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
)


class RepackError(ValueError):
    """Raised when the source artifact does not match the fused MoE contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise RepackError(f"{path}: truncated safetensors header length")
        header_length = struct.unpack("<Q", raw)[0]
        header = json.loads(handle.read(header_length).decode("utf-8"))
    return header, 8 + header_length


def read_safetensors_index(directory: Path) -> dict[str, str]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        return {name: shard for name, shard in weight_map.items()}
    single = directory / "model.safetensors"
    if not single.is_file():
        raise RepackError(f"{directory}: no safetensors shard or index found")
    header, _ = read_safetensors_header(single)
    return {name: single.name for name in header if name != "__metadata__"}


def element_size(dtype: str) -> int:
    size = DTYPE_SIZES.get(dtype)
    if size is None:
        raise RepackError(f"unsupported safetensors dtype {dtype!r}")
    return size


def tensor_bytes(directory: Path, shard: str, name: str) -> tuple[bytes, dict[str, Any]]:
    header, data_start = read_safetensors_header(directory / shard)
    entry = header.get(name)
    if entry is None:
        raise RepackError(f"{shard}: tensor {name} is not present")
    begin, end = entry["data_offsets"]
    with (directory / shard).open("rb") as handle:
        handle.seek(data_start + begin)
        payload = handle.read(end - begin)
    expected = element_size(entry["dtype"])
    for dimension in entry["shape"]:
        expected *= dimension
    if len(payload) != expected:
        raise RepackError(
            f"{shard}:{name}: payload {len(payload)} bytes does not match shape "
            f"{entry['shape']} of {entry['dtype']}"
        )
    return payload, entry


def group_experts(
    weight_map: dict[str, str],
) -> tuple[dict[str, dict[str, dict[str, str]]], list[str]]:
    """Group per-expert tensor names by expert stack, projection, and suffix."""
    groups: dict[str, dict[str, dict[str, str]]] = {}
    plain: list[str] = []
    for name in weight_map:
        match = EXPERT_PATTERN.match(name)
        if match is None:
            plain.append(name)
            continue
        prefix = match.group("prefix")
        index = int(match.group("index"))
        projection = match.group("proj")
        suffix = match.group("suffix")
        stack = groups.setdefault(prefix, {})
        stack.setdefault(f"{index}.{projection}", {})[suffix] = name
    return groups, plain


def stack_sort_key(prefix: str) -> tuple[int, str]:
    match = re.search(r"layers\.(\d+)", prefix)
    return (int(match.group(1)) if match else -1, prefix)


def plan_stack(prefix: str, stack: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Describe the fused tensors implied by one expert stack."""
    indices = sorted({int(key.split(".", 1)[0]) for key in stack})
    projections = {key.split(".", 1)[1] for key in stack}
    unknown = projections - {"gate_proj", "up_proj", "down_proj"}
    if unknown:
        raise RepackError(f"{prefix}: unexpected projections {sorted(unknown)}")
    if "gate_proj" in projections and "up_proj" not in projections:
        raise RepackError(f"{prefix}: gate_proj is present without up_proj")
    suffixes = sorted({suffix for entry in stack.values() for suffix in entry})
    plan: list[dict[str, Any]] = []
    for suffix in suffixes:
        if "gate_proj" in projections and "up_proj" in projections:
            sources = []
            for index in indices:
                for projection in ("gate_proj", "up_proj"):
                    entry = stack.get(f"{index}.{projection}", {})
                    if suffix not in entry:
                        raise RepackError(
                            f"{prefix}: expert {index} {projection} is missing {suffix}"
                        )
                    sources.append(entry[suffix])
            plan.append(
                {
                    "stack": prefix,
                    "kind": "gate_up",
                    "suffix": suffix,
                    "target": f"{prefix}.gate_up_proj.{suffix}",
                    "num_experts": len(indices),
                    "sources": sources,
                }
            )
        if "down_proj" in projections:
            sources = []
            for index in indices:
                entry = stack.get(f"{index}.down_proj", {})
                if suffix not in entry:
                    raise RepackError(f"{prefix}: expert {index} down_proj is missing {suffix}")
                sources.append(entry[suffix])
            plan.append(
                {
                    "stack": prefix,
                    "kind": "down",
                    "suffix": suffix,
                    "target": f"{prefix}.down_proj.{suffix}",
                    "num_experts": len(indices),
                    "sources": sources,
                }
            )
    return plan


def fuse_stack_tensor(
    directory: Path,
    weight_map: dict[str, str],
    entry: dict[str, Any],
) -> tuple[bytes, str, list[int]]:
    """Concatenate gate/up and stack experts into one fused byte block."""
    kind = entry["kind"]
    blocks: list[bytes] = []
    shapes: list[list[int]] = []
    dtypes: set[str] = set()
    for name in entry["sources"]:
        shard = weight_map[name]
        payload, info = tensor_bytes(directory, shard, name)
        blocks.append(payload)
        shapes.append(list(info["shape"]))
        dtypes.add(info["dtype"])
    if len(dtypes) != 1:
        raise RepackError(f"{entry['target']}: mixed dtypes {sorted(dtypes)}")
    dtype = dtypes.pop()

    if kind == "gate_up":
        if len(blocks) % 2 != 0:
            raise RepackError(f"{entry['target']}: gate/up pairs are not balanced")
        if (
            len({tuple(shape) for shape in shapes[0::2]}) != 1
            or len({tuple(shape) for shape in shapes[1::2]}) != 1
        ):
            raise RepackError(f"{entry['target']}: inconsistent per-expert shapes")
        if shapes[0] != shapes[1]:
            raise RepackError(
                f"{entry['target']}: gate/up shape mismatch {shapes[0]} != {shapes[1]}"
            )
        fused_experts = [blocks[index] + blocks[index + 1] for index in range(0, len(blocks), 2)]
        per_expert_shape = [shapes[0][0] * 2, *shapes[0][1:]]
    else:
        if len(set(map(tuple, shapes))) != 1:
            raise RepackError(f"{entry['target']}: inconsistent per-expert shapes")
        fused_experts = blocks
        per_expert_shape = list(shapes[0])

    payload = b"".join(fused_experts)
    expected = element_size(dtype)
    for dimension in per_expert_shape:
        expected *= dimension
    if len(payload) != expected * entry["num_experts"]:
        raise RepackError(
            f"{entry['target']}: fused payload {len(payload)} bytes does not match "
            f"{entry['num_experts']} experts of {per_expert_shape} {dtype}"
        )
    return payload, dtype, [entry["num_experts"], *per_expert_shape]


def write_safetensors(path: Path, tensors: list[tuple[str, bytes, str, list[int]]]) -> None:
    """Write a safetensors file from name/bytes/dtype/shape tuples."""
    header: dict[str, Any] = {}
    offset = 0
    blobs: list[bytes] = []
    for name, payload, dtype, shape in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        blobs.append(payload)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - (len(header_bytes) % 8)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for blob in blobs:
            handle.write(blob)


class ShardWriter:
    """Write tensors into size-bounded safetensors shards."""

    def __init__(self, output: Path, max_shard_bytes: int) -> None:
        self.output = output
        self.max_shard_bytes = max_shard_bytes
        self.buffer: list[tuple[str, bytes, str, list[int]]] = []
        self.buffer_bytes = 0
        self.shards: list[str] = []
        self.weight_map: dict[str, str] = {}
        self.total_size = 0

    def add(self, name: str, payload: bytes, dtype: str, shape: list[int]) -> None:
        if self.buffer and self.buffer_bytes + len(payload) > self.max_shard_bytes:
            self.flush()
        self.buffer.append((name, payload, dtype, shape))
        self.buffer_bytes += len(payload)
        self.total_size += len(payload)

    def flush(self) -> None:
        if not self.buffer:
            return
        name = f"model-{len(self.shards) + 1:05d}.safetensors"
        write_safetensors(self.output / name, self.buffer)
        for tensor_name, _, _, _ in self.buffer:
            self.weight_map[tensor_name] = name
        self.shards.append(name)
        self.buffer = []
        self.buffer_bytes = 0


def repack(source: Path, output: Path, *, max_shard_bytes: int) -> dict[str, Any]:
    weight_map = read_safetensors_index(source)
    groups, plain = group_experts(weight_map)
    if not groups:
        raise RepackError(f"{source}: no per-expert compressed tensors found")

    output.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(output, max_shard_bytes)
    fused_rows: list[dict[str, Any]] = []

    plain_set = set(plain)
    emitted_stacks: set[str] = set()
    for name in sorted(weight_map):
        if name in plain_set:
            shard = weight_map[name]
            payload, info = tensor_bytes(source, shard, name)
            writer.add(name, payload, info["dtype"], list(info["shape"]))
            continue
        match = EXPERT_PATTERN.match(name)
        if match is None:
            continue
        prefix = match.group("prefix")
        if prefix in emitted_stacks:
            continue
        emitted_stacks.add(prefix)
        for entry in plan_stack(prefix, groups[prefix]):
            payload, dtype, shape = fuse_stack_tensor(source, weight_map, entry)
            writer.add(entry["target"], payload, dtype, shape)
            fused_rows.append(
                {
                    "target": entry["target"],
                    "dtype": dtype,
                    "shape": shape,
                    "bytes": len(payload),
                    "num_experts": entry["num_experts"],
                    "sources": len(entry["sources"]),
                }
            )
    writer.flush()

    index = {
        "metadata": {"total_size": writer.total_size},
        "weight_map": writer.weight_map,
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for extra in PASSTHROUGH_FILES:
        candidate = source / extra
        if candidate.is_file():
            shutil.copy2(candidate, output / extra)
    return {
        "source": str(source),
        "output": str(output),
        "expert_stacks": len(groups),
        "fused_tensors": len(fused_rows),
        "plain_tensors": len(plain),
        "output_tensors": len(writer.weight_map),
        "shards": writer.shards,
        "total_size": writer.total_size,
        "fused_rows": fused_rows,
        "weight_map": writer.weight_map,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-shard-bytes", type=int, default=5 * 1024**3)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = repack(args.source, args.output, max_shard_bytes=args.max_shard_bytes)
    except (RepackError, OSError, json.JSONDecodeError, KeyError) as error:
        print(
            json.dumps(
                {"format": REPACK_FORMAT, "passed": False, "error": str(error)},
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    summary["format"] = REPACK_FORMAT
    summary["output_files"] = {
        path.name: sha256_file(path) for path in sorted(args.output.iterdir()) if path.is_file()
    }
    summary["non_claims"] = [
        "This tool rearranges compressed tensors; it does not quantize, dequantize, "
        "or evaluate the model.",
        "A byte-exact rearrangement does not prove that the fused artifact loads in "
        "an inference engine.",
        "No model forward, quality measurement, or benchmark ran.",
    ]
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
