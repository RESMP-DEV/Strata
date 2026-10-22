"""Repack a Flash-Next MTP IQ4 candidate into a Strata runtime directory.

The MMVQ projections are copied as native IQ4_XS/IQ4_NL blocks.  Candidate fused-GR modules,
which Strata currently consumes as BF16, are dequantized with the GGML IQ4 decode and rounded
to BF16; ``--gr preserve-base`` instead keeps the base runtime's GR weights.  Untouched norms,
routers, routed experts, and the draft vocabulary are copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECTIONS = {
    "fc_embedding",
    "fc_hidden",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.indexer.index_qk_proj",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
}
GR_MODULES = {
    "attn_hyper_connection.block_inject_weight",
    "attn_hyper_connection.input_mix_weight_down",
    "attn_hyper_connection.input_mix_weight_up",
    "mlp_hyper_connection.block_inject_weight",
    "mlp_hyper_connection.input_mix_weight_down",
    "mlp_hyper_connection.input_mix_weight_up",
    "hyper_connection_mixer.input_mix_weight_down",
    "hyper_connection_mixer.input_mix_weight_up",
}
FORMATS = {
    "iq4_nl": (32, 18),
    "iq4_xs": (256, 136),
}
KV_IQ4NL = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)


@dataclass(frozen=True)
class CandidateTensor:
    module: str
    format_name: str
    rows: int
    columns: int
    data: bytes

    @property
    def block_values(self) -> int:
        return FORMATS[self.format_name][0]

    @property
    def block_bytes(self) -> int:
        return FORMATS[self.format_name][1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidate(path: Path) -> tuple[dict[str, str], dict[str, CandidateTensor]]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 1 or header_size > (16 << 20) or 8 + header_size > size:
            raise ValueError(f"{path}: invalid safetensors header length {header_size}")
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"{path}: truncated safetensors header")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: malformed safetensors JSON: {error}") from error
    if not isinstance(header, dict) or "__metadata__" not in header:
        raise ValueError(f"{path}: safetensors metadata is missing")
    metadata = header.pop("__metadata__")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: safetensors metadata is not an object")
    try:
        module_formats = json.loads(metadata.get("module_formats", "{}"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: malformed module_formats metadata: {error}") from error
    if not isinstance(module_formats, dict):
        raise ValueError(f"{path}: module_formats metadata is not an object")

    tensors: dict[str, CandidateTensor] = {}
    with path.open("rb") as handle:
        for key, value in header.items():
            if not isinstance(value, dict):
                raise ValueError(f"{path}: tensor {key} metadata is not an object")
            dtype = value.get("dtype")
            shape = value.get("shape")
            offsets = value.get("data_offsets")
            if dtype != "U8" or not isinstance(shape, list) or len(shape) != 3:
                raise ValueError(f"{path}: {key} is not a packed U8 [rows, blocks, bytes] tensor")
            if not all(isinstance(n, int) and n > 0 for n in shape):
                raise ValueError(f"{path}: {key} has an invalid shape {shape}")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"{path}: {key} has invalid data_offsets")
            start, end = offsets
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not (0 <= start < end <= size)
            ):
                raise ValueError(f"{path}: {key} has invalid data_offsets {offsets}")
            base = 8 + header_size
            if (
                start > size - base
                or end > size - base
                or end - start != shape[0] * shape[1] * shape[2]
            ):
                raise ValueError(f"{path}: {key} byte range does not match its shape")
            if not key.endswith(".blocks"):
                raise ValueError(f"{path}: unexpected tensor name {key}")
            module = key[: -len(".blocks")]
            format_name = module_formats.get(module)
            if format_name not in FORMATS:
                raise ValueError(f"{path}: {module} has unsupported format {format_name!r}")
            block_values, block_bytes = FORMATS[format_name]
            if shape[1] <= 0 or shape[2] != block_bytes:
                raise ValueError(f"{path}: {module} block shape does not match {format_name}")
            handle.seek(base + start)
            data = handle.read(end - start)
            if len(data) != end - start:
                raise ValueError(f"{path}: short read for {key}")
            tensors[module] = CandidateTensor(
                module=module,
                format_name=format_name,
                rows=shape[0],
                columns=shape[1] * block_values,
                data=data,
            )
    if set(tensors) != set(module_formats):
        raise ValueError(f"{path}: module_formats and packed tensors disagree")
    return module_formats, tensors


def dequantize_iq4_nl(raw: bytes) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    scales = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32).reshape(-1)
    codes = np.empty((len(blocks), 32), dtype=np.uint8)
    codes[:, :16] = blocks[:, 2:] & 0x0F
    codes[:, 16:] = blocks[:, 2:] >> 4
    return (KV_IQ4NL[codes] * scales[:, None]).reshape(-1)


def dequantize_iq4_xs(raw: bytes) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 136)
    scales_d = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32).reshape(-1)
    scales_h = np.ascontiguousarray(blocks[:, 2:4]).view(np.uint16).astype(np.int32).reshape(-1)
    scales_l = blocks[:, 4:8].astype(np.int32)
    qs = blocks[:, 8:136]
    scales = np.empty((len(blocks), 8), dtype=np.float32)
    codes = np.empty((len(blocks), 256), dtype=np.uint8)
    for group in range(8):
        low = (scales_l[:, group // 2] >> (4 * (group % 2))) & 0x0F
        high = (scales_h >> (2 * group)) & 0x03
        scales[:, group] = scales_d * ((low | (high << 4)) - 32).astype(np.float32)
        packed = qs[:, group * 16 : (group + 1) * 16]
        codes[:, group * 32 : group * 32 + 16] = packed & 0x0F
        codes[:, group * 32 + 16 : group * 32 + 32] = packed >> 4
    return (KV_IQ4NL[codes].reshape(-1, 8, 32) * scales[:, :, None]).reshape(-1)


def float32_to_bf16(values: np.ndarray) -> bytes:
    """Round float32 to little-endian BF16 with the usual nearest-even boundary bit."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).reshape(-1)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16).tobytes()


def parse_dense_manifest(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    tensors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"{path}: malformed line {line!r}")
        name, kind = fields[0], fields[1]
        if name in seen:
            raise ValueError(f"{path}: duplicate tensor {name}")
        seen.add(name)
        tensors.append(
            {
                "name": name,
                "kind": kind,
                "rows": int(fields[2]),
                "cols": int(fields[3]),
                "off": int(fields[4]),
                "bytes": int(fields[5]),
            }
        )
    blob = (path.parent / "dense.bin").read_bytes()
    for tensor in tensors:
        if tensor["off"] < 0 or tensor["bytes"] < 0 or tensor["off"] + tensor["bytes"] > len(blob):
            raise ValueError(f"{path}: {tensor['name']} exceeds dense.bin")
    return tensors, blob


def candidate_bytes(tensor: CandidateTensor, target_kind: str) -> bytes:
    if target_kind == tensor.format_name:
        return tensor.data
    if target_kind == "bf16":
        values = (
            dequantize_iq4_xs(tensor.data)
            if tensor.format_name == "iq4_xs"
            else dequantize_iq4_nl(tensor.data)
        )
        return float32_to_bf16(values)
    raise ValueError(f"unsupported target kind {target_kind}")


def runtime_bytes(kind: str, rows: int, columns: int) -> int:
    if kind == "bf16":
        return rows * columns * 2
    if kind == "f32":
        return rows * columns * 4
    if kind == "q8_0":
        return rows * (columns // 32) * 34
    if kind == "iq4_nl":
        return rows * (columns // 32) * 18
    if kind == "iq4_xs":
        return rows * (columns // 256) * 136
    raise ValueError(f"unsupported dense kind {kind}")


def write_runtime(base: Path, candidate: Path, output: Path, gr_mode: str) -> dict[str, Any]:
    module_formats, sources = read_candidate(candidate)
    tensors, blob = parse_dense_manifest(base / "dense.txt")
    by_name = {tensor["name"]: tensor for tensor in tensors}
    missing = sorted(
        f"{module}.weight" for module in PROJECTIONS if f"{module}.weight" not in by_name
    )
    if missing:
        raise ValueError(f"base runtime is missing MTP projections: {missing}")
    missing_candidates = sorted(PROJECTIONS - set(sources))
    if missing_candidates:
        raise ValueError(f"candidate is missing MTP projections: {missing_candidates}")
    gr_present = GR_MODULES & set(sources)
    if gr_mode == "dequantize-bf16" and gr_present and gr_present != GR_MODULES:
        missing_gr = sorted(GR_MODULES - gr_present)
        raise ValueError(
            f"candidate has a partial fused-GR set; missing {missing_gr}. "
            "Use a complete candidate or --gr preserve-base."
        )
    effective_gr_mode = (
        "dequantize-bf16"
        if gr_mode == "dequantize-bf16" and gr_present == GR_MODULES
        else "preserve-base"
    )

    output.mkdir(parents=True, exist_ok=False)
    try:
        for entry in base.iterdir():
            if entry.name in {"dense.txt", "dense.bin"}:
                continue
            destination = output / entry.name
            if entry.is_dir():
                shutil.copytree(entry, destination)
            else:
                shutil.copy2(entry, destination)

        replacements: dict[str, dict[str, Any]] = {}
        output_parts: list[bytes] = []
        lines: list[str] = []
        offset = 0
        for tensor in tensors:
            name = tensor["name"]
            module = name[: -len(".weight")] if name.endswith(".weight") else None
            source = sources.get(module) if module else None
            target_kind: str | None = None
            raw: bytes | None = None
            if module in PROJECTIONS and source is not None:
                if tensor["kind"] != "q8_0":
                    raise ValueError(
                        f"{name}: expected a q8_0 base projection, got {tensor['kind']}"
                    )
                target_kind, raw = source.format_name, source.data
            elif gr_mode == "dequantize-bf16" and module in GR_MODULES and source is not None:
                if tensor["kind"] != "bf16":
                    raise ValueError(
                        f"{name}: expected a bf16 base fused-GR tensor, got {tensor['kind']}"
                    )
                target_kind, raw = "bf16", candidate_bytes(source, "bf16")
            else:
                target_kind = tensor["kind"]
            replace_candidate = source is not None and (
                module in PROJECTIONS or (gr_mode == "dequantize-bf16" and module in GR_MODULES)
            )
            if replace_candidate:
                if source.rows != tensor["rows"] or source.columns != tensor["cols"]:
                    raise ValueError(
                        f"{name}: candidate shape {source.rows}x{source.columns}, "
                        f"base shape {tensor['rows']}x{tensor['cols']}"
                    )
                expected = runtime_bytes(target_kind, tensor["rows"], tensor["cols"])
                if raw is None or len(raw) != expected:
                    raise ValueError(
                        f"{name}: replacement has {len(raw or b'')} bytes, expected {expected}"
                    )
            if raw is None:
                target_kind = tensor["kind"]
                raw = blob[tensor["off"] : tensor["off"] + tensor["bytes"]]
            padding = (-offset) % 256
            output_parts.append(b"\0" * padding)
            offset += padding
            output_parts.append(raw)
            lines.append(
                f"{name} {target_kind} {tensor['rows']} {tensor['cols']} {offset} {len(raw)}"
            )
            if module in PROJECTIONS and source is not None:
                replacements[name] = {"format": target_kind, "bytes": len(raw)}
            elif module in GR_MODULES and source is not None and replace_candidate:
                replacements[name] = {
                    "format": f"{source.format_name}->bf16",
                    "bytes": len(raw),
                }
            offset += len(raw)

        dense_path = output / "dense.bin"
        with dense_path.open("wb") as handle:
            for part in output_parts:
                handle.write(part)
            handle.flush()
            os.fsync(handle.fileno())
        (output / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        receipt = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "operator": "Codex",
            "base_rt_dir": str(base),
            "candidate": str(candidate),
            "candidate_sha256": sha256_file(candidate),
            "base_dense_sha256": sha256_file(base / "dense.bin"),
            "output_dense_sha256": sha256_file(dense_path),
            "candidate_module_formats": module_formats,
            "gr_requested": gr_mode,
            "gr_effective": effective_gr_mode,
            "replacements": replacements,
            "non_claims": [
                "This repack does not run the CUDA drafter or measure speculative acceptance.",
                "Fused-GR modules are BF16 dequantizations until direct low-bit GR kernels exist.",
            ],
        }
        (output / "mtp-iq4xs-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return receipt
    except BaseException:
        shutil.rmtree(output)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base", required=True, type=Path, help="existing tools/mtp_rt.py runtime directory"
    )
    parser.add_argument(
        "--candidate", required=True, type=Path, help="Flash-Next MTP GGML safetensors artifact"
    )
    parser.add_argument("--out", required=True, type=Path, help="new runtime directory")
    parser.add_argument(
        "--gr",
        choices=("dequantize-bf16", "preserve-base"),
        default="dequantize-bf16",
        help="candidate fused-GR handling (default: dequantize IQ4 to BF16)",
    )
    arguments = parser.parse_args()
    receipt = write_runtime(arguments.base, arguments.candidate, arguments.out, arguments.gr)
    print(
        f"wrote {arguments.out}: {len(receipt['replacements'])} replaced tensors, "
        f"dense sha256 {receipt['output_dense_sha256']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
