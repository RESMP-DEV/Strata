#!/usr/bin/env python3
"""Verify DeepSeek safetensors shard headers without reading tensor payloads."""

from __future__ import annotations

import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

ELEMENT_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
ROUTED = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.(weight|scale)$")
EXPECTED_ROUTED = {
    ("w1", "weight"): ("I8", (2304, 2560)),
    ("w1", "scale"): ("F8_E8M0", (2304, 160)),
    ("w2", "weight"): ("I8", (5120, 1152)),
    ("w2", "scale"): ("F8_E8M0", (5120, 72)),
    ("w3", "weight"): ("I8", (2304, 2560)),
    ("w3", "scale"): ("F8_E8M0", (2304, 160)),
}


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: verify_deepseek_v41_headers.py CHECKPOINT INDEX OUTPUT")
    checkpoint = Path(sys.argv[1])
    index_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    expected_total = int(index["metadata"]["total_size"])
    by_shard: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, set()).add(name)

    failures: list[str] = []
    shard_rows: list[dict[str, object]] = []
    signature_counts: Counter[tuple[str, ...]] = Counter()
    routed_signature_counts: Counter[tuple[str, ...]] = Counter()
    interval_failures = 0
    dtype_shape_failures = 0
    file_size_failures = 0
    total_header_bytes = 0
    total_data_bytes = 0

    for shard in sorted(by_shard):
        path = checkpoint / shard
        try:
            actual_size = path.stat().st_size
        except OSError as error:
            failures.append(f"{shard}: cannot stat shard: {error}")
            file_size_failures += 1
            continue
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                failures.append(f"{shard}: truncated u64 header length")
                continue
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length <= 1 or header_length > 128 * 1024 * 1024:
                failures.append(f"{shard}: implausible header length {header_length}")
                continue
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                failures.append(f"{shard}: truncated header JSON")
                continue
        try:
            header = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            failures.append(f"{shard}: invalid header JSON: {error}")
            continue
        header.pop("__metadata__", None)
        expected_names = by_shard[shard]
        if set(header) != expected_names:
            missing = sorted(expected_names - set(header))[:3]
            extra = sorted(set(header) - expected_names)[:3]
            failures.append(f"{shard}: header/index mismatch missing={missing!r} extra={extra!r}")
            continue

        intervals: list[tuple[int, int, str]] = []
        data_start = 8 + header_length
        for name, spec in header.items():
            if not isinstance(spec, dict):
                failures.append(f"{shard}:{name}: entry is not an object")
                continue
            dtype = spec.get("dtype")
            shape = spec.get("shape")
            offsets = spec.get("data_offsets")
            if dtype not in ELEMENT_BYTES:
                failures.append(f"{shard}:{name}: unknown dtype {dtype!r}")
                dtype_shape_failures += 1
                continue
            if (
                not isinstance(shape, list)
                or not all(isinstance(value, int) for value in shape)
                or any(value < 0 for value in shape)
            ):
                failures.append(f"{shard}:{name}: invalid shape {shape!r}")
                dtype_shape_failures += 1
                continue
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) and value >= 0 for value in offsets)
                or offsets[0] > offsets[1]
            ):
                failures.append(f"{shard}:{name}: invalid offsets {offsets!r}")
                interval_failures += 1
                continue
            expected_bytes = product(shape) * ELEMENT_BYTES[dtype]
            if offsets[1] - offsets[0] != expected_bytes:
                failures.append(
                    f"{shard}:{name}: offsets span {offsets[1] - offsets[0]} bytes, "
                    f"shape/dtype implies {expected_bytes}"
                )
                dtype_shape_failures += 1
                continue
            intervals.append((offsets[0], offsets[1], name))

            routed = ROUTED.match(name)
            if routed is not None:
                key = (f"w{routed.group(4)}", routed.group(5))
                expected_dtype, expected_shape = EXPECTED_ROUTED[key]
                if dtype != expected_dtype or tuple(shape) != expected_shape:
                    failures.append(
                        f"{shard}:{name}: got {dtype}/{shape}, "
                        f"expected {expected_dtype}/{list(expected_shape)}"
                    )
                    dtype_shape_failures += 1
                    continue
                routed_signature_counts[(dtype, tuple(shape))] += 1
            else:
                parts = name.split(".")
                signature_counts[(parts[-2] if len(parts) > 1 else name, dtype, tuple(shape))] += 1

        intervals.sort()
        cursor = 0
        for start, end, name in intervals:
            if start != cursor:
                failures.append(
                    f"{shard}:{name}: interval gap/overlap at {start}, expected {cursor}"
                )
                interval_failures += 1
                break
            cursor = end
        expected_file_size = data_start + cursor
        if cursor == 0 or expected_file_size != actual_size:
            failures.append(
                f"{shard}: file size {actual_size}, header/data imply {expected_file_size}"
            )
            file_size_failures += 1
        total_header_bytes += header_length
        total_data_bytes += cursor
        shard_rows.append(
            {
                "shard": shard,
                "tensors": len(header),
                "header_bytes": header_length,
                "data_bytes": cursor,
                "file_bytes": actual_size,
            }
        )

    passed = not failures and len(by_shard) == 48 and total_data_bytes == expected_total
    result = {
        "format": "strata-deepseek-shard-header-verification-v1",
        "passed": passed,
        "checkpoint": checkpoint.name,
        "payload_bytes_read": 0,
        "shards": len(shard_rows),
        "tensors": sum(int(row["tensors"]) for row in shard_rows),
        "total_header_bytes": total_header_bytes,
        "prefix_bytes_per_shard": 8,
        "total_prefix_bytes": 8 * len(shard_rows),
        "file_size_basis": "8-byte little-endian header-length prefix plus header JSON plus data",
        "total_data_bytes": total_data_bytes,
        "index_declared_total_size_bytes": expected_total,
        "routed_signature_counts": {
            f"{dtype}:{'x'.join(str(value) for value in shape)}": count
            for (dtype, shape), count in sorted(routed_signature_counts.items())
        },
        "top_non_routed_signatures": [
            {"suffix": suffix, "dtype": dtype, "shape": list(shape), "count": count}
            for (suffix, dtype, shape), count in signature_counts.most_common(20)
        ],
        "failures": failures[:100],
        "failure_count": len(failures),
        "interval_failures": interval_failures,
        "dtype_shape_failures": dtype_shape_failures,
        "file_size_failures": file_size_failures,
        "shard_rows": shard_rows,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "shard_rows"}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
