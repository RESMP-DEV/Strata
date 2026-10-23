#!/usr/bin/env python3
"""Host-safe tests for the disk-backed PLE shard index."""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "qwen4exp_ple_stream", TOOLS / "qwen4exp_ple_stream.py"
)
assert SPEC is not None and SPEC.loader is not None
PLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLE)


def write_safetensors(path: Path, tensors: dict[str, tuple[bytes, str, list[int]]]) -> None:
    header = {}
    offset = 0
    for name, (payload, dtype, shape) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - (len(raw) % 8)) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for payload, _, _ in tensors.values():
            handle.write(payload)


PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


class PleStreamIndexTest(unittest.TestCase):
    def test_shard_index_orders_rows_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_safetensors(
                root / "a.safetensors",
                {f"{PREFIX}.shard_0.weight": (b"\x00\x00" * 8, "BF16", [4, 2])},
            )
            write_safetensors(
                root / "b.safetensors",
                {f"{PREFIX}.shard_1.weight": (b"\x00\x00" * 12, "BF16", [6, 2])},
            )
            index = {
                "weight_map": {
                    f"{PREFIX}.shard_0.weight": "a.safetensors",
                    f"{PREFIX}.shard_1.weight": "b.safetensors",
                }
            }
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            shards = PLE.build_shard_index(root)
            self.assertEqual([shard["start_row"] for shard in shards], [0, 4])
            self.assertEqual([shard["rows"] for shard in shards], [4, 6])
            self.assertEqual({shard["dim"] for shard in shards}, {2})
            self.assertEqual({shard["row_bytes"] for shard in shards}, {4})

    def test_missing_shards_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
            with self.assertRaisesRegex(ValueError, "no ngram_embedding shards"):
                PLE.build_shard_index(root)

    def test_gapped_shard_numbers_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_safetensors(
                root / "a.safetensors",
                {
                    "x.ple_embedding.ngram_embedding.shard_0.weight": (b"\x00" * 4, "BF16", [2, 1]),
                    "x.ple_embedding.ngram_embedding.shard_2.weight": (b"\x00" * 4, "BF16", [2, 1]),
                },
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "x.ple_embedding.ngram_embedding.shard_0.weight": "a.safetensors",
                            "x.ple_embedding.ngram_embedding.shard_2.weight": "a.safetensors",
                        }
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "not numbered without gaps"):
                PLE.build_shard_index(root)


if __name__ == "__main__":
    unittest.main()
