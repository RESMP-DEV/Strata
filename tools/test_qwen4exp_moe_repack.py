#!/usr/bin/env python3
"""Host-safe tests for the fused MoE repack tool."""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "qwen4exp_moe_repack", TOOLS / "qwen4exp_moe_repack.py"
)
assert SPEC is not None and SPEC.loader is not None
REPACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPACK)


def write_safetensors(path: Path, tensors: list[tuple[str, bytes, str, list[int]]]) -> None:
    header = {}
    offset = 0
    for name, payload, dtype, shape in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - (len(header_bytes) % 8)) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for _, payload, _, _ in tensors:
            handle.write(payload)


def packed(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def scales(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}B", *values)


def source_fixture(root: Path) -> dict[str, bytes]:
    """Two layers, two experts, gate/up/down packed plus scale tensors."""
    payloads: dict[str, bytes] = {}
    tensors: list[tuple[str, bytes, str, list[int]]] = []
    for layer in range(2):
        for expert in range(2):
            for projection, rows in (("gate_proj", 2), ("up_proj", 2), ("down_proj", 3)):
                base = layer * 100 + expert * 10 + len(projection)
                name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                tensors.append(
                    (
                        f"{name}.weight_packed",
                        packed(list(range(base, base + rows * 2))),
                        "I32",
                        [rows, 2],
                    )
                )
                tensors.append(
                    (
                        f"{name}.weight_scale",
                        scales([base % 251] * rows),
                        "U8",
                        [rows, 1],
                    )
                )
                payloads[f"{name}.weight_packed"] = tensors[-2][1]
                payloads[f"{name}.weight_scale"] = tensors[-1][1]
    tensors.append(("model.embed_tokens.weight", packed([1, 2, 3, 4]), "I32", [2, 2]))
    write_safetensors(root / "model.safetensors", tensors)
    return payloads


class Qwen4ExpMoeRepackTest(unittest.TestCase):
    def test_repack_fuses_experts_and_preserves_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            payloads = source_fixture(source)

            summary = REPACK.repack(source, output, max_shard_bytes=1024 * 1024)
            self.assertEqual(summary["expert_stacks"], 2)
            self.assertEqual(summary["fused_tensors"], 8)

            fused_path = output / "model-00001.safetensors"
            header, data_start = REPACK.read_safetensors_header(fused_path)
            self.assertIn("model.layers.0.mlp.experts.gate_up_proj.weight_packed", header)
            self.assertIn("model.layers.0.mlp.experts.down_proj.weight_scale", header)
            self.assertIn("model.embed_tokens.weight", header)

            def payload_of(name: str) -> bytes:
                entry = header[name]
                begin, end = entry["data_offsets"]
                with fused_path.open("rb") as handle:
                    handle.seek(data_start + begin)
                    return handle.read(end - begin)

            gate_up = payload_of("model.layers.0.mlp.experts.gate_up_proj.weight_packed")
            expected = b"".join(
                payloads[f"model.layers.0.mlp.experts.{expert}.{projection}.weight_packed"]
                for expert in range(2)
                for projection in ("gate_proj", "up_proj")
            )
            self.assertEqual(gate_up, expected)
            self.assertEqual(
                header["model.layers.0.mlp.experts.gate_up_proj.weight_packed"]["shape"], [2, 4, 2]
            )

            down = payload_of("model.layers.0.mlp.experts.down_proj.weight_packed")
            self.assertEqual(
                down,
                payloads["model.layers.0.mlp.experts.0.down_proj.weight_packed"]
                + payloads["model.layers.0.mlp.experts.1.down_proj.weight_packed"],
            )

    def test_index_and_passthrough_files_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            source_fixture(source)
            (source / "config.json").write_text('{"model_type": "qwen4_exp_text"}')
            REPACK.repack(source, output, max_shard_bytes=1024 * 1024)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertIn("model.layers.1.mlp.experts.down_proj.weight_packed", index["weight_map"])
            self.assertGreater(index["metadata"]["total_size"], 0)
            self.assertTrue((output / "config.json").is_file())

    def test_missing_expert_tensors_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            write_safetensors(
                source / "model.safetensors",
                [("model.embed_tokens.weight", packed([1, 2, 3, 4]), "I32", [2, 2])],
            )
            with self.assertRaisesRegex(ValueError, "no per-expert compressed tensors"):
                REPACK.repack(source, output, max_shard_bytes=1024 * 1024)

    def test_gate_up_shape_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            tensors = [
                (
                    "model.layers.0.mlp.experts.0.gate_proj.weight_packed",
                    packed([1, 2]),
                    "I32",
                    [1, 2],
                ),
                (
                    "model.layers.0.mlp.experts.0.up_proj.weight_packed",
                    packed([1, 2, 3]),
                    "I32",
                    [1, 3],
                ),
                (
                    "model.layers.0.mlp.experts.0.down_proj.weight_packed",
                    packed([1, 2]),
                    "I32",
                    [1, 2],
                ),
            ]
            write_safetensors(source / "model.safetensors", tensors)
            with self.assertRaisesRegex(ValueError, "gate/up shape mismatch"):
                REPACK.repack(source, output, max_shard_bytes=1024 * 1024)


if __name__ == "__main__":
    unittest.main()
