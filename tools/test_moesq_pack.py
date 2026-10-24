"""Focused MoESQ-to-Strata NVFP4 conversion tests.

Run with an environment that provides NumPy, for example:
  uv run --python 3.12 --with numpy python -m unittest tools.test_moesq_pack
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moesq_pack


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    data = bytearray()
    header: dict[str, dict[str, object]] = {}
    for name, (dtype, shape, raw) in tensors.items():
        begin = len(data)
        data.extend(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(data)]}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def sparse_from_dense(dense_packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent compression with mask bits 0 and 1 selected."""
    rows, packed_k = dense_packed.shape
    chunks = dense_packed.reshape(rows, packed_k // 4, 4)
    values = chunks[:, :, :2].reshape(rows, packed_k // 2).copy()
    mask = np.full((rows, packed_k // 8), 0x33, np.uint8)
    return values, mask


def expected_ggml(dense_packed: np.ndarray, scale32: np.ndarray) -> bytes:
    """Independent block_nvfp4 construction, without calling the tool's converter."""
    rows, packed_k = dense_packed.shape
    values = packed_k * 2
    nib = np.empty((rows, values), np.uint8)
    nib[:, 0::2] = dense_packed & 0xF
    nib[:, 1::2] = dense_packed >> 4
    sub = nib.reshape(rows, values // 16, 16)
    qs = (sub[:, :, :8] | (sub[:, :, 8:] << 4)).astype(np.uint8)
    scales = np.repeat(scale32, 2, axis=1).reshape(rows, values // 64, 4)
    blocks = np.concatenate((scales, qs.reshape(rows, values // 64, 32)), axis=2)
    return blocks.reshape(rows, (values // 64) * 36).tobytes()


def write_checkpoint(
    root: Path, layers: int = 1, experts: int = 1, bad_shape: bool = False
) -> dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]]:
    index: dict[str, str] = {}
    shard_tensors: dict[Path, dict[str, tuple[str, tuple[int, ...], bytes]]] = {}
    expected: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(1234)
    for layer in range(layers):
        for expert in range(experts):
            for role, (rows, values) in moesq_pack.ROW_SHAPES.items():
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}"
                shape_rows = rows + (1 if bad_shape and role == "gate_proj" else 0)
                packed_k = values // 2
                dense = np.zeros((shape_rows, packed_k), np.uint8)
                # Mask bits 0/1 mean these two byte planes survive compression.
                planes = rng.integers(1, 256, (shape_rows, packed_k // 4, 2), dtype=np.uint8)
                dense.reshape(shape_rows, packed_k // 4, 4)[:, :, :2] = planes
                values_sparse, mask = sparse_from_dense(dense)
                scale32 = rng.integers(1, 0x7F, (shape_rows, values // 32), dtype=np.uint8)
                tensors = {
                    f"{prefix}.weight_sparse_packed": (
                        "U8",
                        values_sparse.shape,
                        values_sparse.tobytes(),
                    ),
                    f"{prefix}.weight_sparse_mask": ("U8", mask.shape, mask.tobytes()),
                    f"{prefix}.weight_scale": ("F8_E4M3", scale32.shape, scale32.tobytes()),
                    f"{prefix}.weight_global_scale": (
                        "F32",
                        (1,),
                        np.array([1.5 + 0.25 * len(role)], "<f4").tobytes(),
                    ),
                }
                shard = root / f"model-{layer + 2:05d}-of-00049.safetensors"
                shard_tensors.setdefault(shard, {}).update(tensors)
                expected[(layer, expert, role)] = (dense, scale32)
                for name in tensors:
                    index[name] = shard.name
    for shard, tensors in shard_tensors.items():
        write_safetensors(shard, tensors)
    for number in range(1, 50):
        name = f"model-{number:05d}-of-00049.safetensors"
        path = root / name
        if not path.exists():
            path.write_bytes(b"")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index}), encoding="utf-8"
    )
    hashes = root / "SHARD_HASHES.sha256"
    hashes.write_text(
        "".join(f"{'0' * 64}  model-{i:05d}-of-00049.safetensors\n" for i in range(1, 50)),
        encoding="utf-8",
    )
    return expected


class MoESQPackTests(unittest.TestCase):
    def test_invalid_paired_mask_is_refused(self):
        values = np.zeros((1, 4), np.uint8)
        mask = np.array([[0x01]], np.uint8)  # one bit, not paired
        with self.assertRaisesRegex(ValueError, "invalid paired-sparse mask"):
            moesq_pack.decompress_pair_bitmask(values, mask)

    def test_scale_expansion_duplicates_each_32_group(self):
        scale = np.array([[0x01, 0x7E, 0x30]], np.uint8)
        got = moesq_pack.expand_scale_32_to_16(scale, 96)
        np.testing.assert_array_equal(got, [[0x01, 0x01, 0x7E, 0x7E, 0x30, 0x30]])
        with self.assertRaisesRegex(ValueError, "groups of 32"):
            moesq_pack.expand_scale_32_to_16(scale, 95)

    def test_base_manifest_binds_every_copied_file(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            base = Path(tmp) / "base"
            (base / "tokenizer").mkdir(parents=True)
            for name, data in (
                ("dense.bin", b"dense"),
                ("index.txt", b"index"),
                ("conversions.json", b"conversions"),
                ("compat-bf16.json", b"compat"),
                ("tokenizer/vocab.json", b"vocab"),
                ("tokenizer/chat_template.jinja", b"template"),
            ):
                (base / name).write_bytes(data)
            manifest = moesq_pack.base_manifest(base)
            expected_paths = set(moesq_pack.BASE_FILES) | {
                "tokenizer/chat_template.jinja",
                "tokenizer/vocab.json",
            }
            self.assertEqual(set(manifest["files"]), expected_paths)
            for relative in expected_paths:
                wanted = hashlib.sha256((base / relative).read_bytes()).hexdigest()
                self.assertEqual(manifest["files"][relative]["sha256"], wanted)

    def test_one_expert_prefix_preserves_blocks_and_partial(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "checkpoint"
            source.mkdir()
            expected = write_checkpoint(source, layers=1, experts=1)
            before = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
            out = root / "pack"
            argv = [
                "moesq_pack.py",
                "--checkpoint",
                str(source),
                "--out",
                str(out),
                "--layers",
                "1",
                "--experts",
                "1",
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(moesq_pack.main(), 0)

            blob = (out / "experts.bin").read_bytes()
            self.assertEqual(len(blob), moesq_pack.BLOB_BYTES)
            for offset, role in enumerate(moesq_pack.ROLES):
                dense, scale = expected[(0, 0, role)]
                exact = expected_ggml(dense, scale)
                self.assertEqual(
                    blob[
                        offset * moesq_pack.GGML_ROW_BYTES : (offset + 1)
                        * moesq_pack.GGML_ROW_BYTES
                    ],
                    exact,
                    f"{role} bytes changed during conversion",
                )
            tail = struct.unpack("<4f", blob[-16:])
            expected_tail = struct.unpack(
                "<4f", struct.pack("<4f", 1.0 / 3.75, 1.0 / 3.25, 1.0 / 3.75, 0.0)
            )
            self.assertEqual(tail, expected_tail)
            side = json.loads((out / "experts.bin.src.json").read_text())
            self.assertTrue(side["partial"])
            self.assertTrue(side["weight_contract"]["codes_exact"])
            self.assertTrue(side["weight_contract"]["scales_exact"])
            self.assertFalse(side["weight_contract"]["global_multiplier_exact"])
            self.assertFalse(side["weight_contract"]["structured_sparsity_preserved"])
            self.assertFalse((out / "native_experts.txt").exists())
            self.assertFalse(list(out.glob("*.tmp")))
            after = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
            self.assertEqual(before, after)

    def test_bad_geometry_leaves_the_previous_output_alone(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "checkpoint"
            source.mkdir()
            write_checkpoint(source, layers=1, experts=1, bad_shape=True)
            out = root / "pack"
            out.mkdir()
            (out / "native_experts.txt").write_text("previous pack", encoding="utf-8")
            argv = [
                "moesq_pack.py",
                "--checkpoint",
                str(source),
                "--out",
                str(out),
                "--layers",
                "1",
                "--experts",
                "1",
            ]
            with (
                patch.object(sys, "argv", argv),
                self.assertRaisesRegex(ValueError, "gate_proj.weight_sparse_packed"),
            ):
                moesq_pack.main()
            self.assertEqual(
                (out / "native_experts.txt").read_text(encoding="utf-8"), "previous pack"
            )
            self.assertFalse((out / "experts.bin").exists())

    def test_interrupted_write_publishes_no_pack(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "checkpoint"
            source.mkdir()
            write_checkpoint(source, layers=1, experts=1)
            out = root / "pack"
            out.mkdir()
            (out / "native_experts.txt").write_text("previous pack", encoding="utf-8")
            checkpoint = moesq_pack.Checkpoint(source, layers=1, experts=1)
            with (
                patch.object(moesq_pack, "expert_blob", side_effect=RuntimeError("stopped")),
                self.assertRaisesRegex(RuntimeError, "stopped"),
            ):
                moesq_pack.write_pack(checkpoint, out, None, 1, 1, "test-revision")
            self.assertFalse((out / "experts.bin").exists())
            self.assertFalse((out / "experts.bin.src.json").exists())
            self.assertFalse((out / "native_experts.txt").exists())


if __name__ == "__main__":
    unittest.main()
