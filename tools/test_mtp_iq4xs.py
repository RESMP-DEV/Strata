"""Host tests for tools/mtp_iq4xs.py, using synthetic runtime and safetensors files."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtp_iq4xs as packer  # noqa: E402


def write_safetensors(
    path: Path, tensors: dict[str, tuple[list[int], bytes]], formats: dict[str, str]
) -> None:
    payload = bytearray()
    header: dict[str, object] = {
        "__metadata__": {
            "requested_format": "iq4_xs",
            "module_formats": json.dumps(formats, sort_keys=True),
        }
    }
    for name, (shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": "U8", "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)


def make_base(root: Path) -> Path:
    rt = root / "rt"
    rt.mkdir()
    lines: list[str] = []
    blob = bytearray()
    entries: list[tuple[str, str, int, int, bytes]] = []
    for module in sorted(packer.PROJECTIONS):
        entries.append(
            (f"{module}.weight", "q8_0", 1, 256, bytes((i * 7 + 3) % 256 for i in range(272)))
        )
    gr_name = "attn_hyper_connection.block_inject_weight.weight"
    gr_raw = np.arange(256, dtype=np.float32)
    gr_bf16 = packer.float32_to_bf16(gr_raw)
    entries.append((gr_name, "bf16", 1, 256, gr_bf16))
    entries.append(("keep.weight", "bf16", 2, 3, b"preserve-me!"))
    for name, kind, rows, cols, raw in entries:
        pad = (-len(blob)) % 256
        blob.extend(b"\0" * pad)
        lines.append(f"{name} {kind} {rows} {cols} {len(blob)} {len(raw)}")
        blob.extend(raw)
    (rt / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (rt / "dense.bin").write_bytes(blob)
    (rt / "experts.bin").write_bytes(b"routed-experts")
    return rt


def make_candidate(path: Path, columns: int = 256, include_gr: bool = True) -> dict[str, str]:
    tensors: dict[str, tuple[list[int], bytes]] = {}
    formats: dict[str, str] = {}
    projection = bytearray(136)
    projection[:2] = bytes(np.array([1.0], dtype=np.float16).view(np.uint8))
    projection[8:] = bytes([0x21]) * 128
    for module in sorted(packer.PROJECTIONS):
        tensors[f"{module}.blocks"] = ([1, columns // 256, 136], bytes(projection))
        formats[module] = "iq4_xs"
    if include_gr:
        for module in sorted(packer.GR_MODULES):
            tensors[f"{module}.blocks"] = ([1, 1, 136], bytes(projection))
            formats[module] = "iq4_xs"
    write_safetensors(path, tensors, formats)
    return formats


class MtpIQ4XsTest(unittest.TestCase):
    def test_repack_preserves_and_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = make_base(root)
            candidate = root / "candidate.safetensors"
            make_candidate(candidate)
            out = root / "out"
            receipt = packer.write_runtime(base, candidate, out, "dequantize-bf16")
            self.assertEqual((out / "experts.bin").read_bytes(), b"routed-experts")
            lines = (out / "dense.txt").read_text().splitlines()
            self.assertEqual(len(lines), 12)
            by_name = {line.split()[0]: line.split() for line in lines}
            self.assertEqual(by_name["fc_embedding.weight"][1], "iq4_xs")
            self.assertEqual(by_name["keep.weight"][1], "bf16")
            blob = (out / "dense.bin").read_bytes()
            keep = by_name["keep.weight"]
            self.assertEqual(blob[int(keep[4]) : int(keep[4]) + int(keep[5])], b"preserve-me!")
            gr = by_name["attn_hyper_connection.block_inject_weight.weight"]
            gr_raw = blob[int(gr[4]) : int(gr[4]) + int(gr[5])]
            decoded = (np.frombuffer(gr_raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            expected = np.array(
                [3328.0] * 16 + [2656.0] * 16
            )  # IQ4NL[1] and IQ4NL[2] with byte 0x21
            np.testing.assert_array_equal(decoded[:32], expected)
            self.assertEqual(len(receipt["replacements"]), 11)
            self.assertEqual(receipt["gr_effective"], "dequantize-bf16")
            self.assertTrue((out / "mtp-iq4xs-receipt.json").exists())

    def test_preserve_base_leaves_gr_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = make_base(root)
            candidate = root / "candidate.safetensors"
            make_candidate(candidate)
            out = root / "out"
            packer.write_runtime(base, candidate, out, "preserve-base")
            original = (base / "dense.bin").read_bytes()
            lines = (out / "dense.txt").read_text().splitlines()
            gr = next(
                line.split()
                for line in lines
                if line.startswith("attn_hyper_connection.block_inject_weight.weight ")
            )
            replaced = (out / "dense.bin").read_bytes()[int(gr[4]) : int(gr[4]) + int(gr[5])]
            base_line = next(
                line.split()
                for line in (base / "dense.txt").read_text().splitlines()
                if line.startswith("attn_hyper_connection.block_inject_weight.weight ")
            )
            old = original[int(base_line[4]) : int(base_line[4]) + int(base_line[5])]
            self.assertEqual(replaced, old)

    def test_shape_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = make_base(root)
            candidate = root / "candidate.safetensors"
            make_candidate(candidate, columns=256, include_gr=False)
            # A one-block IQ4_XS candidate has 256 columns; force a mismatch through the base shape.
            text = (base / "dense.txt").read_text().replace("q8_0 1 256 ", "q8_0 1 512 ", 1)
            (base / "dense.txt").write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shape"):
                packer.write_runtime(base, candidate, root / "out", "dequantize-bf16")


if __name__ == "__main__":
    unittest.main()
