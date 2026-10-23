#!/usr/bin/env python3
"""Host-safe tests for the calibration ``.npy`` conversion contract."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("calibration_npy", TOOLS / "calibration_npy.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def raw_stream(count: int, *, start: int = 0) -> bytes:
    import array

    values = array.array("i", range(start, start + count))
    if values.itemsize != 4:
        raise AssertionError("unexpected itemsize")
    return values.tobytes()


class CalibrationNpyTest(unittest.TestCase):
    def test_round_trip_preserves_tokens_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "tokens.i32"
            raw.write_bytes(raw_stream(32))
            output = root / "tokens.npy"
            receipt = MODULE.build_matrix(
                raw,
                output,
                sequences=4,
                sequence_length=8,
                split="calibration",
                dataset="fixture",
            )
            shape, tokens = MODULE.read_npy_matrix(output)
            self.assertEqual(shape, (4, 8))
            self.assertEqual(tokens, list(range(32)))
            self.assertEqual(receipt["output"]["dtype"], "int32")
            self.assertEqual(receipt["output"]["shape"], [4, 8])
            self.assertEqual(receipt["source"]["tokens"], 32)
            self.assertEqual(len(receipt["source"]["sha256"]), 64)
            self.assertTrue(receipt["non_claims"])

    def test_token_count_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "tokens.i32"
            raw.write_bytes(raw_stream(30))
            with self.assertRaisesRegex(ValueError, "expected 32 tokens"):
                MODULE.build_matrix(
                    raw,
                    root / "tokens.npy",
                    sequences=4,
                    sequence_length=8,
                    split="calibration",
                    dataset="fixture",
                )

    def test_source_receipt_hash_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "tokens.i32"
            raw.write_bytes(raw_stream(32))
            receipt_path = root / "tokens.receipt.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "format": MODULE.RAW_FORMAT,
                        "raw_output_sha256": "0" * 64,
                        "sequences": 4,
                        "sequence_length": 8,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                MODULE.build_matrix(
                    raw,
                    root / "tokens.npy",
                    sequences=4,
                    sequence_length=8,
                    split="calibration",
                    dataset="fixture",
                    source_receipt=receipt_path,
                )

    def test_source_receipt_geometry_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "tokens.i32"
            raw.write_bytes(raw_stream(32))
            observed = MODULE.sha256_file(raw)
            receipt_path = root / "tokens.receipt.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "format": MODULE.RAW_FORMAT,
                        "raw_output_sha256": observed,
                        "sequences": 2,
                        "sequence_length": 16,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "sequence geometry mismatch"):
                MODULE.build_matrix(
                    raw,
                    root / "tokens.npy",
                    sequences=4,
                    sequence_length=8,
                    split="calibration",
                    dataset="fixture",
                    source_receipt=receipt_path,
                )

    def test_non_integer_multiple_raw_stream_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "tokens.i32"
            raw.write_bytes(raw_stream(31)[:-1])
            with self.assertRaisesRegex(ValueError, "not a multiple of 4"):
                MODULE.build_matrix(
                    raw,
                    root / "tokens.npy",
                    sequences=1,
                    sequence_length=8,
                    split="calibration",
                    dataset="fixture",
                )


if __name__ == "__main__":
    unittest.main()
