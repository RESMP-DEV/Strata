#!/usr/bin/env python3
"""Convert Strata raw calibration token streams into exact ``.npy`` matrices.

The W4A16 recipe on the compute host loads its calibration matrix with
``numpy.load(path, mmap_mode="r")`` and verifies a declared SHA-256, so this
tool writes the ``.npy`` container by hand: no numpy import is required on the
control host, and the emitted bytes are reproducible from the raw token stream
alone.  The raw stream is never retokenized, resampled, or reordered.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

RAW_FORMAT = "strata-rco-tokens-v1"
NPY_FORMAT = "strata-calibration-npy-v1"
NPY_MAGIC = b"\x93NUMPY"
NPY_VERSION = b"\x01\x00"
DTYPE_DESCR = "<i4"


class CalibrationError(ValueError):
    """Raised when a calibration stream or output matrix violates its contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw_tokens(path: Path) -> list[int]:
    """Read a Strata raw token stream (little-endian unsigned int32)."""
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise CalibrationError(f"{path}: raw stream length {len(data)} is not a multiple of 4")
    values = array.array("i")
    values.frombytes(data)
    if values.itemsize != 4:
        raise CalibrationError(f"{path}: expected 4-byte tokens, got {values.itemsize}")
    if sys.byteorder != "little":
        values.byteswap()
    if any(value < 0 for value in values):
        raise CalibrationError(f"{path}: raw stream contains negative token IDs")
    return list(values)


def npy_header(shape: tuple[int, int]) -> bytes:
    header = (
        "{'descr': '" + DTYPE_DESCR + "', 'fortran_order': False, "
        "'shape': (" + str(shape[0]) + ", " + str(shape[1]) + "), }"
    )
    prefix = len(NPY_MAGIC) + 2 + 2
    padding = 64 - ((prefix + len(header) + 1) % 64)
    header = header + " " * padding + "\n"
    return NPY_MAGIC + NPY_VERSION + len(header).to_bytes(2, "little") + header.encode("latin1")


def write_npy_int32(path: Path, tokens: list[int], shape: tuple[int, int]) -> None:
    """Write a C-order little-endian int32 ``.npy`` matrix."""
    expected = shape[0] * shape[1]
    if len(tokens) != expected:
        raise CalibrationError(
            f"token count {len(tokens)} does not match shape {shape[0]}x{shape[1]}"
        )
    values = array.array("i", tokens)
    if values.itemsize != 4:
        raise CalibrationError(f"expected 4-byte tokens, got {values.itemsize}")
    if sys.byteorder != "little":
        values.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(npy_header(shape))
        handle.write(values.tobytes())


def read_npy_matrix(path: Path) -> tuple[tuple[int, int], list[int]]:
    """Parse a ``.npy`` file written by this tool and return shape and tokens."""
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != NPY_MAGIC:
            raise CalibrationError(f"{path}: not a .npy file")
        version = handle.read(2)
        if version != NPY_VERSION:
            raise CalibrationError(f"{path}: unsupported .npy version {version!r}")
        header_length = int.from_bytes(handle.read(2), "little")
        header = handle.read(header_length).decode("latin1")
        if f"'{DTYPE_DESCR}'" not in header:
            raise CalibrationError(f"{path}: expected dtype {DTYPE_DESCR}, got {header!r}")
        shape_text = header.split("'shape':", 1)[1].split(")", 1)[0]
        dimensions = [int(part) for part in shape_text.replace("(", "").split(",") if part.strip()]
        if len(dimensions) != 2:
            raise CalibrationError(f"{path}: expected a 2-D matrix, got {dimensions}")
        values = array.array("i")
        values.frombytes(handle.read())
        if values.itemsize != 4:
            raise CalibrationError(f"{path}: expected 4-byte tokens, got {values.itemsize}")
        if sys.byteorder != "little":
            values.byteswap()
        return (dimensions[0], dimensions[1]), list(values)


def build_matrix(
    raw_path: Path,
    output_path: Path,
    *,
    sequences: int,
    sequence_length: int,
    split: str,
    dataset: str,
    source_receipt: Path | None = None,
) -> dict[str, Any]:
    if sequences <= 0 or sequence_length <= 0:
        raise CalibrationError("sequences and sequence_length must be positive")
    tokens = read_raw_tokens(raw_path)
    expected = sequences * sequence_length
    if len(tokens) != expected:
        raise CalibrationError(
            f"{raw_path}: expected {expected} tokens for {sequences}x{sequence_length}, "
            f"got {len(tokens)}"
        )
    write_npy_int32(output_path, tokens, (sequences, sequence_length))
    shape, roundtrip = read_npy_matrix(output_path)
    if shape != (sequences, sequence_length):
        raise CalibrationError(f"{output_path}: round-trip shape {shape} mismatch")
    if roundtrip != tokens:
        raise CalibrationError(f"{output_path}: round-trip token mismatch")
    receipt: dict[str, Any] = {
        "format": NPY_FORMAT,
        "split": split,
        "dataset": dataset,
        "source": {
            "path": str(raw_path),
            "sha256": sha256_file(raw_path),
            "bytes": raw_path.stat().st_size,
            "tokens": len(tokens),
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "dtype": "int32",
            "shape": [sequences, sequence_length],
        },
        "token_ids": {
            "min": min(tokens),
            "max": max(tokens),
        },
        "non_claims": [
            "No tokenizer ran; the token stream was copied byte-for-byte from the "
            "recorded raw stream.",
            "No model forward, quantization, calibration, or measurement ran.",
        ],
    }
    if source_receipt is not None:
        source = json.loads(source_receipt.read_text(encoding="utf-8"))
        if source.get("format") != RAW_FORMAT:
            observed_format = source.get("format")
            raise CalibrationError(
                f"{source_receipt}: expected source format {RAW_FORMAT}, got {observed_format!r}"
            )
        declared = source.get("raw_output_sha256")
        observed = receipt["source"]["sha256"]
        if declared != observed:
            raise CalibrationError(
                f"{source_receipt}: raw stream SHA-256 {declared} != observed {observed}"
            )
        if (
            int(source.get("sequences", -1)) != sequences
            or int(source.get("sequence_length", -1)) != sequence_length
        ):
            raise CalibrationError(f"{source_receipt}: sequence geometry mismatch")
        receipt["source"]["receipt"] = str(source_receipt)
        receipt["source"]["receipt_sha256"] = sha256_file(source_receipt)
        receipt["source"]["dataset_name"] = source.get("dataset_name")
        receipt["source"]["tokenizer"] = source.get("tokenizer")
        receipt["source"]["tokenizer_revision"] = source.get("tokenizer_revision")
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequences", required=True, type=int)
    parser.add_argument("--sequence-length", required=True, type=int)
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = build_matrix(
            args.raw,
            args.output,
            sequences=args.sequences,
            sequence_length=args.sequence_length,
            split=args.split,
            dataset=args.dataset,
            source_receipt=args.source_receipt,
        )
    except (CalibrationError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"format": NPY_FORMAT, "passed": False, "error": str(error)}))
        return 2
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
