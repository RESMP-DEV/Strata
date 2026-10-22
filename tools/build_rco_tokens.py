#!/usr/bin/env python3
"""Build deterministic RCO calibration tensors from local code-corpus JSONL.

The corpus is private by default.  The output receipt records paths, hashes,
counts, and token statistics, never source text.  The tensor file uses RCO's
plain-file contract: a list of `(1, sequence_length)` long tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Source:
    path: Path
    sha256: str
    rows: int
    used_rows: int


@dataclass(frozen=True)
class TokenBuild:
    sequences: list[list[int]]
    sources: list[Source]
    records_read: int
    records_used: int
    full_sequences: int
    padded_sequences: int
    non_padding_tokens: int
    padding_tokens: int
    max_token_id: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(paths: Sequence[Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield path, value


def choose_text(record: dict[str, Any], field: str) -> str:
    if field == "text":
        value = record.get("text")
    elif field == "passage":
        value = record.get("passage")
    elif field == "query":
        value = record.get("query")
    elif field == "auto":
        value = record.get("text") if record.get("text") is not None else record.get("passage")
    else:
        raise ValueError(f"unknown field {field!r}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("selected record field is missing or empty")
    return value


def token_ids(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if hasattr(value, "tolist"):
        return [int(item) for item in value.tolist()]
    raise ValueError(f"tokenizer did not return token IDs; got {type(value).__name__}: {value!r}")


def optional_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def build_sequences(
    paths: Sequence[Path],
    tokenizer: Any,
    *,
    sequence_length: int,
    max_records: int | None,
    max_sequences: int | None,
    field: str,
) -> TokenBuild:
    if sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    if max_records is not None and max_records <= 0:
        raise ValueError("--max-records must be positive")
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("--max-sequences must be positive")

    base_vocabulary = int(getattr(tokenizer, "vocab_size", 0) or 0)
    vocabulary = int(getattr(tokenizer, "__len__", lambda: base_vocabulary)())
    eos = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos, int):
        raise ValueError("tokenizer must provide an integer eos_token_id")
    if vocabulary <= eos:
        raise ValueError(f"tokenizer vocabulary {vocabulary} does not contain EOS {eos}")

    row_counts = {path: 0 for path in paths}
    used_counts = {path: 0 for path in paths}
    records_read = 0
    records_used = 0
    stream: list[int] = []
    sequences: list[list[int]] = []
    full_sequences = 0
    padded_sequence = False

    def take_sequence() -> None:
        nonlocal stream
        nonlocal full_sequences
        if len(stream) >= sequence_length:
            sequences.append(stream[:sequence_length])
            stream = stream[sequence_length:]
            full_sequences += 1

    for path, record in iter_jsonl(paths):
        row_counts[path] += 1
        records_read += 1
        if max_records is not None and records_read > max_records:
            records_read -= 1
            row_counts[path] -= 1
            break
        values = token_ids(
            tokenizer(choose_text(record, field), add_special_tokens=False)["input_ids"]
        )
        if any(value < 0 or value >= vocabulary for value in values):
            observed = max(values, default=0)
            raise ValueError(f"token id outside [0, {vocabulary}): {observed}")
        used_counts[path] += 1
        records_used += 1
        stream.extend(values)
        stream.append(eos)
        while len(stream) >= sequence_length:
            take_sequence()
            if max_sequences is not None and len(sequences) == max_sequences:
                break
        if max_sequences is not None and len(sequences) == max_sequences:
            break

    if stream and (max_sequences is None or len(sequences) < max_sequences):
        padded = stream[:sequence_length]
        sequences.append(padded + [eos] * (sequence_length - len(padded)))
        padded_sequence = True

    # EOS separators are real tokens; only the deliberately padded tail is excluded.
    padding = sequence_length - len(stream) if padded_sequence else 0
    non_padding = sum(len(sequence) for sequence in sequences) - padding
    max_token_id = max((max(sequence, default=0) for sequence in sequences), default=0)
    sources = [
        Source(
            path=path,
            sha256=sha256(path),
            rows=row_counts[path],
            used_rows=used_counts[path],
        )
        for path in paths
    ]
    return TokenBuild(
        sequences=sequences,
        sources=sources,
        records_read=records_read,
        records_used=records_used,
        full_sequences=full_sequences,
        padded_sequences=1 if padded_sequence else 0,
        non_padding_tokens=non_padding,
        padding_tokens=padding,
        max_token_id=max_token_id,
    )


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_rco_tokens(path: Path, sequences: Iterable[Sequence[int]]) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    values = [torch.tensor([sequence], dtype=torch.long) for sequence in sequences]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(values, path)


def save_raw_tokens(path: Path, sequences: Iterable[Sequence[int]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for sequence in sequences:
            if any(value < 0 or value > 0xFFFFFFFF for value in sequence):
                raise ValueError("raw token file requires ids in unsigned int32 range")
            handle.write(struct.pack(f"<{len(sequence)}I", *sequence))
            count += len(sequence)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="RCO-compatible .pt output")
    parser.add_argument(
        "--raw-out",
        type=Path,
        help="unsigned little-endian int32 token stream for Strata's patched llama-imatrix",
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-records", type=optional_positive_int)
    parser.add_argument("--max-sequences", type=optional_positive_int)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="opt in to tokenizer code supplied by the local checkpoint",
    )
    parser.add_argument(
        "--field",
        choices=("auto", "text", "passage", "query"),
        default="auto",
        help="auto uses private `text` when present, otherwise public `passage`",
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--visibility", choices=("private", "public"), default="private")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    build = build_sequences(
        args.jsonl,
        tokenizer,
        sequence_length=args.sequence_length,
        max_records=args.max_records,
        max_sequences=args.max_sequences,
        field=args.field,
    )
    save_rco_tokens(args.out, build.sequences)
    tensor_sha256 = sha256(args.out)
    raw_out = args.raw_out or args.out.with_suffix(".i32")
    raw_count = save_raw_tokens(raw_out, build.sequences)
    receipt = {
        "format": "strata-rco-tokens-v1",
        "dataset_name": args.dataset_name,
        "visibility": args.visibility,
        "field": args.field,
        "sequence_length": args.sequence_length,
        "sources": [source.__dict__ | {"path": str(source.path)} for source in build.sources],
        "records_read": build.records_read,
        "records_used": build.records_used,
        "sequences": len(build.sequences),
        "full_sequences": build.full_sequences,
        "padded_sequences": build.padded_sequences,
        "non_padding_tokens": build.non_padding_tokens,
        "padding_tokens": build.padding_tokens,
        "max_token_id": build.max_token_id,
        "tokenizer": str(args.tokenizer),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_base_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_extended_vocab_size": int(len(tokenizer)),
        "eos_token_id": tokenizer.eos_token_id,
        "output": str(args.out),
        "output_sha256": tensor_sha256,
        "raw_output": str(raw_out),
        "raw_output_sha256": sha256(raw_out),
        "raw_output_tokens": raw_count,
        "non_claims": [
            "No model forward, RCO search, healing, quantization, packing, or evaluation ran.",
            "The receipt intentionally omits source text and private corpus content.",
        ],
    }
    receipt_path = args.out.with_suffix(".receipt.json")
    write_atomic(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.out}: {len(build.sequences)}x{args.sequence_length} tokens "
        f"from {build.records_used} records; receipt={receipt_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
