#!/usr/bin/env python3
"""Prepare and run the pinned RESMP-DEV Qwen calibration token stream.

This is an explicit research path. It does not replace a setup model choice and
does not consume the published mradermacher importance matrix.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET_REPO = "RESMP-DEV/ptq-calibration-corpus"
DATASET_REVISION = "6ef2bbf949b94192ee8bb78d45a2fc840a10cff5"
NPY_PATH = "data/tokenizers/qwen3.8-27b/calibration_256x2048.npy"
RECEIPT_PATH = "data/tokenizers/qwen3.8-27b/calibration_256x2048.receipt.json"
NPY_SHA256 = "d7dd281b1f1756a6492361e1dfb0c65fed259f8a3d65eabbcb51809b50bd35d7"
RECEIPT_SHA256 = "a4f1404d2bbed3aa1403c444c34db39cb7e99e12d10664119043a7b652377806"
EXPECTED_SHAPE = (256, 2048)
TOKEN_FILE_NAME = "calibration_256x2048.i32"
TOKEN_COUNT = EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1]
TOKEN_FILE_SHA256 = "66c6f7e4a68f3e57772064087a22e79e712e5d7c37c2983ec28e00084cab4745"
TOKENIZER_REPO = "Qwen/Qwen3.8-27B"
TOKENIZER_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOKENIZER_JSON_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
TOKENIZER = f"{TOKENIZER_REPO}@{TOKENIZER_REVISION}"


@dataclass(frozen=True)
class StandardTokens:
    directory: Path
    token_file: Path
    token_sha256: str
    max_token_id: int
    tokenizer_json: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{path} SHA-256 is {observed}, expected {expected}")


def parse_npy_int64(path: Path) -> tuple[tuple[int, int], list[int]]:
    """Read the dataset's simple little-endian int64 NPY without NumPy."""
    with path.open("rb") as source:
        if source.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path} is not a NumPy array")
        major = int.from_bytes(source.read(1), "little")
        source.read(1)
        header_size = int.from_bytes(source.read(2 if major == 1 else 4), "little")
        header = ast.literal_eval(source.read(header_size).decode("utf-8").strip())
    descr = header.get("descr")
    shape = tuple(header.get("shape", ()))
    if descr not in ("<i8", "|i8") or header.get("fortran_order") is not False:
        raise ValueError(f"{path} must be a C-order little-endian int64 array, got {header!r}")
    if shape != EXPECTED_SHAPE:
        raise ValueError(f"{path} shape must be {EXPECTED_SHAPE}, got {shape}")
    raw = path.read_bytes()[-TOKEN_COUNT * 8 :]
    if len(raw) != TOKEN_COUNT * 8:
        raise ValueError(f"{path} is truncated: {len(raw)} data bytes")
    values = [int.from_bytes(raw[offset : offset + 8], "little", signed=True) for offset in range(0, len(raw), 8)]
    return shape, values


def export_tokens(npy: Path, destination: Path) -> str:
    _, values = parse_npy_int64(npy)
    output = bytearray(TOKEN_COUNT * 4)
    for index, token in enumerate(values):
        if not 0 <= token <= 0x7FFFFFFF:
            raise ValueError(f"token id {token} at index {index} is not a nonnegative int32")
        output[4 * index : 4 * index + 4] = token.to_bytes(4, "little", signed=False)
    temporary = destination.with_name(f".{destination.name}.{time.time_ns()}.tmp")
    temporary.write_bytes(output)
    temporary.replace(destination)
    return sha256(destination)


def max_token_id(path: Path) -> int:
    observed_max = 0
    with path.open("rb") as source:
        while block := source.read(4):
            if len(block) != 4:
                raise ValueError(f"{path} has a partial int32 token")
            observed_max = max(observed_max, int.from_bytes(block, "little", signed=False))
    return observed_max


def prepare(directory: Path, hf_command: str = "hf") -> StandardTokens:
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        hf_command,
        "download",
        DATASET_REPO,
        NPY_PATH,
        RECEIPT_PATH,
        "--repo-type",
        "dataset",
        "--revision",
        DATASET_REVISION,
        "--local-dir",
        str(directory),
    ]
    tokenizer_directory = directory / "qwen3.8-27b-tokenizer"
    tokenizer_command = [
        hf_command,
        "download",
        TOKENIZER_REPO,
        "tokenizer.json",
        "--revision",
        TOKENIZER_REVISION,
        "--local-dir",
        str(tokenizer_directory),
    ]
    for download in (command, tokenizer_command):
        completed = subprocess.run(download, cwd=ROOT, check=False, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"hf download failed with exit {completed.returncode}: {' '.join(download)}")
    npy = directory / NPY_PATH
    receipt = directory / RECEIPT_PATH
    tokenizer_json = tokenizer_directory / "tokenizer.json"
    verify(npy, NPY_SHA256)
    verify(receipt, RECEIPT_SHA256)
    verify(tokenizer_json, TOKENIZER_JSON_SHA256)
    destination = directory / TOKEN_FILE_NAME
    if destination.is_file() and destination.stat().st_size == TOKEN_COUNT * 4:
        token_hash = sha256(destination)
    else:
        token_hash = export_tokens(npy, destination)
    verify(destination, TOKEN_FILE_SHA256)
    return StandardTokens(directory, destination, token_hash, max_token_id(destination), tokenizer_json)


def write_preparation_receipt(tokens: StandardTokens) -> Path:
    receipt = tokens.directory / "standard-token-export.receipt.json"
    payload = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "npy": NPY_PATH,
        "npy_sha256": NPY_SHA256,
        "receipt": RECEIPT_PATH,
        "receipt_sha256": RECEIPT_SHA256,
        "tokenizer": TOKENIZER,
        "shape": list(EXPECTED_SHAPE),
        "token_count": TOKEN_COUNT,
        "token_file": TOKEN_FILE_NAME,
        "token_file_sha256": tokens.token_sha256,
        "max_token_id": tokens.max_token_id,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "format": "little-endian int32",
    }
    temporary = receipt.with_name(f".{receipt.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(receipt)
    return receipt


def supports_token_file(imatrix: Path) -> bool:
    if not imatrix.is_file():
        raise RuntimeError(f"llama-imatrix not found: {imatrix}")
    completed = subprocess.run([str(imatrix), "--help"], capture_output=True, text=True, check=False)
    return "--token-file" in completed.stdout + completed.stderr


def validate_target_vocabulary(model: Path, tokens: StandardTokens) -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile

    metadata = GGUFFile(model).metadata
    vocabulary = metadata.get("tokenizer.ggml.tokens")
    pre_tokenizer = metadata.get("tokenizer.ggml.pre")
    eos_token = metadata.get("tokenizer.ggml.eos_token_id")
    if not isinstance(vocabulary, list) or len(vocabulary) <= tokens.max_token_id:
        raise RuntimeError(f"{model} vocabulary does not contain standard token id {tokens.max_token_id}")
    if pre_tokenizer != "qwen35" or eos_token != 248046:
        raise RuntimeError(
            f"{model} tokenizer contract differs from the pinned Qwen3.8 view: pre={pre_tokenizer!r}, eos={eos_token!r}"
        )
    tokenizer_json = json.loads(tokens.tokenizer_json.read_text(encoding="utf-8"))
    model_json = tokenizer_json["model"]
    pinned_count = max(model_json["vocab"].values()) + 1
    pinned_tokens = [None] * pinned_count
    for token, token_id in model_json["vocab"].items():
        pinned_tokens[token_id] = token
    for token in tokenizer_json.get("added_tokens", []):
        token_id = token["id"]
        if token_id >= len(pinned_tokens):
            pinned_tokens.extend([None] * (token_id + 1 - len(pinned_tokens)))
        pinned_tokens[token_id] = token["content"]
    if any(token is None for token in pinned_tokens):
        raise RuntimeError("pinned Qwen3.8 tokenizer JSON has vocabulary gaps")
    if vocabulary[:len(pinned_tokens)] != pinned_tokens:
        raise RuntimeError(f"{model} tokenizer tokens differ from the pinned Qwen3.8 tokenizer")
    if any(not re.fullmatch(r"\[PAD[0-9]+\]", token) for token in vocabulary[len(pinned_tokens) :]):
        raise RuntimeError(f"{model} has non-padding tokenizer entries beyond the pinned Qwen3.8 vocabulary")
    pinned_merges = model_json["merges"]
    if vocabulary and metadata.get("tokenizer.ggml.merges") != pinned_merges:
        raise RuntimeError(f"{model} tokenizer merges differ from the pinned Qwen3.8 tokenizer")


def imatrix_chunk_count(path: Path) -> int:
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile

    metadata = GGUFFile(path).metadata
    value = metadata.get("imatrix.chunk_count")
    if not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{path} has no valid imatrix.chunk_count metadata")
    return value


def command_for(
    imatrix: Path,
    model: Path,
    output: Path,
    token_file: Path,
    *,
    context: int = 2048,
    chunks: int = 256,
    resume_output: Path | None = None,
    extra_arguments: list[str] | None = None,
) -> list[str]:
    if context <= 0 or chunks <= 0:
        raise ValueError("context and chunks must be positive")
    if chunks * context > TOKEN_COUNT:
        raise ValueError(f"{chunks} x {context} tokens exceeds the standard {TOKEN_COUNT}-token view")
    command = [
        str(imatrix),
        "--token-file",
        str(token_file),
        "-m",
        str(model),
        "-o",
        str(output),
        "-c",
        str(context),
        "--chunks",
        str(chunks),
        "--no-ppl",
        "--output-frequency",
        "10",
        "--save-frequency",
        "50",
    ]
    if resume_output is not None:
        command += ["--in-file", str(resume_output), "--chunk", str(imatrix_chunk_count(resume_output))]
    command += [
        *(extra_arguments or []),
    ]
    return command


def build_tools(build_directory: Path) -> Path:
    sys.path.insert(0, str(ROOT))
    import setup

    old = os.environ.get("STRATA_TOKEN_IMATRIX")
    os.environ["STRATA_TOKEN_IMATRIX"] = "1"
    try:
        llama = setup.get_llama_cpp()
    finally:
        if old is None:
            os.environ.pop("STRATA_TOKEN_IMATRIX", None)
        else:
            os.environ["STRATA_TOKEN_IMATRIX"] = old
    build_directory.mkdir(parents=True, exist_ok=True)
    configure = [
        "cmake",
        "-S",
        str(llama),
        "-B",
        str(build_directory),
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=OFF",
    ]
    build = ["cmake", "--build", str(build_directory), "--target", "llama-imatrix"]
    for command in (configure, build):
        completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"llama-imatrix build failed with exit {completed.returncode}: {' '.join(command)}")
    executable = (
        build_directory / "llama-imatrix.exe"
        if os.name == "nt"
        else build_directory / "bin" / "llama-imatrix"
    )
    if not executable.is_file():
        candidates = [path for path in build_directory.rglob("llama-imatrix*") if path.is_file()]
        if not candidates:
            raise RuntimeError(f"build did not produce llama-imatrix under {build_directory}")
        executable = candidates[0]
    return executable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=ROOT / "build" / "standard-calibration")
    parser.add_argument("--prepare-only", action="store_true", help="download, verify, and export token ids only")
    parser.add_argument("--build-tools", action="store_true", help="apply the opt-in patch and build llama-imatrix")
    parser.add_argument("--tools-build-dir", type=Path, default=ROOT / "build" / "llama-token-tools")
    parser.add_argument("--imatrix", type=Path, default=ROOT / "build" / "llama-token-tools" / "bin" / "llama-imatrix")
    parser.add_argument("--model", type=Path, help="target GGUF")
    parser.add_argument("--output", type=Path, help="output iMatrix GGUF")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--chunks", type=int, default=256)
    parser.add_argument("--resume", action="store_true", help="load the output iMatrix and continue at its chunk count")
    parser.add_argument("--imatrix-arg", action="append", default=[], help="additional llama-imatrix argument")
    parser.add_argument("--hf", default="hf")
    parser.add_argument("--print-command", action="store_true")
    arguments = parser.parse_args(argv)

    tokens = prepare(arguments.download_dir, arguments.hf)
    receipt = write_preparation_receipt(tokens)
    print(f"prepared {tokens.token_file} ({TOKEN_COUNT} tokens, SHA-256 {tokens.token_sha256})")
    print(f"receipt: {receipt}")
    if arguments.prepare_only:
        return 0
    if arguments.build_tools:
        arguments.imatrix = build_tools(arguments.tools_build_dir)
    if arguments.model is None or arguments.output is None:
        parser.error("--model and --output are required unless --prepare-only is set")
    validate_target_vocabulary(arguments.model, tokens)
    if not supports_token_file(arguments.imatrix):
        raise RuntimeError(f"{arguments.imatrix} lacks --token-file; build it with --build-tools")
    command = command_for(
        arguments.imatrix,
        arguments.model,
        arguments.output,
        tokens.token_file,
        context=arguments.context,
        chunks=arguments.chunks,
        resume_output=arguments.output if arguments.resume and arguments.output.is_file() else None,
        extra_arguments=arguments.imatrix_arg,
    )
    if arguments.print_command:
        print(" ".join(command))
        return 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
