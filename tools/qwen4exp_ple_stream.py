#!/usr/bin/env python3
"""Stream Qwen4Exp PLE n-gram rows from disk instead of holding the table.

Strata's engine reads the PLE table with unbuffered SSD reads and a bounded row
cache (``--ple-io direct``), so the table is never resident.  This module gives
the Python measurement path the same property: the 320,001,536 x 160 table is
addressed as row ranges inside the checkpoint's 128 safetensors shards, and a
forward gathers only the rows its tokens actually select.

The table is read as stored.  A checkpoint materialized from the shipped FP8
table holds BF16 rows; an FP8 shard is decoded with the stored scale.  Rows are
cached in a bounded dictionary keyed by global row id, and every gather reports
rows requested, unique rows, cache hits, bytes read, and elapsed time.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import Any

try:  # keep the module importable on the host, which has no torch
    import torch as _torch

    _MODULE_BASE: Any = _torch.nn.Module
except ImportError:  # pragma: no cover - exercised on the compute host
    _MODULE_BASE = object

PLE_FORMAT = "strata-qwen4exp-ple-stream-v1"


class PleStreamError(ValueError):
    """Raised when the PLE shards do not match the expected layout."""


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length).decode("utf-8"))
    return header, 8 + length


def build_shard_index(model_dir: Path) -> list[dict[str, Any]]:
    """Return ordered shard row ranges for the n-gram embedding table."""
    weight_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    shards: dict[int, str] = {}
    for name in weight_map:
        if ".ngram_embedding.shard_" in name and name.endswith(".weight"):
            index = int(name.rsplit(".shard_", 1)[1].split(".")[0])
            shards[index] = name
    if not shards:
        raise PleStreamError(f"{model_dir}: no ngram_embedding shards found")
    if sorted(shards) != list(range(len(shards))):
        raise PleStreamError("n-gram shards are not numbered without gaps")
    index: list[dict[str, Any]] = []
    start_row = 0
    for key in range(len(shards)):
        name = shards[key]
        path = model_dir / weight_map[name]
        header, base = read_header(path)
        entry = header[name]
        rows, dim = entry["shape"]
        dtype = entry["dtype"]
        row_bytes = dim * (1 if dtype == "F8_E4M3" else 2)
        if entry["data_offsets"][1] - entry["data_offsets"][0] != rows * row_bytes:
            raise PleStreamError(f"{name}: row size does not match the tensor shape")
        index.append(
            {
                "shard": key,
                "path": str(path),
                "data_start": base + entry["data_offsets"][0],
                "start_row": start_row,
                "rows": rows,
                "dim": dim,
                "dtype": dtype,
                "row_bytes": row_bytes,
            }
        )
        start_row += rows
    return index


class DiskBackedNGramEmbedding(_MODULE_BASE):
    """Drop-in replacement for the PLE ``nn.Embedding`` that streams rows."""

    def __init__(
        self,
        model_dir: Path,
        *,
        cache_rows: int = 1 << 18,
        device: str = "cpu",
    ) -> None:
        import torch

        super().__init__()
        self.index = build_shard_index(Path(model_dir))
        self.total_rows = sum(shard["rows"] for shard in self.index)
        self.dim = self.index[0]["dim"]
        self.dtype = self.index[0]["dtype"]
        self.cache_rows = cache_rows
        self.cache: dict[int, Any] = {}
        self.out_cache: dict[str, Any] = {}
        self.out_cache_limit = 8
        self.stats = {
            "gathers": 0,
            "rows_requested": 0,
            "unique_rows": 0,
            "cache_hits": 0,
            "output_cache_hits": 0,
            "bytes_read": 0,
            "elapsed_seconds": 0.0,
        }
        self._handles: dict[str, int] = {}
        # The caller reads ``.weight.device`` to pick an execution device; a
        # tiny CPU marker keeps ids on the CPU where the gather happens.
        self.weight = torch.empty(0, device=device)

    def _shard_for(self, row: int) -> dict[str, Any]:
        for shard in self.index:
            if shard["start_row"] <= row < shard["start_row"] + shard["rows"]:
                return shard
        raise PleStreamError(f"row {row} is outside the table")

    def _read_row(self, row: int) -> Any:
        import torch

        shard = self._shard_for(row)
        handle = self._handles.get(shard["path"])
        if handle is None:
            handle = os.open(shard["path"], os.O_RDONLY)
            self._handles[shard["path"]] = handle
        offset = shard["data_start"] + (row - shard["start_row"]) * shard["row_bytes"]
        raw = os.pread(handle, shard["row_bytes"], offset)
        if len(raw) != shard["row_bytes"]:
            raise PleStreamError(f"short read for row {row}")
        if shard["dtype"] == "F8_E4M3":
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.float8_e4m3fn).to(torch.bfloat16)
        else:
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        self.stats["bytes_read"] += len(raw)
        return tensor

    def __call__(self, ids: Any) -> Any:
        return self.forward(ids)

    def forward(self, ids: Any) -> Any:
        import torch

        started = time.perf_counter()
        cpu_ids = ids.detach().to("cpu", dtype=torch.long)
        digest = hashlib.blake2b(cpu_ids.numpy().tobytes(), digest_size=16).hexdigest()
        key = f"{tuple(ids.shape)}:{digest}"
        cached_output = self.out_cache.get(key)
        if cached_output is not None:
            self.stats["gathers"] += 1
            self.stats["output_cache_hits"] += 1
            self.stats["rows_requested"] += int(cpu_ids.numel())
            self.stats["elapsed_seconds"] += time.perf_counter() - started
            return cached_output.to(ids.device)
        flat = cpu_ids.reshape(-1)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        missing = [int(row) for row in unique if int(row) not in self.cache]
        for row in missing:
            self.cache[row] = self._read_row(row)
        if self.cache_rows and len(self.cache) > self.cache_rows:
            for row in list(self.cache)[: len(self.cache) - self.cache_rows]:
                self.cache.pop(row, None)
        lookup = torch.stack([self.cache[int(row)] for row in unique])
        gathered = lookup.index_select(0, inverse)
        self.stats["gathers"] += 1
        self.stats["rows_requested"] += int(flat.numel())
        self.stats["unique_rows"] += int(unique.numel())
        self.stats["cache_hits"] += int(unique.numel()) - len(missing)
        output = gathered.reshape(*ids.shape, self.dim)
        if len(self.out_cache) >= self.out_cache_limit:
            self.out_cache.pop(next(iter(self.out_cache)))
        self.out_cache[key] = output
        self.stats["elapsed_seconds"] += time.perf_counter() - started
        return output.to(ids.device)


def _materialize_config_buffers(parent: Any) -> None:
    """Recompute the PLE module's config-derived buffers off the meta device."""
    import torch
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        _build_layer_multipliers,
    )

    if not any(buffer.device.type == "meta" for buffer in parent.buffers(recurse=False)):
        return
    parent.layer_multipliers = _build_layer_multipliers(
        parent.unigram_vocab_size,
        parent.ngram_size,
        parent.ple_layer_index,
        parent.seed,
    )
    parent.ngram_heads_vocab_sizes = torch.tensor(parent.head_vocab_sizes, dtype=torch.long)
    parent.ngram_heads_offsets = torch.tensor(parent.head_offsets, dtype=torch.long)


def replace_ple_modules(model: Any, model_dir: Path, *, cache_rows: int) -> list[dict[str, Any]]:
    """Replace every PLE n-gram embedding with the streaming implementation."""
    replaced: list[dict[str, Any]] = []
    for name, _module in list(model.named_modules()):
        if not name.endswith(".ple_embedding.ngram_embedding"):
            continue
        streamed = DiskBackedNGramEmbedding(model_dir, cache_rows=cache_rows)
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, attribute, streamed)
        _materialize_config_buffers(parent)
        replaced.append(
            {
                "module": name,
                "rows": streamed.total_rows,
                "dim": streamed.dim,
                "dtype": streamed.dtype,
                "shards": len(streamed.index),
            }
        )
    return replaced
