#!/usr/bin/env python3
"""Drive the pinned ggml IQ3_S / IQ3_XXS quantize-dequantize bridge."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

IQ3_S = 0
IQ3_XXS = 1
IQ4_NL = 2
IQ4_XS = 3
QK_K = 256
QK4_NL = 32


class Iq3Error(RuntimeError):
    """Raised when the IQ3 bridge is unavailable or rejects a tensor."""


def load_library(path: Path | None = None) -> Any:
    library = path or Path(os.environ.get("STRATA_IQ_QDQ_LIB", "tools/ggml/libiq_qdq.so"))
    if not library.is_file():
        raise Iq3Error(f"IQ3 bridge library not found: {library}")
    # Lazy binding: ggml.c references backend symbols the quantizer never
    # calls, and resolving them eagerly would fail at load time.
    lib = ctypes.CDLL(str(library), mode=os.RTLD_LAZY | os.RTLD_LOCAL)
    lib.iq_qdq.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.iq_qdq.restype = ctypes.c_int
    lib.iq_block_bytes.argtypes = [ctypes.c_int]
    lib.iq_block_bytes.restype = ctypes.c_int
    return lib


def qdq(weights: Any, *, kind: int = IQ3_S, library: Any | None = None) -> tuple[Any, int]:
    """Quantize and dequantize a 2D float32 tensor; return values and bytes."""
    import numpy as np

    if weights.ndim != 2:
        raise Iq3Error(f"expected a 2D tensor, got {weights.shape}")
    rows, cols = weights.shape
    block = QK4_NL if kind == IQ4_NL else QK_K
    if cols % block != 0:
        raise Iq3Error(f"in_features {cols} is not a multiple of {block}")
    source = np.ascontiguousarray(weights, dtype=np.float32)
    output = np.empty_like(source)
    lib = library or load_library()
    out_bytes = ctypes.c_int64(0)
    status = lib.iq_qdq(
        source.ctypes.data_as(ctypes.c_void_p),
        output.ctypes.data_as(ctypes.c_void_p),
        rows,
        cols,
        kind,
        ctypes.byref(out_bytes),
    )
    if status != 0:
        raise Iq3Error(f"iq_qdq failed with status {status}")
    return output, int(out_bytes.value)


def block_bytes(kind: int, *, library: Any | None = None) -> int:
    lib = library or load_library()
    value = lib.iq_block_bytes(kind)
    if value < 0:
        raise Iq3Error(f"unknown IQ3 kind {kind}")
    return int(value)
