#!/usr/bin/env python3
"""Time serial against parallel IQ3 QDQ on real expert gate/up rows."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import torch
from safetensors import safe_open

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import iq3_qdq  # noqa: E402

ROOT = pathlib.Path("/home/kearm/strata-scratch/own-coder-run/pruned-bf16-288")
LIBRARY = pathlib.Path("/home/kearm/strata-scratch/own-coder-run/w4a16/tools/ggml/libiq_qdq.so")


def main() -> None:
    with (ROOT / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    with safe_open(str(ROOT / weight_map[name]), framework="pt") as handle:
        rows = handle.get_slice(name)
        batch = torch.stack([rows[index] for index in range(32)]).float().numpy()
    os.environ["STRATA_IQ_QDQ_LIB"] = str(LIBRARY)
    library = iq3_qdq.load_library(LIBRARY)

    started = time.perf_counter()
    for index in range(8):
        iq3_qdq.qdq(batch[index], kind=iq3_qdq.IQ3_S, library=library)
    serial = time.perf_counter() - started

    started = time.perf_counter()
    _, nbytes = iq3_qdq.qdq_parallel(
        batch.reshape(-1, batch.shape[-1]), kind=iq3_qdq.IQ3_S, workers=8
    )
    parallel = time.perf_counter() - started
    print(
        json.dumps(
            {
                "experts": 8,
                "serial_seconds": round(serial, 3),
                "parallel_seconds": round(parallel, 3),
                "speedup": round(serial / parallel, 2),
                "parallel_bytes": nbytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
