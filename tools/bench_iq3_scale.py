#!/usr/bin/env python3
"""Measure IQ3_S QDQ throughput against worker count on real expert rows."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
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
    flat = np.ascontiguousarray(batch.reshape(-1, batch.shape[-1]))
    rows_per_expert = batch.shape[1]
    runs = []
    for workers in (1, 4, 8, 16, 32):
        subset = flat[: rows_per_expert * 8]
        started = time.perf_counter()
        _, _nbytes = iq3_qdq.qdq_parallel(subset, kind=iq3_qdq.IQ3_S, workers=workers)
        elapsed = time.perf_counter() - started
        runs.append(
            {
                "workers": workers,
                "experts": 8,
                "seconds": round(elapsed, 3),
                "experts_per_second": round(8 / elapsed, 3),
            }
        )
    print(json.dumps({"host_cpus": os.cpu_count(), "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
