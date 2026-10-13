#!/usr/bin/env python3
"""Generate a complete, anchored llama.cpp exact-COPY tensor recipe.

Every non-target tensor keeps the source GGUF's declared type. Target tensors
use the requested research types. Anchoring and regex-escaping each name is
load-bearing: llama.cpp treats recipe entries as regular expressions, so an
unanchored ``ssm_a`` also matches ``ssm_alpha``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RecipeSummary:
    model: Path
    output: Path
    rows: int
    gate_up_rows: int
    down_rows: int
    preserved_rows: int


def desired_type(name: str, source_type: str, *, gate_up_type: str, down_type: str) -> str:
    if name.endswith("ffn_gate_exps.weight") or name.endswith("ffn_up_exps.weight"):
        return gate_up_type
    if name.endswith("ffn_down_exps.weight"):
        return down_type
    return source_type


def recipe_rows(tensors, *, gate_up_type: str, down_type: str) -> tuple[list[str], RecipeSummary]:
    rows: list[str] = []
    seen: set[str] = set()
    gate_up = 0
    down = 0
    preserved = 0
    for tensor in tensors:
        if tensor.name in seen:
            raise ValueError(f"duplicate tensor name in GGUF: {tensor.name}")
        seen.add(tensor.name)
        target = desired_type(
            tensor.name,
            tensor.type_name,
            gate_up_type=gate_up_type,
            down_type=down_type,
        )
        if tensor.name.endswith("ffn_gate_exps.weight") or tensor.name.endswith(
            "ffn_up_exps.weight"
        ):
            gate_up += 1
        elif tensor.name.endswith("ffn_down_exps.weight"):
            down += 1
        else:
            preserved += 1
        rows.append(f"^{re.escape(tensor.name)}$={target}")
    if not rows:
        raise ValueError("model has no tensors")
    return rows, RecipeSummary(
        model=Path("unused"),
        output=Path("unused"),
        rows=len(rows),
        gate_up_rows=gate_up,
        down_rows=down,
        preserved_rows=preserved,
    )


def write_recipe(
    model: Path,
    output: Path,
    *,
    gate_up_type: str = "IQ2_XS",
    down_type: str = "Q2_0",
) -> RecipeSummary:
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile

    rows, summary = recipe_rows(
        GGUFFile(model).tensors,
        gate_up_type=gate_up_type,
        down_type=down_type,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{time.time_ns()}.tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(output)
    return replace(summary, model=model, output=output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gate-up-type", default="IQ2_XS")
    parser.add_argument("--down-type", default="Q2_0")
    args = parser.parse_args()
    summary = write_recipe(
        args.model,
        args.output,
        gate_up_type=args.gate_up_type,
        down_type=args.down_type,
    )
    payload = summary.__dict__ | {"model": str(summary.model), "output": str(summary.output)}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
