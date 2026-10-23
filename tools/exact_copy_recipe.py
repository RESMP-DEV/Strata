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
    ple_rows: int
    preserved_rows: int


def desired_type(
    name: str,
    source_type: str,
    *,
    gate_up_type: str,
    down_type: str,
    ple_type: str | None,
) -> str:
    if name.endswith("ffn_gate_exps.weight") or name.endswith("ffn_up_exps.weight"):
        return gate_up_type
    if name.endswith("ffn_down_exps.weight"):
        return down_type
    if ple_type is not None and name == "per_layer_token_embd.weight":
        return ple_type
    return source_type


def recipe_rows(
    tensors,
    *,
    gate_up_type: str,
    down_type: str,
    ple_type: str | None,
) -> tuple[list[str], RecipeSummary]:
    rows: list[str] = []
    seen: set[str] = set()
    gate_up = 0
    down = 0
    ple = 0
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
            ple_type=ple_type,
        )
        if tensor.name.endswith("ffn_gate_exps.weight") or tensor.name.endswith(
            "ffn_up_exps.weight"
        ):
            gate_up += 1
        elif tensor.name.endswith("ffn_down_exps.weight"):
            down += 1
        elif ple_type is not None and tensor.name == "per_layer_token_embd.weight":
            ple += 1
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
        ple_rows=ple,
        preserved_rows=preserved,
    )


def read_split_tensors(model: Path) -> list:
    """Read tensor metadata from every standard llama.cpp split sibling."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile

    first = GGUFFile(model)
    split_count = first.metadata.get("split.count", 1)
    if not isinstance(split_count, int) or split_count == 1:
        return first.tensors
    if split_count < 1:
        raise ValueError(f"{model.name}: invalid split.count {split_count}")

    marker = re.search(r"-\d{5}-of-\d{5}(?=\.gguf$)", model.name)
    if marker is None or int(marker.group(0)[1:6]) != 1:
        raise ValueError(
            f"{model.name}: multi-split recipes require a standard -00001-of-N input name"
        )

    tensors = []
    for index in range(1, split_count + 1):
        sibling = model.with_name(
            model.name.replace(marker.group(0), f"-{index:05d}-of-{split_count:05d}", 1)
        )
        if not sibling.is_file():
            raise ValueError(f"{model.name}: missing split sibling {sibling.name}")
        tensors.extend(GGUFFile(sibling).tensors)
    return tensors


def write_recipe(
    model: Path,
    output: Path,
    *,
    gate_up_type: str = "IQ2_XS",
    down_type: str = "Q2_0",
    ple_type: str | None = None,
) -> RecipeSummary:
    rows, summary = recipe_rows(
        read_split_tensors(model),
        gate_up_type=gate_up_type,
        down_type=down_type,
        ple_type=ple_type,
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
    parser.add_argument("--ple-type")
    args = parser.parse_args()
    summary = write_recipe(
        args.model,
        args.output,
        gate_up_type=args.gate_up_type,
        down_type=args.down_type,
        ple_type=args.ple_type,
    )
    payload = summary.__dict__ | {"model": str(summary.model), "output": str(summary.output)}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
