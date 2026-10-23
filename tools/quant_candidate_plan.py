#!/usr/bin/env python3
"""Plan exact W4A16/W4A4 measurements and BF16 reference-oracle inputs.

This is a provenance planner, not a benchmark.  It refuses a measurement arm
when its weight-byte costs are only nominal bitwidth arithmetic or when the
BF16 reference cache is not bound to both calibration and held-out identities.
The output always starts non-GPU; execution still has to pass
``tools/optimizer_gate.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

INPUT_FORMAT = "strata-quant-candidate-manifest-v1"
OUTPUT_FORMAT = "strata-quant-measurement-plan-v1"
ACTIVATION_PRECISIONS = {"w4a16": 16, "w4a4": 4}
REFERENCE_FORMATS = {"bf16-full-log-probs", "bf16-topk-log-probs"}
HEX_DIGITS = set("0123456789abcdef")


class PlanError(ValueError):
    """Raised when a candidate or reference provenance contract is invalid."""


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX_DIGITS for character in value)
    )


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanError(f"{name} must be an object")
    return value


def require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlanError(f"{name} must be a non-empty string")
    return value


def require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlanError(f"{name} must be a positive integer")
    return value


def require_sha256(value: Any, name: str) -> str:
    if not is_sha256(value):
        raise PlanError(f"{name} must be a 64-character lowercase SHA-256")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split(value: Any, name: str) -> dict[str, Any]:
    split = require_mapping(value, name)
    require_nonempty_string(split.get("dataset"), f"{name}.dataset")
    require_positive_int(split.get("sequences"), f"{name}.sequences")
    require_positive_int(split.get("sequence_length"), f"{name}.sequence_length")
    require_sha256(split.get("tokens_sha256"), f"{name}.tokens_sha256")
    return split


def validate_cost_artifact(
    value: Any,
    name: str,
    *,
    root: Path,
    verify_hash: bool,
) -> dict[str, Any]:
    artifact = require_mapping(value, name)
    relative = require_nonempty_string(artifact.get("path"), f"{name}.path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise PlanError(f"{name}.path must be relative to the input manifest")
    require_sha256(artifact.get("sha256"), f"{name}.sha256")
    declared_bytes = require_positive_int(artifact.get("bytes"), f"{name}.bytes")
    if verify_hash:
        resolved = root / path
        if not resolved.is_file():
            raise PlanError(f"{name}: cost artifact is not a local JSON file: {resolved}")
        if resolved.suffix != ".json":
            raise PlanError(f"{name}: exact byte costs must come from a JSON cost manifest")
        if resolved.stat().st_size != declared_bytes:
            raise PlanError(
                f"{name}: declared bytes {declared_bytes} != file size {resolved.stat().st_size}"
            )
        observed = file_sha256(resolved)
        if observed != artifact["sha256"]:
            raise PlanError(
                f"{name}: SHA-256 mismatch, expected {artifact['sha256']}, got {observed}"
            )
    return artifact


def validate_tensor_costs(value: Any, name: str) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    if not rows:
        raise PlanError(f"{name} must be a non-empty tensor-cost array")
    seen: set[str] = set()
    total = 0
    for index, raw in enumerate(rows):
        row = require_mapping(raw, f"{name}[{index}]")
        tensor = require_nonempty_string(row.get("tensor"), f"{name}[{index}].tensor")
        if tensor in seen:
            raise PlanError(f"{name}: duplicate tensor {tensor}")
        seen.add(tensor)
        cost = require_positive_int(row.get("bytes"), f"{name}[{index}].bytes")
        total += cost
    return rows


def validate_candidate(
    value: Any,
    index: int,
    *,
    root: Path,
    verify_hash: bool,
) -> dict[str, Any]:
    name = f"candidates[{index}]"
    candidate = require_mapping(value, name)
    identifier = require_nonempty_string(candidate.get("id"), f"{name}.id")
    if identifier not in ACTIVATION_PRECISIONS:
        raise PlanError(f"{name}.id must be one of {sorted(ACTIVATION_PRECISIONS)}")
    if candidate.get("weight_bits") != 4:
        raise PlanError(f"{name}: W4 lanes must declare weight_bits=4")
    expected_activation = ACTIVATION_PRECISIONS[identifier]
    if candidate.get("activation_bits") != expected_activation:
        raise PlanError(
            f"{name}: {identifier} requires activation_bits={expected_activation}, "
            f"got {candidate.get('activation_bits')!r}"
        )
    validate_cost_artifact(
        candidate.get("cost_artifact"),
        f"{name}.cost_artifact",
        root=root,
        verify_hash=verify_hash,
    )
    rows = validate_tensor_costs(candidate.get("tensor_costs"), f"{name}.tensor_costs")
    declared = candidate["cost_artifact"]["bytes"]
    # The JSON cost artifact's own file size is not the model payload size.  Its
    # summed tensor rows are the exact deployment bytes being planned.
    if sum(row["bytes"] for row in rows) <= 0:
        raise PlanError(f"{name}: tensor costs sum to zero")
    candidate["tensor_count"] = len(rows)
    candidate["deployment_weight_bytes"] = sum(row["bytes"] for row in rows)
    candidate["cost_artifact_bytes"] = declared
    return candidate


def validate_reference_oracle(value: Any) -> dict[str, Any]:
    oracle = require_mapping(value, "reference_oracle")
    kind = require_nonempty_string(oracle.get("format"), "reference_oracle.format")
    if kind not in REFERENCE_FORMATS:
        raise PlanError(f"reference_oracle.format must be one of {sorted(REFERENCE_FORMATS)}")
    if oracle.get("model_precision") != "BF16":
        raise PlanError("reference_oracle.model_precision must be BF16")
    topk = oracle.get("topk")
    if kind == "bf16-topk-log-probs":
        topk = require_positive_int(topk, "reference_oracle.topk")
    elif topk is not None:
        raise PlanError("reference_oracle.topk is valid only for bf16-topk-log-probs")
    for split_name in ("calibration", "heldout"):
        artifact = require_mapping(oracle.get(split_name), f"reference_oracle.{split_name}")
        require_positive_int(artifact.get("bytes"), f"reference_oracle.{split_name}.bytes")
        require_sha256(artifact.get("sha256"), f"reference_oracle.{split_name}.sha256")
    return oracle


def build_plan(
    document: dict[str, Any],
    *,
    root: Path,
    verify_hash: bool = True,
) -> dict[str, Any]:
    if document.get("format") != INPUT_FORMAT:
        raise PlanError(f"expected input format {INPUT_FORMAT}")
    model = require_mapping(document.get("model"), "model")
    require_nonempty_string(model.get("id"), "model.id")
    require_sha256(model.get("sha256"), "model.sha256")
    splits = require_mapping(document.get("splits"), "splits")
    calibration = validate_split(splits.get("calibration"), "splits.calibration")
    heldout = validate_split(splits.get("heldout"), "splits.heldout")
    if calibration["tokens_sha256"] == heldout["tokens_sha256"]:
        raise PlanError("calibration and held-out token streams must differ")

    raw_candidates = document.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 2:
        raise PlanError("candidates must contain exactly W4A16 and W4A4 arms")
    candidates = [
        validate_candidate(candidate, index, root=root, verify_hash=verify_hash)
        for index, candidate in enumerate(raw_candidates)
    ]
    identifiers = [candidate["id"] for candidate in candidates]
    if sorted(identifiers) != ["w4a16", "w4a4"]:
        raise PlanError("candidate arms must be exactly w4a16 and w4a4")
    oracle = validate_reference_oracle(document.get("reference_oracle"))

    return {
        "format": OUTPUT_FORMAT,
        "model": model,
        "splits": {"calibration": calibration, "heldout": heldout},
        "arms": candidates,
        "reference_oracle": oracle,
        "gpu_allowed": False,
        "next_gate": "tools/optimizer_gate.py check --ready READY_MANIFEST",
        "non_claims": [
            "No W4A16 or W4A4 measurement, model forward, activation capture, or benchmark ran.",
            "Exact tensor bytes come from the supplied cost manifests; this planner "
            "does not derive bytes from nominal bits.",
            "A BF16 reference cache is an optimization oracle only and is not a deployment arm.",
            "Execution remains blocked until the separate optimizer readiness gate passes.",
        ],
    }


def load_input(path: Path) -> tuple[dict[str, Any], Path]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot read {path}: {error}") from error
    return require_mapping(document, "input manifest"), path.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-hash-verify",
        action="store_true",
        help="validate schema only; never use this to unblock a measurement",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        document, root = load_input(args.input)
        plan = build_plan(
            document,
            root=root,
            verify_hash=not args.no_hash_verify,
        )
        payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0
    except PlanError as error:
        print(
            json.dumps(
                {
                    "format": OUTPUT_FORMAT,
                    "passed": False,
                    "gpu_allowed": False,
                    "error": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
