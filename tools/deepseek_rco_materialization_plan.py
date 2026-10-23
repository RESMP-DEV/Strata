#!/usr/bin/env python3
"""Build a metadata-only materialization plan for DeepSeek RCO expert removal."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
SCRIPTS = ROOT / "scripts"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import deepseek_rco_plan as base  # noqa: E402

PLAN_FORMAT = "strata-deepseek-rco-materialization-plan-v1"
MASK_FORMAT = "strata-deepseek-rco-mask-v1"
SCOPE = (
    "Metadata-only planning contract; no shard opened. Selects and groups tensor "
    "actions from explicit JSON inputs, computes exact pinned routed-expert bytes, "
    "exact backbone router-slice bytes, and the exact target tensor payload before "
    "new safetensors headers, and defers target headers, offsets, shard boundaries, "
    "and file sizes to streaming-header execution."
)

PINNED_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
PINNED_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
DECLARED_INDEX_TOTAL_SIZE_BYTES = 510_286_023_000
DECLARED_MANIFEST_TOTAL_SIZE_BYTES = 510_311_045_295
SHARD_PATTERN = re.compile(r"^model-\d{5}-of-00048\.safetensors$")
BACKBONE_GATE_RE = re.compile(r"^layers\.(?P<layer>\d+)\.ffn\.gate\.(?P<part>weight|bias|bias_vl)$")
GATE_PARTS = ("weight", "bias", "bias_vl")
#: Verified 1017 shard-header geometry of one backbone router (gate) tuple.
#:
#: Every backbone layer carries exactly one gate tuple: a BF16 [384, 5120] weight
#: plus two F32 [384] biases (``ffn.gate.bias`` and ``ffn.gate.bias_vl``). The
#: 48-shard header verification receipt
#: ``evidence/deepseek-v41-rco-header-verification-2026-10-23.json`` counted 80
#: F32 [384] gate tensors with zero dtype or shape failures, and its
#: ``total_data_bytes`` equals the pinned index total, so the index total is
#: tensor payload bytes without headers. Row slicing rewrites only the first
#: extent of each shape.
BACKBONE_GATE_GEOMETRY: dict[str, dict[str, Any]] = {
    "weight": {"dtype": "BF16", "dtype_bytes": 2, "shape": (384, 5120)},
    "bias": {"dtype": "F32", "dtype_bytes": 4, "shape": (384,)},
}
BACKBONE_GATE_BIAS_COUNT = 2
CONFIG_INVARIANTS = {
    "backbone_layers": (40, "num_hidden_layers"),
    "backbone_routed_experts": (384, "n_routed_experts"),
    "backbone_active_experts": (6, "num_experts_per_tok"),
    "mtp_layers": (3, "num_nextn_predict_layers"),
    "mtp_routed_experts": (128, "dspark_n_routed_experts"),
    "mtp_active_experts": (3, "dspark_num_experts_per_tok"),
    "hidden_size": (5120, "hidden_size"),
    "moe_intermediate_size": (2304, "moe_intermediate_size"),
}
HEADER_DEFERRED = "requires_streaming_header"


def require_file(path: Path, label: str) -> None:
    """Require an explicitly named regular JSON file without discovering siblings."""
    if path.is_dir():
        raise base.PlanError(f"{label} path {path} is a directory; supply the explicit JSON file")
    if not path.is_file():
        raise base.PlanError(f"{label} path {path} is not a regular file")


def utc_now() -> str:
    """Return a UTC timestamp in the plan format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    """Validate the manifest fields this planner uses for shard honesty."""
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise base.PlanError(f"manifest at {path} has no model object")
    if model.get("repo") != PINNED_REPO:
        raise base.PlanError(f"manifest repo {model.get('repo')!r} does not match {PINNED_REPO!r}")
    if model.get("revision") != PINNED_REVISION:
        raise base.PlanError(
            f"manifest revision {model.get('revision')!r} does not match {PINNED_REVISION!r}"
        )

    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise base.PlanError(f"manifest at {path} has no checkpoint object")
    if _integer(checkpoint.get("shards")) != 48:
        raise base.PlanError("manifest does not declare exactly 48 checkpoint shards")
    if checkpoint.get("shard_file_pattern") != "model-{index:05d}-of-{total:05d}.safetensors":
        raise base.PlanError(
            "manifest shard-file pattern does not match the pinned 48-shard pattern"
        )
    if _integer(checkpoint.get("shard_manifest_total_bytes")) != (
        DECLARED_MANIFEST_TOTAL_SIZE_BYTES
    ):
        raise base.PlanError("manifest has no integer checkpoint.shard_manifest_total_bytes")

    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict):
        raise base.PlanError(f"manifest at {path} has no tensors object")
    if _integer(tensors.get("total")) != base.EXPECTED_TENSOR_COUNT:
        raise base.PlanError("manifest does not declare the pinned 96085-tensor inventory")
    if _integer(tensors.get("index_total_size_bytes")) != DECLARED_INDEX_TOTAL_SIZE_BYTES:
        raise base.PlanError("manifest index total does not match the pinned declared total")
    shard_range = tensors.get("shard_tensor_count_range")
    if shard_range != [2, 2342]:
        raise base.PlanError("manifest shard tensor-count range does not match [2, 2342]")


def validate_config(config: Mapping[str, Any]) -> dict[str, int]:
    """Validate the exact pinned source geometry before emitting config patches."""
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise base.PlanError("config has no text_config object")
    geometry = base.config_geometry(text)
    for label, (expected, config_key) in CONFIG_INVARIANTS.items():
        actual = geometry[label]
        if actual != expected:
            raise base.PlanError(
                f"text_config.{config_key} is {actual}; the pinned materialization "
                f"contract requires {expected}"
            )
    return geometry


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    unknown = sorted(set(value) - keys)
    if unknown:
        raise base.PlanError(f"{label} has unknown fields: {', '.join(unknown)}")


def _keep_row(value: Any, layer: int) -> list[int]:
    if not isinstance(value, list):
        raise base.PlanError(f"backbone.keep_ids[{layer}] is not an array")
    row: list[int] = []
    previous: int | None = None
    for position, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise base.PlanError(
                f"backbone.keep_ids[{layer}][{position}] is not a non-boolean integer"
            )
        if not 0 <= item < base.BACKBONE_ROUTED_EXPERTS:
            raise base.PlanError(f"backbone.keep_ids[{layer}][{position}] is outside [0, 384)")
        if previous is not None and item <= previous:
            raise base.PlanError(
                f"backbone.keep_ids[{layer}] is not strictly increasing and unique"
            )
        row.append(item)
        previous = item
    return row


def read_mask(path: Path) -> dict[str, Any]:
    """Read and strictly validate the dependency-free RCO selection mask."""
    require_file(path, "mask")
    mask = base.load_json(path, "mask")
    if not isinstance(mask, dict):
        raise base.PlanError(f"mask at {path} is not a JSON object")
    _require_exact_keys(mask, {"format", "model", "backbone", "mtp"}, "mask")
    if mask.get("format") != MASK_FORMAT:
        raise base.PlanError(f"mask format must be exactly {MASK_FORMAT!r}")

    model = mask.get("model")
    if not isinstance(model, dict):
        raise base.PlanError("mask has no model object")
    _require_exact_keys(model, {"repo", "revision"}, "mask.model")
    if model.get("repo") != PINNED_REPO:
        raise base.PlanError(f"mask model.repo does not match {PINNED_REPO!r}")
    if model.get("revision") != PINNED_REVISION:
        raise base.PlanError(f"mask model.revision does not match {PINNED_REVISION!r}")

    backbone = mask.get("backbone")
    if not isinstance(backbone, dict):
        raise base.PlanError("mask has no backbone object")
    _require_exact_keys(
        backbone, {"layers", "routed_experts", "active_experts", "keep_ids"}, "mask.backbone"
    )
    if _integer(backbone.get("layers")) != base.BACKBONE_LAYERS:
        raise base.PlanError("mask backbone.layers must be exactly 40")
    if _integer(backbone.get("routed_experts")) != base.BACKBONE_ROUTED_EXPERTS:
        raise base.PlanError("mask backbone.routed_experts must be exactly 384")
    if _integer(backbone.get("active_experts")) != base.BACKBONE_ACTIVE_EXPERTS:
        raise base.PlanError("mask backbone.active_experts must be exactly 6")
    rows_value = backbone.get("keep_ids")
    if not isinstance(rows_value, list) or len(rows_value) != base.BACKBONE_LAYERS:
        raise base.PlanError("mask backbone.keep_ids must contain exactly 40 rows")
    rows = [_keep_row(row, layer) for layer, row in enumerate(rows_value)]
    retained_counts = {len(row) for row in rows}
    if len(retained_counts) != 1:
        counts = ", ".join(str(count) for count in sorted(retained_counts))
        raise base.PlanError(f"mask rows have differing retained counts: {counts}")
    retained = retained_counts.pop()
    if not base.BACKBONE_ACTIVE_EXPERTS <= retained <= base.BACKBONE_ROUTED_EXPERTS:
        raise base.PlanError(f"mask retained count {retained} is outside [6, 384]")

    mtp = mask.get("mtp")
    if not isinstance(mtp, dict):
        raise base.PlanError("mask has no mtp object")
    _require_exact_keys(mtp, {"policy", "layers", "routed_experts"}, "mask.mtp")
    if mtp.get("policy") != "preserve":
        raise base.PlanError("mask mtp.policy must be 'preserve'; MTP pruning is not designed")
    if _integer(mtp.get("layers")) != base.MTP_LAYERS:
        raise base.PlanError("mask mtp.layers must be exactly 3")
    if _integer(mtp.get("routed_experts")) != base.MTP_ROUTED_EXPERTS:
        raise base.PlanError("mask mtp.routed_experts must be exactly 128")
    return {"retained": retained, "keep_ids": rows}


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while refusing duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise base.PlanError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_index_without_collapsed_duplicates(path: Path) -> None:
    """Parse the index once more specifically to report duplicate JSON keys."""
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except json.JSONDecodeError as error:
        raise base.PlanError(f"index at {path} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise base.PlanError(f"index at {path} is not a JSON object")


def validate_index(
    document: Mapping[str, Any], weight_map: Mapping[str, str], manifest: Mapping[str, Any]
) -> tuple[list[base.TensorClass], dict[str, Any]]:
    """Validate the index metadata, shard identities, and strict tensor inventory."""
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise base.PlanError("index has no metadata object")
    if _integer(metadata.get("total_size")) != DECLARED_INDEX_TOTAL_SIZE_BYTES:
        raise base.PlanError(
            f"index metadata.total_size must be exactly {DECLARED_INDEX_TOTAL_SIZE_BYTES}"
        )

    shard_counts: dict[str, int] = {}
    for shard in weight_map.values():
        if not SHARD_PATTERN.fullmatch(shard) or Path(shard).name != shard:
            raise base.PlanError(f"index shard {shard!r} is not a plain pinned shard basename")
        shard_counts[shard] = shard_counts.get(shard, 0) + 1
    if len(shard_counts) != 48:
        raise base.PlanError(f"index has {len(shard_counts)} unique shards; expected 48")
    minimum, maximum = 2, 2342
    out_of_range = {
        shard: count for shard, count in shard_counts.items() if not minimum <= count <= maximum
    }
    if out_of_range:
        shard, count = sorted(out_of_range.items())[0]
        raise base.PlanError(f"shard {shard} has {count} tensors, outside [{minimum}, {maximum}]")

    classes, counts = base.gate_inventory(weight_map, manifest)
    if len(classes) != base.EXPECTED_TENSOR_COUNT:
        raise base.PlanError(
            f"index has {len(classes)} tensors; expected {base.EXPECTED_TENSOR_COUNT}"
        )
    expected_counts = {
        "routed_backbone": base.EXPECTED_BACKBONE_ROUTED_TENSORS,
        "routed_mtp": base.EXPECTED_MTP_ROUTED_TENSORS,
        "shared_expert": base.EXPECTED_SHARED_EXPERT_TENSORS,
        "dense": base.EXPECTED_DENSE_TENSORS,
    }
    for label, expected in expected_counts.items():
        if counts[label] != expected:
            raise base.PlanError(f"index {label} count is {counts[label]}; expected {expected}")
    return classes, counts


def backbone_gate_names() -> set[str]:
    """Return the exact 120 backbone router tensors that require row slicing."""
    return {
        f"layers.{layer}.ffn.gate.{part}"
        for layer in range(base.BACKBONE_LAYERS)
        for part in GATE_PARTS
    }


def backbone_gate_part_bytes(part: str, rows: int) -> int:
    """Return the exact payload bytes of one backbone gate part at ``rows``."""
    spec = BACKBONE_GATE_GEOMETRY.get(part)
    if spec is None:
        raise base.PlanError(f"unknown backbone gate part {part!r}")
    trailing = 1
    for extent in spec["shape"][1:]:
        trailing *= extent
    return rows * trailing * spec["dtype_bytes"]


def backbone_gate_tuple_bytes(rows: int) -> int:
    """Return the exact payload bytes of one backbone layer's gate tuple."""
    weight = backbone_gate_part_bytes("weight", rows)
    bias = backbone_gate_part_bytes("bias", rows)
    return weight + BACKBONE_GATE_BIAS_COUNT * bias


def backbone_gate_payload_bytes(rows: int) -> int:
    """Return the exact payload bytes of every backbone gate tuple at ``rows``."""
    if not 0 <= rows <= base.BACKBONE_ROUTED_EXPERTS:
        raise base.PlanError(f"backbone gate row count {rows} is outside [0, 384]")
    return base.BACKBONE_LAYERS * backbone_gate_tuple_bytes(rows)


def build_actions(
    classes: list[base.TensorClass], selection: dict[str, Any]
) -> dict[str, list[str]]:
    """Assign every classified tensor to exactly one materialization action."""
    keep_by_layer = {layer: set(row) for layer, row in enumerate(selection["keep_ids"])}
    gates = backbone_gate_names()
    actions: dict[str, list[str]] = {"omit": [], "copy": [], "slice_rows": []}
    for item in classes:
        if item.kind == "routed" and item.scope == "layers":
            assert item.layer is not None and item.expert is not None
            category = "copy" if item.expert in keep_by_layer[item.layer] else "omit"
        elif item.name in gates:
            if item.kind != "dense":
                raise base.PlanError(
                    f"backbone router tensor {item.name} did not classify as dense"
                )
            category = "slice_rows"
        else:
            category = "copy"
        actions[category].append(item.name)

    all_actions = actions["omit"] + actions["copy"] + actions["slice_rows"]
    if len(all_actions) != len(classes) or len(set(all_actions)) != len(classes):
        raise base.PlanError("internal action assignment did not cover every tensor exactly once")
    for names in actions.values():
        names.sort()
    return actions


def build_shard_groups(
    actions: dict[str, list[str]], weight_map: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Group complete source tensors by source shard and derive target emission."""
    grouped = {
        shard: {"omit": [], "copy": [], "slice_rows": []}
        for shard in sorted(set(weight_map.values()))
    }
    for category, names in actions.items():
        for name in names:
            grouped[weight_map[name]][category].append(name)

    groups: list[dict[str, Any]] = []
    for source, group in grouped.items():
        groups.append(
            {
                "source": source,
                "target": source,
                "emit": bool(group["copy"] or group["slice_rows"]),
                "source_tensor_count": sum(len(group[category]) for category in group),
                "target_tensor_count": len(group["copy"]) + len(group["slice_rows"]),
                **group,
            }
        )
    return groups


def build_plan(
    *,
    config_path: Path,
    index_path: Path,
    manifest_path: Path,
    mask_path: Path,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Build the metadata-only materialization plan document."""
    for label, path in (
        ("config", config_path),
        ("index", index_path),
        ("manifest", manifest_path),
        ("mask", mask_path),
    ):
        require_file(path, label)

    manifest = base.load_json(manifest_path, "manifest")
    if not isinstance(manifest, dict):
        raise base.PlanError(f"manifest at {manifest_path} is not a JSON object")
    validate_manifest(manifest, manifest_path)

    config = base.read_config(config_path)
    geometry = validate_config(config)
    selection = read_mask(mask_path)
    load_index_without_collapsed_duplicates(index_path)
    document, weight_map = base.read_index(index_path)
    classes, counts = validate_index(document, weight_map, manifest)
    actions = build_actions(classes, selection)
    shard_groups = build_shard_groups(actions, weight_map)

    bytes_per_w1 = base.expert_tensor_bytes("w1")
    bytes_per_w2 = base.expert_tensor_bytes("w2")
    bytes_per_w3 = base.expert_tensor_bytes("w3")
    bytes_per_expert = base.expert_bytes()
    retained = selection["retained"]
    pruned = base.BACKBONE_ROUTED_EXPERTS - retained
    retained_bytes = base.BACKBONE_LAYERS * retained * bytes_per_expert
    omitted_bytes = base.BACKBONE_LAYERS * pruned * bytes_per_expert
    mtp_bytes = base.MTP_LAYERS * base.MTP_ROUTED_EXPERTS * bytes_per_expert
    manifest_delta = DECLARED_MANIFEST_TOTAL_SIZE_BYTES - DECLARED_INDEX_TOTAL_SIZE_BYTES
    target_count = len(actions["copy"]) + len(actions["slice_rows"])
    source_gate_bytes = backbone_gate_payload_bytes(base.BACKBONE_ROUTED_EXPERTS)
    target_gate_bytes = backbone_gate_payload_bytes(retained)
    router_delta_bytes = source_gate_bytes - target_gate_bytes
    target_payload_bytes = DECLARED_INDEX_TOTAL_SIZE_BYTES - omitted_bytes - router_delta_bytes

    return {
        "format": PLAN_FORMAT,
        "recorded_at": recorded_at or utc_now(),
        "scope": SCOPE,
        "inputs": {
            "config": {
                "path": str(config_path),
                "sha256": base.sha256(config_path),
            },
            "index": {
                "path": str(index_path),
                "sha256": base.sha256(index_path),
                "declared_total_size_bytes": DECLARED_INDEX_TOTAL_SIZE_BYTES,
            },
            "manifest": {
                "path": str(manifest_path),
                "sha256": base.sha256(manifest_path),
            },
            "mask": {
                "path": str(mask_path),
                "sha256": base.sha256(mask_path),
            },
        },
        "selection": {
            "policy": "physical_expert_removal",
            "router_policy": "slice_axis_zero",
            "keep_per_backbone_layer": retained,
            "prune_per_backbone_layer": pruned,
            "router_gate_weight_rows_source": base.BACKBONE_ROUTED_EXPERTS,
            "router_gate_weight_rows_target": retained,
            "omit": actions["omit"],
            "copy": actions["copy"],
            "slice_rows": [
                {"name": name, "axis": 0, "selector": "backbone_keep_ids"}
                for name in actions["slice_rows"]
            ],
        },
        "shards": shard_groups,
        "tensor_count": {
            "source": len(classes),
            "omit": len(actions["omit"]),
            "copy": len(actions["copy"]),
            "slice_rows": len(actions["slice_rows"]),
            "target": target_count,
            "retained_backbone_routed": base.BACKBONE_LAYERS * retained * 6,
            "preserved_mtp_routed": counts["routed_mtp"],
            "shared_pass_through": counts["shared_expert"],
            "dense_total": counts["dense"],
            "dense_gate_slice": len(actions["slice_rows"]),
            "dense_gate_slice_tuples": base.BACKBONE_LAYERS,
            "dense_pass_through": counts["dense"] - len(actions["slice_rows"]),
        },
        "byte_accounting": {
            "basis": (
                "pinned routed header shapes; I8 and F8_E8M0 are one byte per declared element"
            ),
            "bytes_per_w1": bytes_per_w1,
            "bytes_per_w2": bytes_per_w2,
            "bytes_per_w3": bytes_per_w3,
            "bytes_per_routed_expert": bytes_per_expert,
            "retained_backbone_routed_bytes": retained_bytes,
            "omitted_backbone_routed_bytes": omitted_bytes,
            "preserved_mtp_routed_bytes": mtp_bytes,
            "declared_index_total_size_bytes": DECLARED_INDEX_TOTAL_SIZE_BYTES,
            "declared_index_total_minus_omitted_routed_bytes": (
                DECLARED_INDEX_TOTAL_SIZE_BYTES - omitted_bytes
            ),
            "declared_download_manifest_total_size_bytes": (DECLARED_MANIFEST_TOTAL_SIZE_BYTES),
            "download_manifest_minus_index_bytes": manifest_delta,
            "download_manifest_minus_index_status": "noted and unexplained; not reconciled",
            "backbone_gate_source_payload_bytes": source_gate_bytes,
            "backbone_gate_target_payload_bytes": target_gate_bytes,
            "backbone_gate_bytes_per_retained_row": (
                backbone_gate_payload_bytes(1) - backbone_gate_payload_bytes(0)
            ),
            "router_slice_delta_bytes": router_delta_bytes,
            "target_payload_bytes": target_payload_bytes,
            "target_payload_bytes_basis": (
                "declared index total minus omitted routed-expert bytes minus the "
                "backbone router slice delta; tensor payload bytes only"
            ),
            "target_payload_bytes_excludes": (
                "new safetensors headers, offsets, padding, shard boundaries, and shard file sizes"
            ),
            "target_header_bytes": HEADER_DEFERRED,
            "target_offsets": HEADER_DEFERRED,
            "target_shard_boundaries": HEADER_DEFERRED,
            "target_shard_file_bytes": HEADER_DEFERRED,
        },
        "mtp": {
            "policy": "preserve",
            "layers": base.MTP_LAYERS,
            "routed_experts_per_layer": base.MTP_ROUTED_EXPERTS,
            "active_experts_per_layer": base.MTP_ACTIVE_EXPERTS,
            "routed_tensors": counts["routed_mtp"],
            "routed_bytes": mtp_bytes,
            "router_and_shared_tensors": "copy_without_expert_selection",
        },
        "config_patches": [
            {
                "pointer": "/text_config/n_routed_experts",
                "source": geometry["backbone_routed_experts"],
                "target": retained,
            },
            {
                "pointer": "/text_config/original_num_experts",
                "operation": "set",
                "source": None,
                "target": geometry["backbone_routed_experts"],
            },
        ],
        "planning_non_claims": [
            "No checkpoint shard was opened, hashed, downloaded, converted, quantized, "
            "or executed.",
            "The pinned routed byte counts are not proof that source shard headers "
            "match those shapes.",
            "The backbone router byte counts use the pinned 1017 shard-header geometry "
            "recorded in the header verification receipt; this planner reads no shard "
            "header.",
            "The exact target payload byte count is tensor payload bytes only; it "
            "excludes new safetensors headers, offsets, padding, shard boundaries, and "
            "shard file sizes.",
            "No target safetensors header, offset, padding, shard boundary, or final "
            "file size is proven.",
            "No quality, perplexity, KL, accuracy, memory-fit, throughput, or "
            "loader-compatibility claim is made.",
            "This is not a Strata backend load claim.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="explicit config.json path")
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        help="explicit model.safetensors.index.json path",
    )
    parser.add_argument("--mask", required=True, type=Path, help="explicit RCO mask JSON path")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="explicit pinned shape-manifest JSON path",
    )
    parser.add_argument("--out", type=Path, help="write the plan JSON to this path")
    parser.add_argument("--json", action="store_true", help="print the plan JSON on stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        plan = build_plan(
            config_path=arguments.config,
            index_path=arguments.index,
            manifest_path=arguments.manifest,
            mask_path=arguments.mask,
        )
    except (base.PlanError, OSError) as error:
        print(f"FAIL deepseek-rco-materialization-plan: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(payload, encoding="utf-8")
    if arguments.json or arguments.out is None:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
