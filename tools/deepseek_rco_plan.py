#!/usr/bin/env python3
"""Plan a DeepSeek-V4.1-Flash expert-pruning budget for Strata's RCO track.

This is a local, dependency-free preflight planner. It reads only two JSON
metadata files from a candidate checkpoint, ``config.json`` and
``model.safetensors.index.json``, classifies every indexed tensor against the
checked-in shape contract in ``data/deepseek_v41_manifest.json``, and derives
exact integer budgets for the routed-expert pruning that RCO would search over.

It reads index metadata and config metadata only. It never opens, hashes, or
runs a checkpoint shard, never converts or quantizes a checkpoint, and makes no
claim that any Strata backend can load DeepSeek-V4.1-Flash. The output document
is the ``strata-deepseek-rco-plan-v1`` plan format; it is a budget contract,
not a selected expert mask.

All arithmetic is exact. Retained fractions, per-active expansions, and byte
totals are computed with integer numerator/denominator pairs and
``fractions.Fraction``; no float rounding decides a retained-expert count.

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
SCRIPTS = ROOT / "scripts"
DEFAULT_MANIFEST = ROOT / "data" / "deepseek_v41_manifest.json"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_deepseek  # noqa: E402

PLAN_FORMAT = "strata-deepseek-rco-plan-v1"
SCOPE = (
    "Reads one candidate DeepSeek-V4.1-Flash config.json and "
    "model.safetensors.index.json, classifies every indexed tensor against the "
    "checked-in 1017 shape manifest, and derives exact integer expert-pruning "
    "budgets. It does not open, hash, download, convert, quantize, or run any "
    "checkpoint shard, does not select an expert mask, does not run an RCO "
    "search, and does not claim that any Strata backend can load "
    "DeepSeek-V4.1-Flash."
)

# ---------------------------------------------------------------- checkpoint shape

#: The pinned 1017 geometry this planner is written against.
BACKBONE_LAYERS = 40
MTP_LAYERS = 3
BACKBONE_ROUTED_EXPERTS = 384
BACKBONE_ACTIVE_EXPERTS = 6
MTP_ROUTED_EXPERTS = 128
MTP_ACTIVE_EXPERTS = 3
HIDDEN_SIZE = 5120
MOE_INTERMEDIATE_SIZE = 2304

#: Qwen3.8-Flash-Next comparator: 288 of 512 routed experts kept, 10 active.
QWEN_COMPARATOR = {
    "name": "qwen3.8-flash-next-coder",
    "routed_experts": 512,
    "retained_experts": 288,
    "active_experts": 10,
}

#: Expected index inventory of the pinned 1017 checkpoint.
EXPECTED_TENSOR_COUNT = 96_085
EXPECTED_BACKBONE_ROUTED_TENSORS = 92_160
EXPECTED_MTP_ROUTED_TENSORS = 2_304
EXPECTED_SHARED_EXPERT_TENSORS = 258
EXPECTED_DENSE_TENSORS = 1_363

# ---------------------------------------------------------------- expert packing

#: Authoritative 1017 shard-header geometry for one routed expert.
#:
#: Each of w1, w2, w3 is an I8 tensor holding packed FP4 values, paired with an
#: F8_E8M0 scale tensor. The scale tensors are per-output-row scales, not
#: 32x32 block scales, so the packed byte count is the product of the two
#: declared shapes rather than a blocked reconstruction.
EXPERT_TENSOR_GEOMETRY: dict[str, dict[str, Any]] = {
    "w1": {
        "weight": {"dtype": "I8", "shape": (2304, 2560)},
        "scale": {"dtype": "F8_E8M0", "shape": (2304, 160)},
    },
    "w2": {
        "weight": {"dtype": "I8", "shape": (5120, 1152)},
        "scale": {"dtype": "F8_E8M0", "shape": (5120, 72)},
    },
    "w3": {
        "weight": {"dtype": "I8", "shape": (2304, 2560)},
        "scale": {"dtype": "F8_E8M0", "shape": (2304, 160)},
    },
}

ROUTED_TENSOR_RE = re.compile(
    r"^(?P<scope>layers|mtp)\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.w(?P<gate>[123])"
    r"\.(?P<part>weight|scale)$"
)
SHARED_TENSOR_RE = re.compile(
    r"^(?P<scope>layers|mtp)\.(?P<layer>\d+)\.ffn\.shared_experts\.w(?P<gate>[123])"
    r"\.(?P<part>weight|scale)$"
)


class PlanError(Exception):
    """A candidate checkpoint or a requested budget is unusable."""


@dataclass(frozen=True)
class TensorClass:
    """How one indexed tensor is classified against the shape contract."""

    name: str
    scope: str
    kind: str
    layer: int | None
    expert: int | None
    part: str | None
    dtype: str | None
    elements: int | None
    packed_bytes: int | None


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expert_tensor_bytes(projection: str) -> int:
    """Return the exact packed byte count of one expert's ``projection`` tensors."""
    geometry = EXPERT_TENSOR_GEOMETRY.get(projection)
    if geometry is None:
        raise PlanError(f"unknown expert projection {projection!r}")
    total = 0
    for part in ("weight", "scale"):
        spec = geometry[part]
        count = 1
        for extent in spec["shape"]:
            count *= extent
        total += count
    return total


def expert_bytes() -> int:
    """Return the exact packed byte count of one routed expert."""
    return sum(expert_tensor_bytes(projection) for projection in ("w1", "w2", "w3"))


# ---------------------------------------------------------------- exact arithmetic


def round_ratio(value: Fraction, mode: str) -> int:
    """Round an exact fraction to an integer under a named mode.

    ``floor`` and ``ceil`` are exact. ``nearest`` rounds half away from zero,
    which is the only tie rule that keeps the retained count on the larger side
    of an exact half and therefore never under-retains relative to the
    requested fraction.
    """
    if mode == "floor":
        return value.numerator // value.denominator
    if mode == "ceil":
        return -((-value.numerator) // value.denominator)
    if mode == "nearest":
        if value.denominator == 1:
            return value.numerator
        doubled = 2 * value.numerator
        if doubled >= value.denominator:
            return (doubled + value.denominator) // (2 * value.denominator)
        return (doubled - value.denominator) // (2 * value.denominator)
    raise PlanError(f"unknown rounding mode {mode!r}")


def retained_from_fraction(routed_experts: int, fraction: Fraction, mode: str) -> int:
    """Return the retained expert count for an exact retained fraction."""
    if not 0 < fraction <= 1:
        raise PlanError(f"retained fraction {fraction} must lie in (0, 1]")
    return round_ratio(Fraction(routed_experts) * fraction, mode)


def retained_from_active_ratio(active_experts: int, ratio: Fraction, mode: str, *, cap: int) -> int:
    """Return the retained expert count for an exact retained-per-active ratio.

    The ratio is the retained expert count divided by the number of experts a
    token activates, so a ratio of one keeps every expert. The retained count
    is therefore the ratio times the active count, capped at the routed expert
    count.
    """
    if ratio <= 0:
        raise PlanError(f"retained-per-active ratio {ratio} must be positive")
    if active_experts <= 0:
        raise PlanError("the active expert count must be positive")
    retained = round_ratio(Fraction(active_experts) * ratio, mode)
    return min(retained, cap)


def exact_bytes(fraction: Fraction, total: int) -> int:
    """Return the exact byte count of ``fraction`` of ``total`` bytes."""
    scaled = Fraction(total) * fraction
    if scaled.denominator == 1:
        return scaled.numerator
    raise PlanError(f"fraction {fraction} of {total} bytes is not an integer")


# ---------------------------------------------------------------- classification


def _classify_dense(name: str) -> TensorClass:
    return TensorClass(
        name=name,
        scope="dense",
        kind="dense",
        layer=None,
        expert=None,
        part=None,
        dtype=None,
        elements=None,
        packed_bytes=None,
    )


def classify_tensor(name: str) -> TensorClass:
    """Classify one indexed tensor name against the 1017 routed/shared taxonomy.

    The routed and shared expert patterns are matched exactly. Every other
    name is dense. A routed name whose layer or expert index is out of the
    declared range is refused rather than silently treated as dense.
    """
    routed = ROUTED_TENSOR_RE.match(name)
    if routed is not None:
        scope = routed.group("scope")
        layer = int(routed.group("layer"))
        expert = int(routed.group("expert"))
        part = routed.group("part")
        projection = f"w{routed.group('gate')}"
        limit = BACKBONE_ROUTED_EXPERTS if scope == "layers" else MTP_ROUTED_EXPERTS
        layers = BACKBONE_LAYERS if scope == "layers" else MTP_LAYERS
        if not 0 <= layer < layers:
            raise PlanError(f"{name}: {scope} layer {layer} is outside [0, {layers})")
        if not 0 <= expert < limit:
            raise PlanError(f"{name}: {scope} expert {expert} is outside [0, {limit})")
        geometry = EXPERT_TENSOR_GEOMETRY[projection][part]
        elements = 1
        for extent in geometry["shape"]:
            elements *= extent
        return TensorClass(
            name=name,
            scope=scope,
            kind="routed",
            layer=layer,
            expert=expert,
            part=part,
            dtype=geometry["dtype"],
            elements=elements,
            packed_bytes=elements,
        )
    shared = SHARED_TENSOR_RE.match(name)
    if shared is not None:
        scope = shared.group("scope")
        layer = int(shared.group("layer"))
        layers = BACKBONE_LAYERS if scope == "layers" else MTP_LAYERS
        if not 0 <= layer < layers:
            raise PlanError(f"{name}: {scope} layer {layer} is outside [0, {layers})")
        return TensorClass(
            name=name,
            scope=scope,
            kind="shared",
            layer=layer,
            expert=None,
            part=shared.group("part"),
            dtype=None,
            elements=None,
            packed_bytes=None,
        )
    return _classify_dense(name)


def classify_inventory(names: Iterable[str]) -> list[TensorClass]:
    """Classify every indexed tensor name."""
    return [classify_tensor(name) for name in names]


# ---------------------------------------------------------------- inventory gate


def load_json(path: Path, label: str) -> Any:
    """Read one JSON document, refusing a malformed file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PlanError(f"cannot read {label} at {path}: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise PlanError(f"{label} at {path} is not valid JSON: {error}") from error


def expected_tensor_names(manifest: dict[str, Any]) -> dict[str, str]:
    """Return the expected tensor name to owning family mapping."""
    try:
        plan = check_deepseek.build_plan(manifest)
        return check_deepseek.expected_tensor_names(plan)
    except (check_deepseek.ManifestError, KeyError, TypeError, ValueError) as error:
        raise PlanError(f"the shape manifest is unusable: {error}") from error


def gate_inventory(
    weight_map: dict[str, Any], manifest: dict[str, Any]
) -> tuple[list[TensorClass], dict[str, Any]]:
    """Classify the indexed inventory and refuse an unknown or missing tensor.

    Every indexed tensor must be present in the checked-in shape contract and
    every tensor the contract expects must be indexed. A tensor that matches no
    family, or an expected tensor that is absent, is a refusal.
    """
    expected = expected_tensor_names(manifest)
    unknown = sorted(name for name in weight_map if name not in expected)
    if unknown:
        raise PlanError(
            f"{len(unknown)} indexed tensors match no known family, first is {unknown[0]!r}"
        )
    missing = sorted(name for name in expected if name not in weight_map)
    if missing:
        raise PlanError(
            f"{len(missing)} expected tensors are absent from the index, first is {missing[0]!r}"
        )
    classes = classify_inventory(weight_map)
    counts = {
        "routed_backbone": 0,
        "routed_mtp": 0,
        "shared_expert": 0,
        "dense": 0,
    }
    for item in classes:
        if item.kind == "routed":
            counts["routed_backbone" if item.scope == "layers" else "routed_mtp"] += 1
        elif item.kind == "shared":
            counts["shared_expert"] += 1
        else:
            counts["dense"] += 1
    return classes, counts


def read_index(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the safetensors index and refuse a malformed document."""
    document = load_json(path, "index")
    if not isinstance(document, dict):
        raise PlanError(f"index at {path} is not a JSON object")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise PlanError(f"index at {path} has no metadata object")
    total_size = metadata.get("total_size")
    if isinstance(total_size, bool) or not isinstance(total_size, int):
        raise PlanError(f"index at {path} has no integer metadata.total_size")
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PlanError(f"index at {path} has no non-empty weight_map object")
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise PlanError(f"index at {path} has a non-string tensor name or shard file")
    return document, weight_map


def read_config(path: Path) -> dict[str, Any]:
    """Read the checkpoint config and refuse a malformed document."""
    document = load_json(path, "config")
    if not isinstance(document, dict):
        raise PlanError(f"config at {path} is not a JSON object")
    text = document.get("text_config")
    if not isinstance(text, dict):
        raise PlanError(f"config at {path} has no text_config object")
    return document


def config_geometry(text: dict[str, Any]) -> dict[str, int]:
    """Extract the geometry this planner reasons about from a text config."""
    keys = {
        "backbone_layers": ("num_hidden_layers",),
        "backbone_routed_experts": ("n_routed_experts",),
        "backbone_active_experts": ("num_experts_per_tok",),
        "mtp_layers": ("num_nextn_predict_layers",),
        "mtp_routed_experts": ("dspark_n_routed_experts",),
        "mtp_active_experts": ("dspark_num_experts_per_tok",),
        "hidden_size": ("hidden_size",),
        "moe_intermediate_size": ("moe_intermediate_size",),
    }
    geometry: dict[str, int] = {}
    for label, candidates in keys.items():
        for candidate in candidates:
            value = text.get(candidate)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            geometry[label] = value
            break
        else:
            raise PlanError(f"text_config has no integer {candidates[0]!r}")
    return geometry


# ---------------------------------------------------------------- budgets


@dataclass(frozen=True)
class Budget:
    """One derived pruning budget for the backbone routed experts."""

    name: str
    basis: str
    retained_experts: int
    removed_experts: int
    retained_fraction: Fraction
    active_experts: int
    retained_per_active: Fraction
    retained_bytes: int
    removed_bytes: int

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "basis": self.basis,
            "retained_experts": self.retained_experts,
            "removed_experts": self.removed_experts,
            "retained_fraction": {
                "numerator": self.retained_fraction.numerator,
                "denominator": self.retained_fraction.denominator,
                "decimal": float(self.retained_fraction),
            },
            "active_experts": self.active_experts,
            "retained_per_active": {
                "numerator": self.retained_per_active.numerator,
                "denominator": self.retained_per_active.denominator,
                "decimal": float(self.retained_per_active),
            },
            "retained_bytes": self.retained_bytes,
            "removed_bytes": self.removed_bytes,
            "retained_bytes_decimal_gb": self.retained_bytes / 1_000_000_000,
            "retained_bytes_gib": self.retained_bytes / 2**30,
        }


def build_budget(
    name: str,
    basis: str,
    retained_experts: int,
    routed_experts: int,
    active_experts: int,
    bytes_per_expert: int,
) -> Budget:
    """Assemble one budget with exact fractions and exact byte counts."""
    if not 0 < retained_experts <= routed_experts:
        raise PlanError(
            f"budget {name}: retained {retained_experts} is outside (0, {routed_experts}]"
        )
    if not 0 < active_experts <= retained_experts:
        raise PlanError(
            f"budget {name}: {active_experts} active experts exceed {retained_experts} retained"
        )
    retained_fraction = Fraction(retained_experts, routed_experts)
    retained_per_active = Fraction(retained_experts, active_experts)
    retained_bytes = exact_bytes(retained_fraction, routed_experts * bytes_per_expert)
    removed_bytes = exact_bytes(
        Fraction(routed_experts - retained_experts, routed_experts),
        routed_experts * bytes_per_expert,
    )
    return Budget(
        name=name,
        basis=basis,
        retained_experts=retained_experts,
        removed_experts=routed_experts - retained_experts,
        retained_fraction=retained_fraction,
        active_experts=active_experts,
        retained_per_active=retained_per_active,
        retained_bytes=retained_bytes,
        removed_bytes=removed_bytes,
    )


def derive_budgets(
    *,
    routed_experts: int,
    active_experts: int,
    bytes_per_expert: int,
    retained_fraction: Fraction,
    active_ratio: Fraction,
) -> list[Budget]:
    """Derive the exact budget candidates for the backbone routed experts.

    Two independent bases are derived. The first keeps an exact retained
    fraction of the routed experts. The second holds the ratio of retained
    experts to active experts at an exact value, which is the quantity an RCO
    search trades against; it is reported under all three rounding modes.
    """
    budgets: list[Budget] = []
    fraction_retained = retained_from_fraction(routed_experts, retained_fraction, "floor")
    budgets.append(
        build_budget(
            name="retained-fraction",
            basis=(
                f"exact retained fraction {retained_fraction.numerator}/"
                f"{retained_fraction.denominator} of {routed_experts} routed experts, "
                "rounded down"
            ),
            retained_experts=fraction_retained,
            routed_experts=routed_experts,
            active_experts=active_experts,
            bytes_per_expert=bytes_per_expert,
        )
    )
    for mode in ("floor", "nearest", "ceil"):
        retained = retained_from_active_ratio(
            active_experts, active_ratio, mode, cap=routed_experts
        )
        budgets.append(
            build_budget(
                name=f"retained-per-active-{mode}",
                basis=(
                    f"exact retained-per-active ratio {active_ratio.numerator}/"
                    f"{active_ratio.denominator} with {active_experts} active experts, "
                    f"rounded {mode}"
                ),
                retained_experts=retained,
                routed_experts=routed_experts,
                active_experts=active_experts,
                bytes_per_expert=bytes_per_expert,
            )
        )
    return budgets


def comparator_retained_fraction(comparator: dict[str, Any]) -> Fraction:
    """Return the exact retained fraction of the Qwen comparator."""
    routed = int(comparator["routed_experts"])
    retained = int(comparator["retained_experts"])
    active = int(comparator["active_experts"])
    if not 0 < retained <= routed:
        raise PlanError("the Qwen comparator retained count is outside its routed count")
    if not 0 < active <= retained:
        raise PlanError("the Qwen comparator active count exceeds its retained count")
    return Fraction(retained, routed)


def comparator_retained_per_active(comparator: dict[str, Any]) -> Fraction:
    """Return the exact retained-per-active ratio of the Qwen comparator."""
    retained = int(comparator["retained_experts"])
    active = int(comparator["active_experts"])
    if not 0 < active <= retained:
        raise PlanError("the Qwen comparator active count exceeds its retained count")
    return Fraction(retained, active)


# ---------------------------------------------------------------- plan


def build_plan(
    *,
    config_path: Path,
    index_path: Path,
    manifest_path: Path,
    retained_fraction: Fraction,
    active_ratio: Fraction,
    preserve_mtp: bool,
) -> dict[str, Any]:
    """Build the ``strata-deepseek-rco-plan-v1`` document."""
    manifest = load_json(manifest_path, "manifest")
    if not isinstance(manifest, dict):
        raise PlanError(f"manifest at {manifest_path} is not a JSON object")
    config = read_config(config_path)
    _, weight_map = read_index(index_path)
    classes, counts = gate_inventory(weight_map, manifest)

    geometry = config_geometry(config["text_config"])
    if geometry["backbone_layers"] != BACKBONE_LAYERS:
        raise PlanError(
            f"config declares {geometry['backbone_layers']} backbone layers, "
            f"this planner is written for {BACKBONE_LAYERS}"
        )
    if geometry["backbone_routed_experts"] != BACKBONE_ROUTED_EXPERTS:
        raise PlanError(
            f"config declares {geometry['backbone_routed_experts']} backbone routed "
            f"experts, this planner is written for {BACKBONE_ROUTED_EXPERTS}"
        )
    if geometry["backbone_active_experts"] != BACKBONE_ACTIVE_EXPERTS:
        raise PlanError(
            f"config declares {geometry['backbone_active_experts']} active experts, "
            f"this planner is written for {BACKBONE_ACTIVE_EXPERTS}"
        )
    if geometry["hidden_size"] != HIDDEN_SIZE:
        raise PlanError(
            f"config declares hidden size {geometry['hidden_size']}, "
            f"this planner is written for {HIDDEN_SIZE}"
        )
    if geometry["moe_intermediate_size"] != MOE_INTERMEDIATE_SIZE:
        raise PlanError(
            f"config declares moe intermediate size {geometry['moe_intermediate_size']}, "
            f"this planner is written for {MOE_INTERMEDIATE_SIZE}"
        )

    total_tensors = len(classes)
    if total_tensors != EXPECTED_TENSOR_COUNT:
        raise PlanError(
            f"index holds {total_tensors} tensors, the pinned 1017 inventory holds "
            f"{EXPECTED_TENSOR_COUNT}"
        )
    if counts["routed_backbone"] != EXPECTED_BACKBONE_ROUTED_TENSORS:
        raise PlanError(
            f"index holds {counts['routed_backbone']} backbone routed-expert tensors, "
            f"expected {EXPECTED_BACKBONE_ROUTED_TENSORS}"
        )
    if counts["routed_mtp"] != EXPECTED_MTP_ROUTED_TENSORS:
        raise PlanError(
            f"index holds {counts['routed_mtp']} MTP routed-expert tensors, "
            f"expected {EXPECTED_MTP_ROUTED_TENSORS}"
        )
    if counts["shared_expert"] != EXPECTED_SHARED_EXPERT_TENSORS:
        raise PlanError(
            f"index holds {counts['shared_expert']} shared-expert tensors, "
            f"expected {EXPECTED_SHARED_EXPERT_TENSORS}"
        )
    if counts["dense"] != EXPECTED_DENSE_TENSORS:
        raise PlanError(
            f"index holds {counts['dense']} dense tensors, expected {EXPECTED_DENSE_TENSORS}"
        )

    bytes_per_expert = expert_bytes()
    budgets = derive_budgets(
        routed_experts=BACKBONE_ROUTED_EXPERTS,
        active_experts=BACKBONE_ACTIVE_EXPERTS,
        bytes_per_expert=bytes_per_expert,
        retained_fraction=retained_fraction,
        active_ratio=active_ratio,
    )

    mtp_total = MTP_LAYERS * MTP_ROUTED_EXPERTS * bytes_per_expert
    mtp_plan: dict[str, Any] = {
        "preserved": preserve_mtp,
        "layers": MTP_LAYERS,
        "routed_experts": MTP_ROUTED_EXPERTS,
        "active_experts": MTP_ACTIVE_EXPERTS,
        "bytes_per_expert": bytes_per_expert,
        "total_bytes": mtp_total,
        "policy": (
            "MTP and DSpark routed experts are preserved in full by default; they are "
            "the draft path and pruning them changes the draft distribution."
            if preserve_mtp
            else "MTP and DSpark routed experts are excluded from the pruning scope "
            "but counted as preserved bytes."
        ),
    }

    backbone_total = BACKBONE_LAYERS * BACKBONE_ROUTED_EXPERTS * bytes_per_expert

    return {
        "format": PLAN_FORMAT,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {"system": platform.platform(), "python": sys.version},
        "scope": SCOPE,
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "index": str(index_path),
            "index_sha256": sha256(index_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        },
        "geometry": {
            "backbone_layers": BACKBONE_LAYERS,
            "backbone_routed_experts": BACKBONE_ROUTED_EXPERTS,
            "backbone_active_experts": BACKBONE_ACTIVE_EXPERTS,
            "mtp_layers": MTP_LAYERS,
            "mtp_routed_experts": MTP_ROUTED_EXPERTS,
            "mtp_active_experts": MTP_ACTIVE_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "config_declared": geometry,
        },
        "inventory": {
            "total_tensors": total_tensors,
            "routed_backbone_tensors": counts["routed_backbone"],
            "routed_mtp_tensors": counts["routed_mtp"],
            "shared_expert_tensors": counts["shared_expert"],
            "dense_tensors": counts["dense"],
            "shards": len(set(weight_map.values())),
            "classification": "strict; every indexed tensor matched exactly one family",
        },
        "packing": {
            "expert_dtype": "I8-packed FP4 values with F8_E8M0 scales",
            "scale_layout": "per-output-row scales, not 32x32 block scales",
            "tensors": {
                projection: {
                    "weight": {
                        "dtype": spec["weight"]["dtype"],
                        "shape": list(spec["weight"]["shape"]),
                        "bytes": expert_tensor_bytes(projection) - _part_bytes(projection, "scale"),
                    },
                    "scale": {
                        "dtype": spec["scale"]["dtype"],
                        "shape": list(spec["scale"]["shape"]),
                        "bytes": _part_bytes(projection, "scale"),
                    },
                }
                for projection, spec in EXPERT_TENSOR_GEOMETRY.items()
            },
            "bytes_per_expert": bytes_per_expert,
            "backbone_routed_expert_bytes": backbone_total,
            "mtp_routed_expert_bytes": mtp_total,
        },
        "budgets": [budget.as_json() for budget in budgets],
        "mtp": mtp_plan,
        "comparator": {
            "name": QWEN_COMPARATOR["name"],
            "routed_experts": QWEN_COMPARATOR["routed_experts"],
            "retained_experts": QWEN_COMPARATOR["retained_experts"],
            "active_experts": QWEN_COMPARATOR["active_experts"],
            "retained_fraction": {
                "numerator": QWEN_COMPARATOR["retained_experts"],
                "denominator": QWEN_COMPARATOR["routed_experts"],
                "decimal": QWEN_COMPARATOR["retained_experts"] / QWEN_COMPARATOR["routed_experts"],
            },
            "retained_per_active": {
                "numerator": QWEN_COMPARATOR["retained_experts"],
                "denominator": QWEN_COMPARATOR["active_experts"],
                "decimal": QWEN_COMPARATOR["retained_experts"] / QWEN_COMPARATOR["active_experts"],
            },
            "note": (
                "The Qwen comparator is a different model family with a different "
                "expert geometry. It bounds the retained-per-active ratio this "
                "planner applies; it is not a DeepSeek-V4.1 budget."
            ),
        },
        "non_claims": [
            "No checkpoint shard was opened, hashed, downloaded, converted, quantized, "
            "or run. Only config.json and model.safetensors.index.json were read.",
            "No expert mask was selected and no RCO search ran. The budgets are "
            "candidates for a search, not a search result.",
            "The byte counts are arithmetic over the declared shard-header geometry. "
            "They are not a measurement of any produced artifact.",
            "No quality, perplexity, KL, accuracy, throughput, or memory-fit claim is "
            "made for any budget.",
            "This does not claim that any Strata backend can load DeepSeek-V4.1-Flash.",
        ],
    }


def _part_bytes(projection: str, part: str) -> int:
    """Return the exact byte count of one part of one expert projection."""
    geometry = EXPERT_TENSOR_GEOMETRY[projection][part]
    count = 1
    for extent in geometry["shape"]:
        count *= extent
    return count


def parse_fraction(value: str, label: str) -> Fraction:
    """Parse an exact ratio such as ``9/16`` or ``0.5625``."""
    text = value.strip()
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        try:
            return Fraction(int(numerator), int(denominator))
        except (ValueError, ZeroDivisionError) as error:
            raise PlanError(f"{label} {value!r} is not an exact ratio") from error
    try:
        return Fraction(text)
    except ValueError as error:
        raise PlanError(f"{label} {value!r} is not an exact ratio") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="candidate config.json")
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        help="candidate model.safetensors.index.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"checked-in shape manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--retained-fraction",
        default=None,
        help=(
            "exact retained fraction of the routed experts; defaults to the Qwen "
            "comparator's 288 of 512"
        ),
    )
    parser.add_argument(
        "--active-ratio",
        default=None,
        help=(
            "exact retained-per-active ratio; defaults to the Qwen comparator's "
            "288 retained over 10 active"
        ),
    )
    parser.add_argument(
        "--prune-mtp",
        action="store_true",
        help="record that the MTP and DSpark routed experts are pruned too",
    )
    parser.add_argument("--out", type=Path, help="write the plan JSON to this path")
    parser.add_argument("--json", action="store_true", help="print the plan JSON on stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.retained_fraction is None:
            retained_fraction = comparator_retained_fraction(QWEN_COMPARATOR)
        else:
            retained_fraction = parse_fraction(arguments.retained_fraction, "retained fraction")
        if arguments.active_ratio is None:
            active_ratio = comparator_retained_per_active(QWEN_COMPARATOR)
        else:
            active_ratio = parse_fraction(arguments.active_ratio, "active ratio")
        plan = build_plan(
            config_path=arguments.config,
            index_path=arguments.index,
            manifest_path=arguments.manifest,
            retained_fraction=retained_fraction,
            active_ratio=active_ratio,
            preserve_mtp=not arguments.prune_mtp,
        )
    except PlanError as error:
        print(f"FAIL deepseek-rco-plan: {error}", file=sys.stderr)
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
