#!/usr/bin/env python3
"""Plan a Qwen3.8-Flash-Next phase-specialized prefiller artifact.

The published Qwen3.8-27B NVFP4 prefiller is a format and training template,
not a Strata runtime model.  This tool binds a plan to the deployed Flash-Next
decoder pack, selects the prompt-state tensors and per-layer hot routed experts,
and writes a deterministic JSON manifest with MXFP4 size accounting.  It reads
metadata only: no model payload, training data, accelerator, or benchmark is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import make_profile  # noqa: E402

N_LAYERS = 48
N_EXPERT = 512
SOURCE_EXPERT_COUNT = N_EXPERT
PHYSICAL_EXPERT_COUNTS = (256, 288, 384, 512)
EXPERT_ID_MAP_SCHEMA = "strata-expert-id-map-v1"
NATIVE_CONTEXT_TOKENS = 262_144
TRACE_TOP_K = 10
TRACE_MAGIC = 0x50525453
TRACE_VERSION = 2
TRACE_PHASE_PREFILL = 1
TRACE_FLAGS_NONE = 0
TRACE_RECORD_HEADER_BYTES = 32
TRACE_HEADER = struct.Struct("<IHHIQQQI")
TRACE_RECORD = struct.Struct("<iiqiiQ")
TRACE_FORMAT_LEGACY = "legacy-routing-v0"
TRACE_FORMAT_STRP_PREFILL = "strp-batched-prefill-v2"
HIDDEN = 2560
EXPERT_WIDTH = 640
MXFP4_GROUP = 32
MXFP4_GROUP_BYTES = 17
TEMPLATE_MODEL_MARKERS = ("qwen3.8-27b", "nvfp4-prefiller")

DENSE_ROLES: dict[str, tuple[str, ...]] = {
    "gdn_state": (
        ".attn_qkv.weight",
        ".attn_gate.weight",
        ".ssm_alpha.weight",
        ".ssm_beta.weight",
        ".ssm_out.weight",
    ),
    "qsa_state": (
        ".attn_q.weight",
        ".attn_k.weight",
        ".attn_v.weight",
        ".attn_output.weight",
        ".indexer.q_proj.weight",
        ".indexer.k_proj.weight",
    ),
    "hyper_connection": (
        ".hc_attn_down.weight",
        ".hc_attn_up.weight",
        ".hc_attn_inject.weight",
        ".hc_ffn_down.weight",
        ".hc_ffn_up.weight",
        ".hc_ffn_inject.weight",
    ),
    "shared_expert": (
        ".ffn_gate_shexp.weight",
        ".ffn_up_shexp.weight",
        ".ffn_down_shexp.weight",
    ),
}
DENSE_SUFFIXES = {suffix for suffixes in DENSE_ROLES.values() for suffix in suffixes}
EXPECTED_DENSE_SHAPES = {
    ".attn_gate.weight": (2560, 6144),
    ".attn_qkv.weight": (2560, 10240),
    ".ssm_alpha.weight": (2560, 48),
    ".ssm_beta.weight": (2560, 48),
    ".ssm_out.weight": (6144, 2560),
    ".attn_k.weight": (2560, 512),
    ".attn_output.weight": (6144, 2560),
    ".attn_q.weight": (2560, 12288),
    ".attn_v.weight": (2560, 512),
    ".indexer.k_proj.weight": (2560, 128),
    ".indexer.q_proj.weight": (2560, 512),
    ".hc_attn_down.weight": (10240, 320),
    ".hc_attn_inject.weight": (10240, 4),
    ".hc_attn_up.weight": (320, 10240),
    ".hc_ffn_down.weight": (10240, 320),
    ".hc_ffn_inject.weight": (10240, 4),
    ".hc_ffn_up.weight": (320, 10240),
    ".ffn_gate_shexp.weight": (2560, 640),
    ".ffn_up_shexp.weight": (2560, 640),
    ".ffn_down_shexp.weight": (640, 2560),
}
EXPECTED_DENSE_COUNT = 684
EXPECTED_DENSE_ELEMENTS = 3_571_384_320
FULL_EXPERT_ELEMENTS = 120_795_955_200


@dataclass(frozen=True)
class DenseTensor:
    name: str
    group: str
    elements: int


@dataclass(frozen=True)
class NativeExpertLayout:
    source: str
    n_expert: int
    layers: tuple[int, ...]
    total_bytes: int


@dataclass(frozen=True)
class ExpertIdMap:
    physical_count: int
    layers: int
    physical_to_source: tuple[tuple[int, ...], ...]
    source_count: int = SOURCE_EXPERT_COUNT
    basis: str = "identity_full_physical"
    sidecar_sha256: str | None = None
    mapping_sha256: str = ""

    def physical_for_source(self, layer: int, source_id: int) -> int:
        """Return the physical slot that owns a source expert in one layer."""
        try:
            return self.physical_to_source[layer].index(source_id)
        except ValueError as exc:
            raise ValueError(f"source expert {source_id} is not mapped at layer {layer}") from exc


def _mapping_digest(mapping: tuple[tuple[int, ...], ...]) -> str:
    canonical = json.dumps(mapping, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def identity_expert_id_map(physical_count: int) -> ExpertIdMap:
    """Return the only implicit mapping allowed: a complete 512-slot pack."""
    if physical_count != SOURCE_EXPERT_COUNT:
        raise ValueError(
            f"physical expert count {physical_count} requires an explicit "
            f"{EXPERT_ID_MAP_SCHEMA} sidecar; identity is valid only for 512 slots"
        )
    mapping = tuple(tuple(range(SOURCE_EXPERT_COUNT)) for _ in range(N_LAYERS))
    return ExpertIdMap(
        physical_count=physical_count,
        layers=N_LAYERS,
        physical_to_source=mapping,
        mapping_sha256=_mapping_digest(mapping),
    )


def _reject_json_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def read_expert_id_map(path: Path, physical_count: int) -> ExpertIdMap:
    """Read and fully validate a physical-to-source expert sidecar.

    `physical_to_source[layer][physical_id]` is the source expert ID.  The mapping is
    per-layer because a pruned pack has no global guarantee that physical slot order means
    the same source expert in every layer.
    """
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_json_duplicate,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid expert-ID map JSON: {exc}") from exc

    expected_keys = {
        "schema",
        "layers",
        "source_expert_count",
        "physical_expert_count",
        "physical_to_source",
    }
    if not isinstance(document, dict) or set(document) != expected_keys:
        observed = sorted(document) if isinstance(document, dict) else type(document).__name__
        raise ValueError(
            f"{path}: expert-ID map fields must be exactly {sorted(expected_keys)}, got {observed}"
        )

    def required_int(name: str) -> int:
        value = document[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path}: {name} must be an integer")
        return value

    schema = document["schema"]
    layers = required_int("layers")
    source_count = required_int("source_expert_count")
    sidecar_physical_count = required_int("physical_expert_count")
    mapping_value = document["physical_to_source"]

    if schema != EXPERT_ID_MAP_SCHEMA:
        raise ValueError(f"{path}: schema {schema!r}, expected {EXPERT_ID_MAP_SCHEMA!r}")
    if layers != N_LAYERS:
        raise ValueError(f"{path}: layers {layers}, expected {N_LAYERS}")
    if source_count != SOURCE_EXPERT_COUNT:
        raise ValueError(
            f"{path}: source_expert_count {source_count}, expected {SOURCE_EXPERT_COUNT}"
        )
    if sidecar_physical_count != physical_count:
        raise ValueError(
            f"{path}: physical_expert_count {sidecar_physical_count}, "
            f"native layout says {physical_count}"
        )
    if physical_count not in PHYSICAL_EXPERT_COUNTS:
        raise ValueError(
            f"{path}: physical_expert_count {physical_count}, "
            f"expected one of {list(PHYSICAL_EXPERT_COUNTS)}"
        )
    if not isinstance(mapping_value, list) or len(mapping_value) != layers:
        raise ValueError(f"{path}: physical_to_source must contain exactly {layers} layer rows")

    rows: list[tuple[int, ...]] = []
    for layer, row in enumerate(mapping_value):
        if not isinstance(row, list) or len(row) != physical_count:
            raise ValueError(
                f"{path}: physical_to_source layer {layer} has "
                f"{len(row) if isinstance(row, list) else 'a non-array'}, expected {physical_count}"
            )
        parsed_row: list[int] = []
        for physical_id, source_id in enumerate(row):
            if isinstance(source_id, bool) or not isinstance(source_id, int):
                raise ValueError(
                    f"{path}: layer {layer} physical expert {physical_id} is not an integer"
                )
            if not 0 <= source_id < source_count:
                raise ValueError(
                    f"{path}: layer {layer} physical expert {physical_id} maps to "
                    f"source expert {source_id}, outside [0, {source_count})"
                )
            parsed_row.append(source_id)
        if len(set(parsed_row)) != len(parsed_row):
            raise ValueError(
                f"{path}: layer {layer} maps multiple physical slots to one source expert"
            )
        rows.append(tuple(parsed_row))

    mapping = tuple(rows)
    return ExpertIdMap(
        physical_count=physical_count,
        layers=layers,
        physical_to_source=mapping,
        source_count=source_count,
        basis="explicit_sidecar",
        sidecar_sha256=sha256(path),
        mapping_sha256=_mapping_digest(mapping),
    )


def resolve_expert_id_map(path: Path | None, physical_count: int) -> ExpertIdMap:
    if path is None:
        return identity_expert_id_map(physical_count)
    return read_expert_id_map(path, physical_count)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_role(name: str) -> str:
    return re.sub(r"^blk\.\d+\.", "blk.N.", name)


def dense_group(name: str) -> str | None:
    role = tensor_role(name)
    for group, suffixes in DENSE_ROLES.items():
        if role.endswith(suffixes):
            return group
    return None


def read_pack_index(path: Path) -> tuple[str, dict[str, tuple[int, int]], int]:
    """Read the v3 dense index without opening dense.bin or any GGUF payload."""
    source = ""
    declared_tensors = None
    shapes: dict[str, tuple[int, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("#"):
                text = line.strip()
                match = re.search(r" from (.+)$", text)
                if match and not source:
                    source = match.group(1)
                match = re.search(r"align (\d+) pool \d+ tensors (\d+)", text)
                if match:
                    declared_tensors = int(match.group(2))
                continue
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 9:
                raise ValueError(f"{path}:{line_number}: expected at least nine index fields")
            name = fields[0]
            if name in shapes:
                raise ValueError(f"{path}:{line_number}: duplicate tensor {name}")
            try:
                ne0, ne1 = int(fields[7]), int(fields[8])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid tensor shape") from exc
            if ne0 < 0 or ne1 < 0:
                raise ValueError(f"{path}:{line_number}: negative shape for {name}")
            shapes[name] = (ne0, ne1)
    if not source:
        raise ValueError(f"{path}: missing source comment")
    lowered = source.lower()
    if any(marker in lowered for marker in TEMPLATE_MODEL_MARKERS):
        raise ValueError(
            f"{source}: the Qwen3.8-27B NVFP4 prefiller is a template, not a runtime dependency"
        )
    if "qwen3.8-flash-next" not in lowered:
        raise ValueError(f"{source}: index does not identify as Qwen3.8-Flash-Next")
    if declared_tensors is None:
        raise ValueError(f"{path}: missing tensor-count header")
    if declared_tensors != len(shapes):
        raise ValueError(f"{path}: header declares {declared_tensors} tensors, found {len(shapes)}")
    return source, shapes, declared_tensors


def read_native_experts(path: Path) -> NativeExpertLayout:
    """Read the v4 native-expert layout header and layer rows."""
    source = ""
    n_expert = 0
    total_bytes = 0
    layers: list[int] = []
    offsets: list[int] = []
    expected_offset = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("#"):
                text = line.strip()
                if text.startswith("# strata native experts"):
                    source = text
                match = re.search(r"n_expert (\d+), total (\d+)", text)
                if match:
                    n_expert, total_bytes = int(match.group(1)), int(match.group(2))
                continue
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 8:
                raise ValueError(
                    f"{path}:{line_number}: expected at least eight native-expert fields"
                )
            try:
                layer = int(fields[0])
                gate_type = int(fields[1])
                down_type = int(fields[2])
                offsets.append(int(fields[3]))
                blob_bytes = int(fields[4])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid native-expert row") from exc
            if gate_type <= 0 or down_type <= 0 or blob_bytes <= 0:
                raise ValueError(f"{path}:{line_number}: non-positive native-expert field")
            if offsets[-1] != expected_offset:
                raise ValueError(
                    f"{path}:{line_number}: offset {offsets[-1]}, expected {expected_offset}"
                )
            expected_offset += blob_bytes * n_expert
            layers.append(layer)
    if not source:
        raise ValueError(f"{path}: missing native experts v4 header")
    if n_expert not in PHYSICAL_EXPERT_COUNTS:
        raise ValueError(
            f"{path}: n_expert {n_expert}, expected one of {list(PHYSICAL_EXPERT_COUNTS)}"
        )
    if tuple(layers) != tuple(range(N_LAYERS)):
        raise ValueError(f"{path}: expected layer rows 0..{N_LAYERS - 1}")
    if expected_offset != total_bytes:
        raise ValueError(
            f"{path}: rows account for {expected_offset} bytes, header says {total_bytes}"
        )
    experts_bin = path.with_name("experts.bin")
    if experts_bin.exists() and experts_bin.stat().st_size != total_bytes:
        raise ValueError(f"{experts_bin}: size does not match native-expert total")
    return NativeExpertLayout(
        source=source, n_expert=n_expert, layers=tuple(layers), total_bytes=total_bytes
    )


def dense_tensors(shapes: dict[str, tuple[int, int]]) -> list[DenseTensor]:
    selected: list[DenseTensor] = []
    for name, (ne0, ne1) in shapes.items():
        group = dense_group(name)
        if group is not None:
            if ne0 <= 0 or ne1 <= 0:
                raise ValueError(f"dense candidate {name} has non-positive shape")
            role = tensor_role(name)
            expected = EXPECTED_DENSE_SHAPES[
                next(suffix for suffix in DENSE_SUFFIXES if role.endswith(suffix))
            ]
            if (ne0, ne1) != expected:
                raise ValueError(f"dense candidate {name} shape {(ne0, ne1)}, expected {expected}")
            selected.append(DenseTensor(name=name, group=group, elements=ne0 * ne1))
    selected.sort(key=lambda item: (item.group, item.name))
    if len(selected) != EXPECTED_DENSE_COUNT:
        raise ValueError(f"selected {len(selected)} dense tensors, expected {EXPECTED_DENSE_COUNT}")
    elements = sum(item.elements for item in selected)
    if elements != EXPECTED_DENSE_ELEMENTS:
        raise ValueError(f"dense candidate elements {elements}, expected {EXPECTED_DENSE_ELEMENTS}")
    shared = [
        shapes[name] for name in shapes if tensor_role(name).endswith(".ffn_gate_shexp.weight")
    ]
    if len(shared) != N_LAYERS or any(set(shape) != {HIDDEN, EXPERT_WIDTH} for shape in shared):
        raise ValueError("shared expert matrices do not establish Flash-Next 2560x640 geometry")
    return selected


def mxfp4_bytes(elements: int) -> int:
    if elements % MXFP4_GROUP:
        raise ValueError(f"MXFP4 group is not byte-aligned for {elements} elements")
    return elements // MXFP4_GROUP * MXFP4_GROUP_BYTES


def select_experts(
    traces: list[Path],
    profile: Path | None,
    keep: int,
    weighted: bool,
    expert_map: ExpertIdMap | None = None,
    trace_domain: str = "physical",
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[list[int]],
    list[list[int]],
    list[list[int]],
    list[list[int]],
    dict[tuple[int, int], Fraction],
    int,
    dict[int, int],
    str | None,
]:
    expert_map = expert_map or identity_expert_id_map(SOURCE_EXPERT_COUNT)
    scores: dict[tuple[int, int], Fraction] = defaultdict(Fraction)
    records = 0
    per_layer: dict[int, int] = {}
    trace_format: str | None = None
    if traces:
        scores, records, per_layer, trace_format = read_trace_scores_exact(
            traces, weighted=weighted, expert_map=expert_map, trace_domain=trace_domain
        )
    profile_rank: list[tuple[int, int]] = []
    if profile is not None:
        profile_rank = [
            (layer, expert_map.physical_to_source[layer][physical_id])
            for layer, physical_id in read_profile_checked(profile, expert_map.physical_count)
        ]
        if len(set(profile_rank)) != len(profile_rank):
            raise ValueError(f"{profile}: duplicate expert in profile ranking")
    profile_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, expert in profile_rank:
        profile_by_layer[layer].append(expert)

    selected: list[list[int]] = []
    profile_fills: list[list[int]] = []
    geometric_fills: list[list[int]] = []
    for layer in range(N_LAYERS):
        rows = [
            (expert, score)
            for (row_layer, expert), score in scores.items()
            if row_layer == layer
            if expert in expert_map.physical_to_source[layer]
        ]
        rows.sort(key=lambda row: (-row[1], row[0]))
        chosen = [expert for expert, _ in rows[:keep]]
        used = set(chosen)
        layer_profile_fills: list[int] = []
        for expert in profile_by_layer.get(layer, []):
            if len(chosen) == keep:
                break
            if expert not in used:
                chosen.append(expert)
                layer_profile_fills.append(expert)
                used.add(expert)
        layer_geometric_fills: list[int] = []
        for expert in sorted(expert_map.physical_to_source[layer]):
            if len(chosen) == keep:
                break
            if expert not in used:
                chosen.append(expert)
                layer_geometric_fills.append(expert)
                used.add(expert)
        if len(chosen) != keep:
            raise AssertionError("internal expert fill error")
        selected.append(chosen)
        profile_fills.append(layer_profile_fills)
        geometric_fills.append(layer_geometric_fills)
    selected_physical = [
        [expert_map.physical_for_source(layer, expert) for expert in experts]
        for layer, experts in enumerate(selected)
    ]
    profile_physical_fills = [
        [expert_map.physical_for_source(layer, expert) for expert in experts]
        for layer, experts in enumerate(profile_fills)
    ]
    geometric_physical_fills = [
        [expert_map.physical_for_source(layer, expert) for expert in experts]
        for layer, experts in enumerate(geometric_fills)
    ]
    return (
        selected,
        selected_physical,
        profile_fills,
        profile_physical_fills,
        geometric_fills,
        geometric_physical_fills,
        dict(scores),
        records,
        per_layer,
        trace_format,
    )


def expert_coverage(
    selected: list[list[int]], scores: dict[tuple[int, int], Fraction]
) -> tuple[Fraction, Fraction] | None:
    total = sum(scores.values(), Fraction(0))
    if not total:
        return None
    selected_total = sum(
        (
            scores.get((layer, expert), Fraction(0))
            for layer, experts in enumerate(selected)
            for expert in experts
        ),
        Fraction(0),
    )
    return selected_total, total


def read_trace_scores_exact(
    paths: list[Path],
    weighted: bool,
    expert_map: ExpertIdMap | None = None,
    trace_domain: str = "physical",
) -> tuple[dict[tuple[int, int], Fraction], int, dict[int, int], str | None]:
    """Read legacy or batched-prefill routing traces into exact source-domain scores.

    Returns the scores, position-count semantics for trace records, per-layer position counts,
    and the common wire format.  A batched STRP chunk expands to one record per token, so its
    counts remain comparable with the legacy one-record-per-position format.
    """
    if trace_domain not in {"physical", "source"}:
        raise ValueError(f"trace domain {trace_domain!r}, expected 'physical' or 'source'")
    expert_map = expert_map or identity_expert_id_map(SOURCE_EXPERT_COUNT)
    scores: dict[tuple[int, int], Fraction] = defaultdict(Fraction)
    records = 0
    per_layer: dict[int, int] = defaultdict(int)

    def source_expert_id(path: Path, layer: int, expert: int) -> int:
        if trace_domain == "physical":
            if not 0 <= expert < expert_map.physical_count:
                raise ValueError(
                    f"{path}: physical-domain expert {expert} outside "
                    f"[0, {expert_map.physical_count})"
                )
            return expert_map.physical_to_source[layer][expert]
        if not 0 <= expert < SOURCE_EXPERT_COUNT:
            raise ValueError(
                f"{path}: source-domain expert {expert} outside [0, {SOURCE_EXPERT_COUNT})"
            )
        return expert

    def add_position(
        path: Path,
        layer: int,
        ids: tuple[int, ...],
        weights: tuple[float, ...],
        position_description: str,
    ) -> None:
        nonlocal records, per_layer
        seen_experts: set[int] = set()
        for expert, weight in zip(ids, weights, strict=True):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"{path}: invalid router weight {weight} at layer {layer} "
                    f"{position_description}"
                )
            source_expert = source_expert_id(path, layer, expert)
            if expert in seen_experts:
                raise ValueError(
                    f"{path}: duplicate trace-domain expert {expert} at layer {layer} "
                    f"{position_description}"
                )
            seen_experts.add(expert)
            exact = Fraction.from_float(weight)
            if weighted and exact == 0:
                raise ValueError(
                    f"{path}: zero router weight at layer {layer} {position_description}; "
                    "use --unweighted"
                )
            scores[(layer, source_expert)] += exact if weighted else Fraction(1)
        records += 1
        per_layer[layer] += 1

    def read_legacy(path: Path, blob: bytes, file_per_layer: dict[int, int]) -> None:
        offset = 0
        observed_layers: set[int] = set()
        while offset < len(blob):
            if offset + 8 > len(blob):
                raise ValueError(f"{path}: truncated routing header at byte {offset}")
            layer, count = struct.unpack_from("<ii", blob, offset)
            offset += 8
            if count < 0 or offset + 8 * count > len(blob):
                raise ValueError(f"{path}: invalid routing count {count} at layer {layer}")
            if count != TRACE_TOP_K:
                raise ValueError(
                    f"{path}: Flash-Next routing record has k={count}, expected {TRACE_TOP_K}"
                )
            ids = struct.unpack_from(f"<{count}i", blob, offset)
            weights = struct.unpack_from(f"<{count}f", blob, offset + 4 * count)
            offset += 8 * count
            if not 0 <= layer < N_LAYERS:
                raise ValueError(f"{path}: layer {layer} outside Flash-Next range")
            observed_layers.add(layer)
            before = records
            add_position(path, layer, ids, weights, "record")
            file_per_layer[layer] += records - before
        if observed_layers != set(range(N_LAYERS)):
            missing = sorted(set(range(N_LAYERS)) - observed_layers)
            raise ValueError(f"{path}: trace does not contain all 48 layers; missing {missing[:8]}")
        if len(set(file_per_layer.values())) != 1:
            raise ValueError(f"{path}: layer record counts differ across the 48 layers")

    def read_batched_strp(path: Path, blob: bytes, file_per_layer: dict[int, int]) -> None:
        if len(blob) < TRACE_HEADER.size:
            raise ValueError(f"{path}: truncated STRP header, expected {TRACE_HEADER.size} bytes")
        (
            magic,
            version,
            phase,
            flags,
            layer_count,
            top_k,
            record_header_bytes,
            reserved,
        ) = TRACE_HEADER.unpack_from(blob)
        if magic != TRACE_MAGIC:
            raise ValueError(f"{path}: STRP magic 0x{magic:08x}, expected 0x{TRACE_MAGIC:08x}")
        if version != TRACE_VERSION:
            raise ValueError(f"{path}: STRP version {version}, expected {TRACE_VERSION}")
        if phase != TRACE_PHASE_PREFILL:
            raise ValueError(
                f"{path}: STRP phase {phase}, expected prefill phase {TRACE_PHASE_PREFILL}"
            )
        if flags != TRACE_FLAGS_NONE:
            raise ValueError(f"{path}: STRP flags {flags}, expected {TRACE_FLAGS_NONE}")
        if layer_count != N_LAYERS:
            raise ValueError(f"{path}: STRP layer_count {layer_count}, expected {N_LAYERS}")
        if top_k != TRACE_TOP_K:
            raise ValueError(f"{path}: STRP top_k {top_k}, expected {TRACE_TOP_K}")
        if record_header_bytes != TRACE_RECORD_HEADER_BYTES:
            raise ValueError(
                f"{path}: STRP record_header_bytes {record_header_bytes}, "
                f"expected {TRACE_RECORD_HEADER_BYTES}"
            )
        if reserved != 0:
            raise ValueError(f"{path}: STRP record reserved field {reserved}, expected 0")

        offset = TRACE_HEADER.size
        observed_layers: set[int] = set()
        intervals: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        request_layers: dict[int, set[int]] = defaultdict(set)
        request_layer_positions: dict[tuple[int, int], set[int]] = defaultdict(set)
        while offset < len(blob):
            record_end = offset + TRACE_RECORD.size
            if record_end > len(blob):
                raise ValueError(f"{path}: truncated STRP record header at byte {offset}")
            layer, count, position, token_count, record_reserved, request_id = (
                TRACE_RECORD.unpack_from(blob, offset)
            )
            offset = record_end
            if not 0 <= layer < N_LAYERS:
                raise ValueError(f"{path}: layer {layer} outside Flash-Next range")
            if count != TRACE_TOP_K:
                raise ValueError(f"{path}: STRP record k={count}, expected {TRACE_TOP_K}")
            if token_count <= 0:
                raise ValueError(f"{path}: STRP token_count {token_count} must be positive")
            if record_reserved != 0:
                raise ValueError(
                    f"{path}: STRP record reserved field {record_reserved}, expected 0"
                )
            if position < 0 or position + token_count > NATIVE_CONTEXT_TOKENS:
                raise ValueError(
                    f"{path}: STRP position range [{position}, {position + token_count}) "
                    f"outside [0, {NATIVE_CONTEXT_TOKENS})"
                )
            values = token_count * count
            payload_end = offset + 8 * values
            if payload_end > len(blob):
                raise ValueError(
                    f"{path}: truncated STRP payload at byte {offset}; expected {8 * values} bytes"
                )
            ids = struct.unpack_from(f"<{values}i", blob, offset)
            weights = struct.unpack_from(f"<{values}f", blob, offset + 4 * values)
            offset = payload_end
            observed_layers.add(layer)
            intervals[(request_id, layer)].append((position, position + token_count))
            request_layers[request_id].add(layer)
            request_layer_positions[(request_id, layer)].update(
                range(position, position + token_count)
            )
            for token in range(token_count):
                start = token * count
                before = records
                add_position(
                    path,
                    layer,
                    ids[start : start + count],
                    weights[start : start + count],
                    f"position {position + token}",
                )
                file_per_layer[layer] += records - before

        if observed_layers != set(range(N_LAYERS)):
            missing = sorted(set(range(N_LAYERS)) - observed_layers)
            raise ValueError(f"{path}: trace does not contain all 48 layers; missing {missing[:8]}")
        for request_id, request_layers_seen in request_layers.items():
            if request_layers_seen != set(range(N_LAYERS)):
                missing = sorted(set(range(N_LAYERS)) - request_layers_seen)
                raise ValueError(
                    f"{path}: request {request_id} does not contain all 48 layers; "
                    f"missing {missing[:8]}"
                )
        for (request_id, layer), layer_intervals in intervals.items():
            layer_intervals.sort()
            for (start, end), following in zip(layer_intervals, layer_intervals[1:], strict=False):
                if end > following[0]:
                    raise ValueError(
                        f"{path}: overlapping STRP positions for request {request_id} "
                        f"at layer {layer}: [{start}, {end}) overlaps "
                        f"[{following[0]}, {following[1]})"
                    )
        if len(set(file_per_layer.values())) != 1:
            raise ValueError(f"{path}: layer record counts differ across the 48 layers")
        for request_id in request_layers:
            reference = request_layer_positions[(request_id, 0)]
            for layer in range(N_LAYERS):
                if request_layer_positions[(request_id, layer)] != reference:
                    raise ValueError(
                        f"{path}: request {request_id} layer {layer} positions differ from layer 0"
                    )

    invocation_format: str | None = None
    for path in paths:
        blob = path.read_bytes()
        file_per_layer: dict[int, int] = defaultdict(int)
        file_format = (
            TRACE_FORMAT_STRP_PREFILL
            if len(blob) >= 4 and struct.unpack_from("<I", blob)[0] == TRACE_MAGIC
            else TRACE_FORMAT_LEGACY
        )
        if invocation_format is None:
            invocation_format = file_format
        elif file_format != invocation_format:
            raise ValueError(
                f"{path}: cannot mix {invocation_format} and {file_format} routing traces"
            )
        if file_format == TRACE_FORMAT_STRP_PREFILL:
            read_batched_strp(path, blob, file_per_layer)
        else:
            read_legacy(path, blob, file_per_layer)
    return dict(scores), records, dict(per_layer), invocation_format


def read_profile_checked(
    path: Path, physical_count: int = SOURCE_EXPERT_COUNT
) -> list[tuple[int, int]]:
    blob = path.read_bytes()
    if len(blob) < 24:
        raise ValueError(f"{path}: truncated expert profile")
    version, layers, experts, slots, ranked = struct.unpack_from("<5I", blob, 4)
    if slots != ranked:
        raise ValueError(f"{path}: profile slot/ranked counts disagree")
    expected_size = 24 + 4 * ranked + 4 * N_LAYERS * physical_count
    if (version, layers, experts) != (1, N_LAYERS, physical_count):
        raise ValueError(f"{path}: profile is not Flash-Next STRP v1")
    if len(blob) != expected_size:
        raise ValueError(f"{path}: profile size {len(blob)}, expected {expected_size}")
    ranked = make_profile.read_profile(path, physical_count)
    invalid_pairs = [
        pair for pair in ranked if not 0 <= pair[0] < N_LAYERS or not 0 <= pair[1] < physical_count
    ]
    if invalid_pairs:
        raise ValueError(f"{path}: out-of-range expert pair {invalid_pairs[0]}")
    return ranked


def build_manifest(
    index: Path,
    native: Path,
    traces: list[Path],
    profile: Path | None,
    keep: int,
    weighted: bool,
    expert_id_map: Path | None = None,
    trace_domain: str = "physical",
) -> dict[str, Any]:
    source, shapes, tensor_count = read_pack_index(index)
    layout = read_native_experts(native)
    expert_map = resolve_expert_id_map(expert_id_map, layout.n_expert)
    dense = dense_tensors(shapes)
    (
        selected,
        selected_physical,
        profile_fills,
        profile_physical_fills,
        geometric_fills,
        geometric_physical_fills,
        scores,
        records,
        records_per_layer,
        trace_format,
    ) = select_experts(
        traces,
        profile,
        keep,
        weighted,
        expert_map=expert_map,
        trace_domain=trace_domain,
    )
    if not traces and profile is None:
        raise ValueError("selecting experts requires either --trace routing traces or --profile")

    dense_elements = sum(item.elements for item in dense)
    physical_expert_elements = N_LAYERS * layout.n_expert * 3 * HIDDEN * EXPERT_WIDTH
    selected_expert_elements = N_LAYERS * keep * 3 * HIDDEN * EXPERT_WIDTH
    if keep == layout.n_expert and selected_expert_elements != physical_expert_elements:
        raise AssertionError("full-expert element accounting mismatch")
    full_bytes = mxfp4_bytes(FULL_EXPERT_ELEMENTS)
    dense_bytes = mxfp4_bytes(dense_elements)
    expert_bytes = mxfp4_bytes(selected_expert_elements)
    physical_pack_bytes = mxfp4_bytes(physical_expert_elements)
    coverage = expert_coverage(selected, scores)
    coverage_value = float(coverage[0] / coverage[1]) if coverage is not None else None
    # The coverage MEANS different things depending on the trace: a one-shot trace carries
    # the router's own weights, while a serve-mode trace carries unit activations (the
    # serve dispatcher does not surface the weights).  The manifest names the field after
    # the measure, so an activation share can never be read as a router-weight share.
    coverage_field: str | None
    if traces and weighted:
        coverage_field = "aggregate_router_weight_coverage"
    elif traces:
        coverage_field = "aggregate_routed_activation_coverage"
    else:
        coverage_field = None

    return {
        "schema": "strata-prefiller-plan-v2",
        "target": {
            "model": "Qwen3.8-Flash-Next",
            "architecture": "qwen4exp",
            "role": "phase-specialized auxiliary prompt-state prefiller",
            "decoder": "frozen deployed low-bit Flash-Next checkpoint",
            "template_only": [
                "ISTA-DASLab/Qwen3.8-27B-NVFP4-prefiller",
                "arXiv:2609.26333",
            ],
            "runtime_dependencies": [],
        },
        "decoder_binding": {
            "pack_index_name": index.name,
            "pack_index_sha256": sha256(index),
            "native_experts_name": native.name,
            "native_experts_sha256": sha256(native),
            "index_source": source,
            "native_expert_header": layout.source,
            "index_tensors": tensor_count,
            "layers": N_LAYERS,
            "experts_per_layer": expert_map.source_count,
            "physical_experts_per_layer": layout.n_expert,
            "source_experts_per_layer": expert_map.source_count,
            "expert_id_map_name": expert_id_map.name if expert_id_map else None,
            "expert_id_map_sha256": expert_map.sidecar_sha256,
            "expert_id_mapping_sha256": expert_map.mapping_sha256,
            "expert_id_map_basis": expert_map.basis,
        },
        "dense_tensors": {
            "groups": {group: sorted(suffixes) for group, suffixes in DENSE_ROLES.items()},
            "count": len(dense),
            "elements": dense_elements,
            "names": [
                {"name": item.name, "group": item.group, "elements": item.elements}
                for item in dense
            ],
        },
        "expert_selection": {
            "basis": "routing_trace_weighted"
            if traces and weighted
            else "routing_trace_activation"
            if traces
            else "profile_rank",
            "keep_per_layer": keep,
            "physical_expert_count": expert_map.physical_count,
            "source_expert_count": expert_map.source_count,
            "trace_domain": trace_domain if traces else None,
            "trace_format": trace_format if traces else None,
            "trace_names": [path.name for path in traces],
            "trace_sha256": [sha256(path) for path in traces],
            "trace_records": records,
            "profile_name": profile.name if profile else None,
            "profile_sha256": sha256(profile) if profile else None,
            "coverage_measure": coverage_field,
            "coverage_value": coverage_value,
            "coverage_exact": None
            if coverage is None
            else {
                "selected_numerator": coverage[0].numerator,
                "selected_denominator": coverage[0].denominator,
                "total_numerator": coverage[1].numerator,
                "total_denominator": coverage[1].denominator,
            },
            "coverage_note": (
                "Aggregate router-weight share of the traced routing; not a per-request "
                "cache-hit rate and not a quality measure."
                if coverage is not None and weighted
                else "Aggregate share of routed activations (unit weights, as a serve-mode "
                "trace records); not a router-weight share, not a cache-hit rate, and not a "
                "quality measure."
                if coverage is not None
                else "Profile ranking carries no frequency score; coverage is intentionally null."
            ),
            "trace_records_per_layer": records_per_layer,
            "selected_ids": selected,
            "selected_physical_ids": selected_physical,
            "profile_filled_ids": profile_fills,
            "profile_filled_physical_ids": profile_physical_fills,
            "geometric_filled_ids": geometric_fills,
            "geometric_filled_physical_ids": geometric_physical_fills,
        },
        "mxfp4_estimate": {
            "bits_per_element": 4.25,
            "group_elements": MXFP4_GROUP,
            "group_bytes": MXFP4_GROUP_BYTES,
            "dense_bytes": dense_bytes,
            "dense_gib": round(dense_bytes / 2**30, 6),
            "selected_expert_bytes": expert_bytes,
            "selected_expert_gib": round(expert_bytes / 2**30, 6),
            "physical_pack_expert_bytes": physical_pack_bytes,
            "physical_pack_expert_gib": round(physical_pack_bytes / 2**30, 6),
            "total_bytes": dense_bytes + expert_bytes,
            "total_gib": round((dense_bytes + expert_bytes) / 2**30, 6),
            "full_expert_bytes": full_bytes,
            "full_expert_gib": round(full_bytes / 2**30, 6),
            "source_universe_expert_bytes": full_bytes,
            "source_universe_expert_gib": round(full_bytes / 2**30, 6),
        },
        "training_policy": {
            "train_phase": "prompt positions only",
            "decode_phase": "frozen low-bit decoder weights",
            "initial_train_groups": list(DENSE_ROLES),
            "initial_hot_experts": keep,
            "share_initially": [
                "token embedding",
                "output head",
                "norms and scalar parameters",
                "routers",
                "PLE key/value and n-gram table",
            ],
            "objective": (
                "phase-masked QADD; response loss reaches prompt weights through GDN/QSA/PLE state"
            ),
        },
        "non_claims": [
            "Metadata-only plan; no model, trainer, accelerator, or benchmark ran.",
            "Profile-rank selection has no frequency coverage; the profile stores order only.",
            "MXFP4 is a size policy here; sm_86 execution is W4A16 dequantization, "
            "not native FP4 speed.",
            "MXFP4 sizes are logical payload estimates; headers, alignment, and padding "
            "are excluded.",
            "This plan does not import or depend on the dense 27B checkpoint.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", required=True, type=Path, help="deployed pack's index.txt")
    parser.add_argument(
        "--native-experts", required=True, type=Path, help="deployed pack's native_experts.txt"
    )
    parser.add_argument(
        "--expert-id-map",
        type=Path,
        help=f"required {EXPERT_ID_MAP_SCHEMA} JSON sidecar for 256/288/384-slot packs",
    )
    parser.add_argument("traces", nargs="*", type=Path, help="engine --dump-routing files")
    parser.add_argument(
        "--trace-domain",
        choices=("physical", "source"),
        default="physical",
        help="ID domain used by routing traces (native packs emit physical IDs)",
    )
    parser.add_argument(
        "--profile", type=Path, help="expert-profile.bin fallback or deterministic fill order"
    )
    parser.add_argument(
        "--hot-experts", type=int, default=64, help="routed experts per layer (default 64)"
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="rank traces by activation count, not router weight",
    )
    parser.add_argument("--out", required=True, type=Path, help="JSON manifest output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    physical_count = read_native_experts(args.native_experts).n_expert
    if physical_count not in PHYSICAL_EXPERT_COUNTS:
        raise SystemExit(
            f"--native-experts declares {physical_count} physical experts; expected one of "
            f"{list(PHYSICAL_EXPERT_COUNTS)}"
        )
    if not 0 <= args.hot_experts <= physical_count:
        raise SystemExit(f"--hot-experts must be in [0, {physical_count}]")
    if args.expert_id_map is None and physical_count != SOURCE_EXPERT_COUNT:
        raise SystemExit(f"--expert-id-map is required for {physical_count} physical expert slots")
    if not args.traces and args.profile is None:
        raise SystemExit("provide routing traces or --profile")
    inputs = {
        path.resolve()
        for path in (
            args.index,
            args.native_experts,
            args.expert_id_map,
            args.profile,
            *args.traces,
        )
        if path
    }
    if args.out.resolve() in inputs:
        raise SystemExit("--out must not overwrite an input")
    manifest = build_manifest(
        args.index,
        args.native_experts,
        args.traces,
        args.profile,
        args.hot_experts,
        weighted=not args.unweighted,
        expert_id_map=args.expert_id_map,
        trace_domain=args.trace_domain,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    estimate = manifest["mxfp4_estimate"]
    selection = manifest["expert_selection"]
    coverage = selection["coverage_value"]
    coverage_text = "null" if coverage is None else f"{coverage:.4f}"
    coverage_measure = selection["coverage_measure"] or "coverage"
    print(
        f"wrote {args.out}: {args.hot_experts} experts/layer, "
        f"dense {estimate['dense_gib']:.3f} GiB, selected {estimate['total_gib']:.3f} GiB, "
        f"{coverage_measure} {coverage_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
