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
    if n_expert != N_EXPERT:
        raise ValueError(f"{path}: n_expert {n_expert}, expected {N_EXPERT} for full Flash-Next")
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
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[list[int]],
    dict[tuple[int, int], Fraction],
    int,
]:
    scores: dict[tuple[int, int], Fraction] = defaultdict(Fraction)
    records = 0
    if traces:
        scores, records = read_trace_scores_exact(traces, weighted=weighted)
    profile_rank: list[tuple[int, int]] = []
    if profile is not None:
        profile_rank = read_profile_checked(profile)
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
            (expert, score) for (row_layer, expert), score in scores.items() if row_layer == layer
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
        for expert in range(N_EXPERT):
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
    return selected, profile_fills, geometric_fills, dict(scores), records


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
    paths: list[Path], weighted: bool
) -> tuple[dict[tuple[int, int], Fraction], int]:
    """Read routing traces with exact binary-rational router-weight sums."""
    scores: dict[tuple[int, int], Fraction] = defaultdict(Fraction)
    records = 0
    for path in paths:
        blob = path.read_bytes()
        offset = 0
        observed_layers: set[int] = set()
        while offset < len(blob):
            if offset + 8 > len(blob):
                raise ValueError(f"{path}: truncated routing header at byte {offset}")
            layer, count = struct.unpack_from("<ii", blob, offset)
            offset += 8
            if count < 0 or offset + 8 * count > len(blob):
                raise ValueError(f"{path}: invalid routing count {count} at layer {layer}")
            if count != 10:
                raise ValueError(f"{path}: Flash-Next routing record has k={count}, expected 10")
            if count:
                ids = struct.unpack_from(f"<{count}i", blob, offset)
                weights = struct.unpack_from(f"<{count}f", blob, offset + 4 * count)
            else:
                ids, weights = (), ()
            offset += 8 * count
            if not 0 <= layer < N_LAYERS:
                raise ValueError(f"{path}: layer {layer} outside Flash-Next range")
            observed_layers.add(layer)
            records += 1
            seen_experts: set[int] = set()
            for expert, weight in zip(ids, weights, strict=True):
                if not math.isfinite(weight) or weight < 0.0:
                    raise ValueError(f"{path}: invalid router weight {weight} at layer {layer}")
                if not 0 <= expert < N_EXPERT:
                    raise ValueError(f"{path}: expert {expert} outside Flash-Next range")
                if expert in seen_experts:
                    raise ValueError(f"{path}: duplicate expert {expert} at layer {layer}")
                seen_experts.add(expert)
                exact = Fraction.from_float(weight)
                if weighted and exact == 0:
                    raise ValueError(
                        f"{path}: zero router weight at layer {layer}; use --unweighted"
                    )
                scores[(layer, expert)] += exact if weighted else Fraction(1)
        if observed_layers != set(range(N_LAYERS)):
            missing = sorted(set(range(N_LAYERS)) - observed_layers)
            raise ValueError(f"{path}: trace does not contain all 48 layers; missing {missing[:8]}")
    return dict(scores), records


def read_profile_checked(path: Path) -> list[tuple[int, int]]:
    blob = path.read_bytes()
    if len(blob) < 24:
        raise ValueError(f"{path}: truncated expert profile")
    version, layers, experts, slots, ranked = struct.unpack_from("<5I", blob, 4)
    if slots != ranked:
        raise ValueError(f"{path}: profile slot/ranked counts disagree")
    expected_size = 24 + 4 * ranked + 4 * N_LAYERS * N_EXPERT
    if (version, layers, experts) != (1, N_LAYERS, N_EXPERT):
        raise ValueError(f"{path}: profile is not Flash-Next STRP v1")
    if len(blob) != expected_size:
        raise ValueError(f"{path}: profile size {len(blob)}, expected {expected_size}")
    ranked = make_profile.read_profile(path, N_EXPERT)
    invalid_pairs = [
        pair for pair in ranked if not 0 <= pair[0] < N_LAYERS or not 0 <= pair[1] < N_EXPERT
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
) -> dict[str, Any]:
    source, shapes, tensor_count = read_pack_index(index)
    layout = read_native_experts(native)
    dense = dense_tensors(shapes)
    selected, profile_fills, geometric_fills, scores, records = select_experts(
        traces, profile, keep, weighted
    )
    if not traces and profile is None:
        raise ValueError("selecting experts requires either --trace routing traces or --profile")

    dense_elements = sum(item.elements for item in dense)
    selected_expert_elements = N_LAYERS * keep * 3 * HIDDEN * EXPERT_WIDTH
    if keep == N_EXPERT and selected_expert_elements != FULL_EXPERT_ELEMENTS:
        raise AssertionError("full-expert element accounting mismatch")
    full_bytes = mxfp4_bytes(FULL_EXPERT_ELEMENTS)
    dense_bytes = mxfp4_bytes(dense_elements)
    expert_bytes = mxfp4_bytes(selected_expert_elements)
    coverage = expert_coverage(selected, scores)
    coverage_value = float(coverage[0] / coverage[1]) if coverage is not None else None

    return {
        "schema": "strata-prefiller-plan-v1",
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
            "experts_per_layer": N_EXPERT,
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
            "trace_names": [path.name for path in traces],
            "trace_sha256": [sha256(path) for path in traces],
            "trace_records": records,
            "profile_name": profile.name if profile else None,
            "profile_sha256": sha256(profile) if profile else None,
            "aggregate_positive_weight_coverage": coverage_value,
            "coverage_exact": None
            if coverage is None
            else {
                "selected_numerator": coverage[0].numerator,
                "selected_denominator": coverage[0].denominator,
                "total_numerator": coverage[1].numerator,
                "total_denominator": coverage[1].denominator,
            },
            "coverage_note": (
                "Aggregate router-weight share; not per-request cache-hit rate or quality."
                if coverage is not None
                else "Profile ranking carries no frequency score; coverage is intentionally null."
            ),
            "selected_ids": selected,
            "profile_filled_ids": profile_fills,
            "geometric_filled_ids": geometric_fills,
        },
        "mxfp4_estimate": {
            "bits_per_element": 4.25,
            "group_elements": MXFP4_GROUP,
            "group_bytes": MXFP4_GROUP_BYTES,
            "dense_bytes": dense_bytes,
            "dense_gib": round(dense_bytes / 2**30, 6),
            "selected_expert_bytes": expert_bytes,
            "selected_expert_gib": round(expert_bytes / 2**30, 6),
            "total_bytes": dense_bytes + expert_bytes,
            "total_gib": round((dense_bytes + expert_bytes) / 2**30, 6),
            "full_expert_bytes": full_bytes,
            "full_expert_gib": round(full_bytes / 2**30, 6),
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
    parser.add_argument("traces", nargs="*", type=Path, help="engine --dump-routing files")
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
    if not 0 <= args.hot_experts <= N_EXPERT:
        raise SystemExit(f"--hot-experts must be in [0, {N_EXPERT}]")
    if not args.traces and args.profile is None:
        raise SystemExit("provide routing traces or --profile")
    inputs = {
        path.resolve()
        for path in (args.index, args.native_experts, args.profile, *args.traces)
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
    coverage = manifest["expert_selection"]["aggregate_positive_weight_coverage"]
    coverage_text = "null" if coverage is None else f"{coverage:.4f}"
    print(
        f"wrote {args.out}: {args.hot_experts} experts/layer, "
        f"dense {estimate['dense_gib']:.3f} GiB, selected {estimate['total_gib']:.3f} GiB, "
        f"coverage {coverage_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
