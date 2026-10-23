#!/usr/bin/env python3
"""Plan and validate a Strata-owned Flash-Next Coder artifact.

This is a deterministic planner, not a model executor.  It validates an expert
mask, compares it with the published Coder allocation, calculates exact expert
footprints for Strata-native recipes, optionally emits RCO prune masks, and
writes the downstream prune/convert/iMatrix/quantize command manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, TypedDict

ROOT = Path(__file__).resolve().parents[1]
N_LAYERS = 48
N_EXPERTS = 512
EXPERT_PARAMETERS = 2 * 640 * 2560 + 640 * 2560
ALLOWED_KEEP = (256, 288, 384, 512)

RECIPES: dict[str, dict[str, Any]] = {
    "parent-mix": {
        "gate_up": "mixed",
        "down": "mixed",
        "bpw": 3.3307,
        "description": "published Coder allocation exact retained-expert width",
    },
    "iq3s-iq4nl": {
        "gate_up": "IQ3_S",
        "down": "IQ4_NL",
        "bpw": 3.7916666666666665,
        "description": "Strata-native gate/up and executable 4.5-bpw down path",
    },
    "iq4xs-iq4nl": {
        "gate_up": "IQ4_XS",
        "down": "IQ4_NL",
        "bpw": 4.333333333333333,
        "description": "higher-width lane; still executable by the existing native kernels",
    },
}


@dataclass(frozen=True)
class Comparison:
    own_kept: int
    parent_kept: int
    intersection: int
    union: int
    jaccard: float
    parent_capture: float
    own_capture: float
    min_layer_intersection: int
    max_layer_intersection: int


class CommandSpec(TypedDict):
    """A recorded command: the exact argv plus a shell-safe rendering."""

    argv: list[str]
    shell: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_own_mask(path: Path) -> tuple[int, list[set[int]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != "strata-reap-mask-v1":
        raise ValueError(f"{path}: expected format strata-reap-mask-v1")
    if value.get("n_expert") != N_EXPERTS:
        raise ValueError(f"{path}: n_expert must be {N_EXPERTS}")
    rows = value.get("keep_ids")
    if not isinstance(rows, list) or len(rows) != N_LAYERS:
        raise ValueError(f"{path}: keep_ids must contain {N_LAYERS} layers")
    keep = len(rows[0]) if isinstance(rows[0], list) and rows[0] else 0
    if keep not in ALLOWED_KEEP:
        raise ValueError(f"{path}: keep count {keep} is not one of {ALLOWED_KEEP}")
    result: list[set[int]] = []
    for layer, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != keep:
            raise ValueError(f"{path}: layer {layer} does not keep exactly {keep} experts")
        if any(not isinstance(item, int) or not 0 <= item < N_EXPERTS for item in row):
            raise ValueError(f"{path}: layer {layer} has an invalid expert id")
        if len(set(row)) != len(row):
            raise ValueError(f"{path}: layer {layer} has duplicate experts")
        result.append(set(row))
    return keep, result


def parse_parent_allocation(path: Path) -> list[set[int]]:
    pattern = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\[(\d+)\]:\s*(\S*)\s*$")
    observed: dict[tuple[int, int, str], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        layer, kind, expert, quant = int(match[1]), match[2], int(match[3]), match[4]
        # An empty quantization type marks a pruned tensor, not a retained one.
        if quant:
            observed[(layer, expert, kind)] = quant
    result: list[set[int]] = []
    for layer in range(N_LAYERS):
        kept: set[int] = set()
        for expert in range(N_EXPERTS):
            kinds = tuple(
                observed.get((layer, expert, kind), "") for kind in ("gate", "up", "down")
            )
            if any(kinds) and not all(kinds):
                raise ValueError(
                    f"{path}:{layer}:{expert}: gate/up/down retention disagreement {kinds}"
                )
            if all(kinds):
                kept.add(expert)
        result.append(kept)
    if not all(len(row) == 256 for row in result):
        sizes = sorted({len(row) for row in result})
        raise ValueError(f"{path}: parent allocation is not uniform 256/layer: {sizes}")
    return result


def compare_masks(own: list[set[int]], parent: list[set[int]]) -> Comparison:
    intersections = [len(left & right) for left, right in zip(own, parent, strict=True)]
    intersection = sum(intersections)
    union = sum(len(left | right) for left, right in zip(own, parent, strict=True))
    own_total = sum(len(row) for row in own)
    parent_total = sum(len(row) for row in parent)
    return Comparison(
        own_kept=own_total,
        parent_kept=parent_total,
        intersection=intersection,
        union=union,
        jaccard=intersection / union,
        parent_capture=intersection / parent_total,
        own_capture=intersection / own_total,
        min_layer_intersection=min(intersections),
        max_layer_intersection=max(intersections),
    )


def budget(keep: int, recipe_name: str) -> dict[str, Any]:
    recipe = RECIPES[recipe_name]
    bits = Fraction(N_LAYERS * keep * EXPERT_PARAMETERS) * Fraction(str(recipe["bpw"]))
    parent_bits = Fraction(N_LAYERS * 256 * EXPERT_PARAMETERS) * Fraction(
        str(RECIPES["parent-mix"]["bpw"])
    )
    expert_bytes = bits / 8
    return {
        "recipe": recipe_name,
        "gate_up": recipe["gate_up"],
        "down": recipe["down"],
        "bpw": recipe["bpw"],
        "kept_experts": N_LAYERS * keep,
        "removed_experts": N_LAYERS * (N_EXPERTS - keep),
        "expert_bytes": round(expert_bytes),
        "expert_gb_decimal": float(expert_bytes / 1_000_000_000),
        "expert_gib": float(expert_bytes / 2**30),
        "relative_to_parent_expert_bytes": float(bits / parent_bits),
    }


def write_rco_prune_mask(path: Path, own: list[set[int]]) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    rows = []
    for kept in own:
        rows.append([expert not in kept for expert in range(N_EXPERTS)])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(rows, dtype=torch.bool), path)


def command(command_arguments: list[str]) -> CommandSpec:
    return {"argv": command_arguments, "shell": shlex.join(command_arguments)}


def quantize_arguments(
    quantizer: Path,
    imatrix_path: Path,
    input_gguf: Path,
    output_gguf: Path,
    tensor_type_file: Path,
    gate_up_type: str,
    threads: int,
) -> list[str]:
    """Build one `llama-quantize` argv.

    The own-mask arm and the standard-calibration control arm differ only in
    the imatrix file and the output file. A complete per-tensor recipe is
    mandatory: llama.cpp ignores manual overrides when the global type is COPY
    because COPY is not a quantized default. The quantized default plus recipe
    restores every non-target tensor to its source type.
    """
    return [
        str(quantizer),
        "--imatrix",
        str(imatrix_path),
        "--tensor-type-file",
        str(tensor_type_file),
        "--keep-split",
        str(input_gguf),
        str(output_gguf),
        gate_up_type,
        str(threads),
    ]


def validate_quantize_layout(
    argv: list[str],
    imatrix_path: Path,
    input_gguf: Path,
    output_gguf: Path,
    tensor_type_file: Path,
    default_type: str,
    threads: int,
) -> None:
    """Fail loudly when a `llama-quantize` argv no longer has the expected layout.

    The manifest records these argv lists verbatim, so a layout drift that
    swapped an input for an output would still look plausible in the receipt.
    A complete recipe and quantized default are load-bearing: llama.cpp only
    applies manual tensor overrides when the default type is quantized.
    """
    if argv[-2:] != [default_type, str(threads)]:
        raise ValueError(
            f"llama-quantize argv must end with {default_type} and the thread count: {argv}"
        )
    if argv.count("--imatrix") != 1:
        raise ValueError(f"llama-quantize argv must carry exactly one --imatrix flag: {argv}")
    imatrix_index = argv.index("--imatrix")
    if imatrix_index + 1 >= len(argv) or argv[imatrix_index + 1] != str(imatrix_path):
        raise ValueError(f"llama-quantize argv must name the requested imatrix: {argv}")
    if argv.count("--tensor-type") != 0:
        raise ValueError(f"llama-quantize argv must not use scattered --tensor-type flags: {argv}")
    if argv.count("--tensor-type-file") != 1:
        raise ValueError(f"llama-quantize argv must carry one --tensor-type-file flag: {argv}")
    recipe_index = argv.index("--tensor-type-file") + 1
    if recipe_index >= len(argv) or argv[recipe_index] != str(tensor_type_file):
        raise ValueError(f"llama-quantize argv must name the complete recipe: {tensor_type_file}")
    if "--keep-split" not in argv:
        raise ValueError("llama-quantize argv must preserve the source split layout")
    if str(input_gguf) not in argv:
        raise ValueError(f"llama-quantize argv must name the requested input file: {argv}")
    if str(output_gguf) not in argv:
        raise ValueError(f"llama-quantize argv must name the requested output file: {argv}")


def rco_mask_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    own = args.rco_mask_out
    if own is None:
        own = args.work_root / "own-prune-mask.pt"
    parent = args.parent_rco_mask_out
    if parent is None:
        raise ValueError("--parent-rco-mask-out is required for the parent control arm")
    if own.resolve() == parent.resolve():
        raise ValueError("own and parent RCO mask destinations must be different files")
    destinations = (("own mask", own), ("parent mask", parent))
    sources = (("own mask input", args.mask), ("parent allocation input", args.parent_allocation))
    for destination_label, destination in destinations:
        for source_label, source in sources:
            if destination.resolve() == source.resolve():
                raise ValueError(
                    f"{destination_label} destination must not overwrite {source_label}: {source}"
                )
    return own, parent


def build_manifest(
    args: argparse.Namespace,
    keep: int,
    *,
    own: list[set[int]] | None = None,
    parent: list[set[int]] | None = None,
) -> dict[str, Any]:
    """Build the manifest for the requested recipe.

    `own` and `parent` are the already-parsed masks.  When they are omitted the
    files named by `args.mask` and `args.parent_allocation` are parsed here, so
    direct callers keep working.  Supplied rows must have been parsed from
    those exact paths, because the recorded digests are computed over them.
    """
    if own is None:
        _, own = load_own_mask(args.mask)
    if parent is None:
        parent = parse_parent_allocation(args.parent_allocation)
    if len(own) != N_LAYERS or len(parent) != N_LAYERS:
        raise ValueError(f"parsed masks must hold {N_LAYERS} layers each")
    if any(len(row) != keep for row in own):
        raise ValueError(f"parsed own mask does not keep {keep} experts per layer")
    comparison = compare_masks(own, parent)
    work = args.work_root
    own_prune_mask, parent_prune_mask = rco_mask_paths(args)
    pruned_model = work / f"pruned-bf16-{keep}"
    bf16_gguf = work / f"flash-next-own-{keep}-bf16.gguf"
    imatrix = work / f"own-{keep}-imatrix.gguf"
    control_imatrix = work / "standard-ptq-imatrix.gguf"
    tensor_type_recipe = work / f"own-{keep}-{args.recipe}-tensor-types.txt"
    quantized = work / f"flash-next-own-{keep}-{args.recipe}.gguf"
    control_quantized = work / f"flash-next-own-{keep}-{args.recipe}-standard-cal.gguf"
    rco = args.rco_root
    python = args.python
    recipe = RECIPES[args.recipe]

    gate_up_type = "IQ3_S" if recipe["gate_up"] == "IQ3_S" else "IQ4_XS"
    quantize_argv = quantize_arguments(
        args.llama_quantize,
        imatrix,
        bf16_gguf,
        quantized,
        tensor_type_recipe,
        gate_up_type,
        args.threads,
    )
    control_quantize_argv = quantize_arguments(
        args.llama_quantize,
        control_imatrix,
        bf16_gguf,
        control_quantized,
        tensor_type_recipe,
        gate_up_type,
        args.threads,
    )
    validate_quantize_layout(
        quantize_argv,
        imatrix,
        bf16_gguf,
        quantized,
        tensor_type_recipe,
        gate_up_type,
        args.threads,
    )
    validate_quantize_layout(
        control_quantize_argv,
        control_imatrix,
        bf16_gguf,
        control_quantized,
        tensor_type_recipe,
        gate_up_type,
        args.threads,
    )
    return {
        "format": "strata-custom-coder-plan-v1",
        "mask": str(args.mask),
        "mask_sha256": sha256(args.mask),
        "parent_allocation": str(args.parent_allocation),
        "parent_allocation_sha256": sha256(args.parent_allocation),
        "keep_per_layer": keep,
        "physical_constraint": (
            "the Strata expert sidecar admits uniform 256, 288, 384, or 512 slots"
        ),
        "mask_comparison": comparison.__dict__,
        "budgets": [budget(item, name) for name in RECIPES for item in (keep,)],
        "selected_budget": budget(keep, args.recipe),
        "commands": {
            "search_own_mask": command(
                [
                    python,
                    str(rco / "rco_search_prune.py"),
                    "--model",
                    str(args.model),
                    "--sparsity",
                    str((N_EXPERTS - keep) / N_EXPERTS),
                    "--per-layer-budget",
                    "--calibration-data",
                    str(args.calibration_tokens),
                    "--batch-size",
                    str(args.batch_size),
                    "--save-mask",
                    str(own_prune_mask),
                ]
            ),
            "eval_own_mask": command(
                [
                    python,
                    str(rco / "rco_search_prune.py"),
                    "--model",
                    str(args.model),
                    "--eval-mask",
                    str(own_prune_mask),
                    "--calibration-data",
                    str(args.evaluation_tokens),
                    "--batch-size",
                    str(args.batch_size),
                ]
            ),
            "eval_parent_mask": command(
                [
                    python,
                    str(rco / "rco_search_prune.py"),
                    "--model",
                    str(args.model),
                    "--eval-mask",
                    str(parent_prune_mask),
                    "--calibration-data",
                    str(args.evaluation_tokens),
                    "--batch-size",
                    str(args.batch_size),
                ]
            ),
            "materialize_pruned_bf16": command(
                [
                    python,
                    str(rco / "run_build_checkpoint.py"),
                    "prune",
                    "--model",
                    str(args.model),
                    "--prune-mask",
                    str(own_prune_mask),
                    "--output",
                    str(pruned_model),
                    "--prune-mode",
                    "remove",
                ]
            ),
            "convert_bf16_gguf": command(
                [
                    python,
                    str(args.llama_convert),
                    str(pruned_model),
                    "--outfile",
                    str(bf16_gguf),
                    "--outtype",
                    "bf16",
                    "--split-max-size",
                    args.split_max_size,
                ]
            ),
            "build_own_imatrix": command(
                [
                    str(args.llama_imatrix),
                    "--token-file",
                    str(args.calibration_raw_tokens),
                    "-m",
                    str(bf16_gguf),
                    "-o",
                    str(imatrix),
                    "-c",
                    str(args.sequence_length),
                    "--chunks",
                    str(args.calibration_chunks),
                    "--no-ppl",
                    "--output-frequency",
                    "10",
                    "--save-frequency",
                    "50",
                ]
            ),
            "build_standard_control_imatrix": command(
                [
                    str(args.llama_imatrix),
                    "--token-file",
                    str(args.control_raw_tokens),
                    "-m",
                    str(bf16_gguf),
                    "-o",
                    str(control_imatrix),
                    "-c",
                    str(args.sequence_length),
                    "--chunks",
                    str(args.calibration_chunks),
                    "--no-ppl",
                    "--output-frequency",
                    "10",
                    "--save-frequency",
                    "50",
                ]
            ),
            "build_quantization_recipe": command(
                [
                    python,
                    str(ROOT / "tools" / "exact_copy_recipe.py"),
                    "--model",
                    str(bf16_gguf),
                    "--output",
                    str(tensor_type_recipe),
                    "--gate-up-type",
                    gate_up_type,
                    "--down-type",
                    "IQ4_NL",
                    "--ple-type",
                    "IQ4_NL",
                ]
            ),
            "quantize_experts": command(quantize_argv),
            "quantize_standard_calibration_control": command(control_quantize_argv),
        },
        "acceptance": {
            "heldout_rco_kl": (
                "own mask must beat the parent mask on the untouched internal-code split"
            ),
            "calibration_control": (
                "the own-data and pinned-standard iMatrix artifacts are compared separately"
            ),
            "heal_gate": ("heal only if unhealed own-mask held-out KL loses to the parent mask"),
            "pack_inversion": (
                "a production verifier must compare every native experts.bin slice "
                "to its named GGUF shard byte-for-byte"
            ),
            "runtime_gate": (
                "same-day vendor Coder and full-model controls precede throughput claims"
            ),
        },
        "non_claims": [
            "Footprints count routed-expert bytes only; dense weights, metadata, "
            "the PLE table, and serving arenas are not included.",
            "No RCO search, model forward, heal, conversion, iMatrix, quantization, "
            "pack, or runtime ran while writing this manifest.",
            "More retained experts and higher width remain hypotheses until held-out "
            "quality and same-day runtime gates pass.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--parent-allocation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rco-mask-out", type=Path)
    parser.add_argument(
        "--parent-rco-mask-out",
        type=Path,
        required=True,
        help="destination for the parent mask used by the required control arm",
    )
    parser.add_argument(
        "--model", type=Path, default=Path.home() / "models/Qwen3.8-Flash-Next-BF16"
    )
    parser.add_argument(
        "--rco-root", type=Path, default=Path.home() / ".local/share/strata-tools/RCO"
    )
    parser.add_argument("--llama-convert", required=True, type=Path)
    parser.add_argument("--llama-imatrix", required=True, type=Path)
    parser.add_argument("--llama-quantize", required=True, type=Path)
    parser.add_argument("--calibration-raw-tokens", required=True, type=Path)
    parser.add_argument(
        "--control-raw-tokens",
        required=True,
        type=Path,
        help="pinned standard PTQ token stream for the calibration control arm",
    )
    parser.add_argument("--calibration-tokens", required=True, type=Path)
    parser.add_argument("--evaluation-tokens", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--recipe", choices=("iq3s-iq4nl", "iq4xs-iq4nl"), default="iq3s-iq4nl")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--calibration-chunks", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--split-max-size", default="40G")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keep, own = load_own_mask(args.mask)
    own_prune_mask, parent_prune_mask = rco_mask_paths(args)
    parent = parse_parent_allocation(args.parent_allocation)
    write_rco_prune_mask(own_prune_mask, own)
    write_rco_prune_mask(parent_prune_mask, parent)
    manifest = build_manifest(args, keep, own=own, parent=parent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.out.write_text(payload, encoding="utf-8")
    selected = manifest["selected_budget"]
    print(
        f"wrote {args.out}: keep={keep}/layer, {selected['expert_gb_decimal']:.3f} GB "
        f"experts at {selected['bpw']:.4f} bpw"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
