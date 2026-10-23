#!/usr/bin/env python3
"""Stream a uniform expert-pruned Flash-Next BF16 checkpoint.

The source checkpoint is read one safetensors shard and one tensor at a time.
Only routed expert/router tensors are sliced; all other language-model tensors
are copied unchanged. Vision and MTP tensors are omitted because the output is
the text-only CausalLM checkpoint consumed by the Flash-Next converter.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import custom_coder_plan  # noqa: E402,I001


EXPERT_PATTERNS = {
    "router": re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$"),
    "gate_up": re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$"),
    "down": re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.down_proj$"),
}
TEXT_PREFIXES = ("model.language_model.", "lm_head")


def classify_tensor(name: str) -> tuple[str, int] | None:
    for kind, pattern in EXPERT_PATTERNS.items():
        match = pattern.fullmatch(name)
        if match is not None:
            return kind, int(match.group(1))
    return None


def prune_shape(shape: Sequence[int], kind: str, keep: int) -> tuple[int, ...]:
    values = tuple(shape)
    if kind == "router":
        if len(values) != 2:
            raise ValueError(f"router shape {values} is not two-dimensional")
        return (keep, values[1])
    if kind in {"gate_up", "down"}:
        if len(values) != 3:
            raise ValueError(f"expert shape {values} is not three-dimensional")
        return (keep, values[1], values[2])
    raise ValueError(f"unknown expert tensor kind {kind}")


def source_name_allowed(name: str) -> bool:
    return name.startswith(TEXT_PREFIXES)


def output_config(source_config: dict[str, Any], keep: int) -> dict[str, Any]:
    text = source_config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("source config has no text_config object")
    result = dict(text)
    result["architectures"] = ["Qwen4ExpForCausalLM"]
    result["num_experts"] = keep
    result["original_num_experts"] = custom_coder_plan.N_EXPERTS
    result["torch_dtype"] = "bfloat16"
    # The materializer intentionally emits only the language model and lm_head.
    # Qwen's text config can still describe a one-layer MTP block, so clear that
    # declaration instead of asking a loader to instantiate tensors that were
    # deliberately omitted.
    result["mtp_num_hidden_layers"] = 0
    if isinstance(result.get("mtp"), dict):
        result["mtp"] = {
            **result["mtp"],
            "layer_types": [],
            "mtp_use_hidden_state_from_layer": None,
            "num_hidden_layers": 0,
        }
    return result


def validate_tokenizer_files(source: Path) -> list[str]:
    prefixes = (
        "tokenizer",
        "special_tokens_map",
        "added_tokens",
        "generation_config",
        "chat_template",
    )
    names: list[str] = []
    for path in sorted(source.iterdir()):
        if not path.is_file() or not path.name.startswith(prefixes):
            continue
        names.append(path.name)
    if not any(name.startswith("tokenizer") for name in names):
        raise ValueError("source checkpoint has no tokenizer files")
    return names


def copy_tokenizer_files(source: Path, output: Path, names: Sequence[str]) -> list[str]:
    copied: list[str] = []
    for name in names:
        target = output / name
        target.write_bytes((source / name).read_bytes())
        copied.append(name)
    return copied


def validate_source_expert_dimension(tensor: Any, kind: str, name: str) -> tuple[int, ...]:
    values = tuple(tensor.shape)
    expected_rank = 2 if kind == "router" else 3
    if len(values) != expected_rank:
        raise ValueError(f"{name}: expert tensor rank {len(values)}, expected {expected_rank}")
    if values[0] != custom_coder_plan.N_EXPERTS:
        raise ValueError(
            f"{name}: source expert dimension {values[0]} "
            f"does not match {custom_coder_plan.N_EXPERTS}"
        )
    return values


def materialize(
    source: Path,
    mask_path: Path,
    output: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    keep, keep_rows = custom_coder_plan.load_own_mask(mask_path)
    if output.exists():
        raise ValueError(f"output already exists; refuse to overwrite: {output}")
    index_path = source / "model.safetensors.index.json"
    config_path = source / "config.json"
    tokenizer_files = validate_tokenizer_files(source)
    source_index = json.loads(index_path.read_text(encoding="utf-8"))
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    mask_document = json.loads(mask_path.read_text(encoding="utf-8"))
    ordered_keep_rows = mask_document.get("keep_ids")
    if not isinstance(ordered_keep_rows, list):
        raise ValueError(f"{mask_path}: keep_ids is missing")
    weight_map: dict[str, str] = source_index.get("weight_map", {})
    if not weight_map:
        raise ValueError("source weight_map is empty")

    expected: Counter[str] = Counter()
    selected_names: list[str] = []
    for name in weight_map:
        classified = classify_tensor(name)
        if classified is not None:
            kind, layer = classified
            if not 0 <= layer < custom_coder_plan.N_LAYERS:
                raise ValueError(f"layer outside [0,{custom_coder_plan.N_LAYERS}): {name}")
            expected[kind] += 1
            selected_names.append(name)
        elif source_name_allowed(name):
            selected_names.append(name)
    layer_count = custom_coder_plan.N_LAYERS
    if expected != Counter({"gate_up": layer_count, "down": layer_count, "router": layer_count}):
        raise ValueError(f"unexpected source expert tensor counts: {dict(expected)}")

    plan = {
        "source": str(source),
        "source_index_sha256": custom_coder_plan.sha256(index_path),
        "source_config_sha256": custom_coder_plan.sha256(config_path),
        "mask": str(mask_path),
        "mask_sha256": custom_coder_plan.sha256(mask_path),
        "output": str(output),
        "keep_per_layer": keep,
        "selected_tensors": len(selected_names),
        "omitted_tensors": len(weight_map) - len(selected_names),
        "tokenizer_files": tokenizer_files,
        "dry_run": dry_run,
    }
    if dry_run:
        return plan

    import torch  # pylint: disable=import-outside-toplevel
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel
    from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel

    output.mkdir(parents=True)
    output_index: dict[str, str] = {}
    total_size = 0
    copied_counts: Counter[str] = Counter()
    source_shards = sorted(set(weight_map.values()))
    selection_index = torch.tensor(ordered_keep_rows, dtype=torch.long)
    started = time.monotonic()

    for shard_number, shard_name in enumerate(source_shards, 1):
        shard_path = source / shard_name
        names = sorted(name for name, target in weight_map.items() if target == shard_name)
        selected_set = set(selected_names)
        selected = {name: weight_map[name] for name in names if name in selected_set}
        tensors: dict[str, Any] = {}
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for name in names:
                if name not in selected:
                    continue
                tensor = handle.get_tensor(name)
                classified = classify_tensor(name)
                if classified is None:
                    tensors[name] = tensor
                    copied_counts["other"] += 1
                    total_size += tensors[name].numel() * tensors[name].element_size()
                else:
                    kind, layer = classified
                    source_shape = validate_source_expert_dimension(tensor, kind, name)
                    wanted = prune_shape(source_shape, kind, keep)
                    tensors[name] = tensor.index_select(0, selection_index[layer])
                    if tuple(tensors[name].shape) != wanted:
                        raise ValueError(
                            f"{name}: pruned shape {tuple(tensors[name].shape)}, expected {wanted}"
                        )
                    copied_counts[kind] += 1
                    total_size += tensors[name].numel() * tensors[name].element_size()
                del tensor
        if tensors:
            save_file(tensors, str(output / shard_name), metadata={"format": "pt"})
            output_index.update(dict.fromkeys(tensors, shard_name))
            del tensors
        print(
            f"[{shard_number}/{len(source_shards)}] {shard_name}: "
            f"wrote {len(output_index)} tensors",
            flush=True,
        )

    if copied_counts != expected + Counter({"other": len(selected_names) - sum(expected.values())}):
        raise ValueError(f"unexpected materialized tensor counts: {dict(copied_counts)}")
    output_index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(output_index.items())),
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(output_index_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(output_config(source_config, keep), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expert_map = {
        "schema": "strata-expert-id-map-v1",
        "layers": custom_coder_plan.N_LAYERS,
        "source_expert_count": custom_coder_plan.N_EXPERTS,
        "physical_expert_count": keep,
        "physical_to_source": ordered_keep_rows,
    }
    (output / "expert-id-map.json").write_text(
        json.dumps(expert_map, indent=2) + "\n", encoding="utf-8"
    )
    copied_tokenizer = copy_tokenizer_files(source, output, tokenizer_files)
    receipt = {
        **plan,
        "format": "strata-pruned-qwen4exp-bf16-v1",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output_shards": len(list(output.glob("*.safetensors"))),
        "output_index_tensors": len(output_index),
        "output_total_size": total_size,
        "materialized_tensor_counts": dict(copied_counts),
        "tokenizer_files": copied_tokenizer,
        "expert_id_map_sha256": custom_coder_plan.sha256(output / "expert-id-map.json"),
        "non_claims": [
            "No model forward, RCO quality search, healing, quantization, GGUF "
            "conversion, or pack verification ran.",
            "The mask is deterministic input; this materializer does not claim "
            "its selection is quality-optimal.",
        ],
    }
    (output / "materialization.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = materialize(args.source, args.mask, args.output, dry_run=args.dry_run)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
