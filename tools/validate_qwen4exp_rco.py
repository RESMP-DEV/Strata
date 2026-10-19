#!/usr/bin/env python3
"""Validate RCO's Qwen4-Exp router and fused-expert materializer on a tiny model.

This is the pre-GPU compatibility gate.  It instantiates a tiny random
Qwen4ExpForCausalLM (not the 360 GB target), installs RCO's STE pruning wrapper,
executes the fused expert block, physically removes one expert per layer through
RCO's checkpoint materializer, and executes the shrunk block again.  It requires
Transformers >= 5.16 because older releases do not recognize qwen4_exp.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

DEFAULT_RCO = Path.home() / ".local/share/strata-tools/RCO"
MIN_TRANSFORMERS = (5, 16, 0)


def parse_version(value: str) -> tuple[int, ...]:
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)\.([0-9]+).*", value)
    if match is None:
        raise ValueError(f"expected Transformers X.Y.Z, got {value!r}")
    return tuple(int(part) for part in match.groups())


def load_rco_module(root: Path, relative: str, name: str):
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=None,
        ple_embed_dim=None,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        max_position_embeddings=128,
        vocab_size=128,
    )


def run(root: Path, seed: int, as_json: bool) -> dict[str, object]:
    import transformers

    version = str(transformers.__version__)
    if parse_version(version) < MIN_TRANSFORMERS:
        raise RuntimeError(
            f"Transformers {version} lacks qwen4_exp support; "
            f"need >={'.'.join(map(str, MIN_TRANSFORMERS))}"
        )

    sys.path.insert(0, str(root / "src"))
    prune = load_rco_module(root, "src/search/prune.py", "strata_rco_prune_probe")
    builder = load_rco_module(
        root, "run_build_checkpoint.py", "strata_rco_build_checkpoint_probe"
    )

    torch.manual_seed(seed)
    model = Qwen4ExpForCausalLM(tiny_config()).eval()
    info = prune.get_moe_info(model)
    if info != (2, 4, 2):
        raise AssertionError(f"unexpected MoE geometry: {info}")

    layer = model.model.layers[0].mlp
    router_values = layer.gate(torch.randn(3, 32))
    if len(router_values) != 3:
        raise AssertionError(f"unexpected router arity: {len(router_values)}")

    state = {"ste_masks": torch.ones(2, 4)}
    wrappers = prune.install_wrappers(model, info[0], info[1], state)
    try:
        with torch.no_grad():
            patched = layer(torch.randn(1, 3, 32))
    finally:
        prune.remove_wrappers(wrappers)
    if patched.shape != (1, 3, 32) or not bool(torch.isfinite(patched).all()):
        raise AssertionError(f"bad STE output: {patched.shape}")

    for layer_idx in range(info[0]):
        mlp = model.model.layers[layer_idx].mlp
        kept = torch.tensor([0, 1, 2])
        kept_actual = builder._remove_experts(mlp, kept, info[1])
        if kept_actual != 3:
            raise AssertionError(f"layer {layer_idx} kept {kept_actual}, expected 3")
        if mlp.gate.weight.shape[0] != 3:
            raise AssertionError(f"layer {layer_idx} router was not shrunk")
        if mlp.experts.gate_up_proj.shape[0] != 3 or mlp.experts.down_proj.shape[0] != 3:
            raise AssertionError(f"layer {layer_idx} fused experts were not shrunk")

    model.config.num_experts = 3
    model.config.original_num_experts = 4
    with torch.no_grad():
        shrunk = model.model.layers[0].mlp(torch.randn(1, 3, 32))
    if shrunk.shape != (1, 3, 32) or not bool(torch.isfinite(shrunk).all()):
        raise AssertionError(f"bad shrunk output: {shrunk.shape}")

    return {
        "status": "PASS",
        "transformers_version": version,
        "torch_version": torch.__version__,
        "rco_root": str(root),
        "seed": seed,
        "moe_info": list(info),
        "router_class": type(layer.gate).__name__,
        "experts_class": type(layer.experts).__name__,
        "router_arity": len(router_values),
        "patched_output_shape": list(patched.shape),
        "patched_output_finite": True,
        "physical_kept_per_layer": 3,
        "shrunk_output_shape": list(shrunk.shape),
        "shrunk_output_finite": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rco-root", type=Path, default=DEFAULT_RCO)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.rco_root, args.seed, args.json)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("TINY_QWEN4EXP_RCO_PROBE_OK")
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
