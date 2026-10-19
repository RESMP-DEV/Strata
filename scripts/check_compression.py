#!/usr/bin/env python3
"""Preflight the Strata-owned Qwen expert-pruning and quantization inputs.

This check is deliberately host-safe.  It reads JSON metadata and file names,
never opens a safetensors payload, downloads a model, imports torch, or starts
GPU work.  Its positive result is therefore an input/runtime compatibility
check, not a pruning, healing, quantization, packing, or model-run claim.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GSQ = Path.home() / ".local/share/strata-tools/GSQ"
DEFAULT_RCO = Path.home() / ".local/share/strata-tools/RCO"
GSQ_REVISION = "03fc16484c369e3127225615d5e03e8d3a6043e3"
RCO_REVISION = "9a1e09c07d468109cbe60a1b87d5036034a79d10"
MIN_TRANSFORMERS = (5, 16, 0)
RCO_DECLARED_TRANSFORMERS = (5, 7, 0)

EXPECTED_TEXT = {
    "model_type": "qwen4_exp_text",
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "ple_layer_ids": [2],
    "split_ngram_parts": 128,
}


@dataclass
class Finding:
    name: str
    ok: bool
    detail: str


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_version(value: str) -> tuple[int, ...]:
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)\.([0-9]+).*", value.strip())
    if match is None:
        raise ValueError(f"not a semantic X.Y.Z version: {value!r}")
    return tuple(int(part) for part in match.groups())


def installed_transformers_version() -> str | None:
    try:
        import transformers  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return str(getattr(transformers, "__version__", ""))


def layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}"


def validate_model(model: Path) -> list[Finding]:
    findings: list[Finding] = []
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"

    try:
        config = read_json(config_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [Finding("model.config", False, f"{config_path}: {exc}")]
    try:
        index = read_json(index_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [Finding("model.index", False, f"{index_path}: {exc}")]

    architecture_ok = "Qwen4ExpForConditionalGeneration" in config.get("architectures", [])
    findings.append(
        Finding(
            "model.architecture",
            architecture_ok,
            f"architectures={config.get('architectures')!r}",
        )
    )
    findings.append(
        Finding(
            "model.type",
            config.get("model_type") == "qwen4_exp",
            f"model_type={config.get('model_type')!r}",
        )
    )

    text = config.get("text_config")
    if not isinstance(text, dict):
        findings.append(
            Finding("model.text_config", False, "text_config is missing or is not an object")
        )
        return findings

    for field, expected in EXPECTED_TEXT.items():
        actual = text.get(field)
        findings.append(
            Finding(
                f"model.{field}",
                actual == expected,
                f"expected={expected!r} actual={actual!r}",
            )
        )

    layer_types = text.get("layer_types")
    actual_count = len(layer_types) if isinstance(layer_types, list) else -1
    actual_kind = type(layer_types).__name__
    if not isinstance(layer_types, list) or len(layer_types) != 48:
        findings.append(
            Finding(
                "model.layer_types",
                False,
                f"expected 48 entries, actual={actual_count} ({actual_kind})",
            )
        )
    else:
        counts = Counter(layer_types)
        layer_types_ok = counts == {"linear_attention": 36, "full_attention": 12}
        findings.append(
            Finding(
                "model.layer_types",
                layer_types_ok,
                f"linear_attention={counts['linear_attention']} "
                f"full_attention={counts['full_attention']}",
            )
        )

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        findings.append(Finding("model.weight_map", False, "weight_map is missing or empty"))
        return findings

    expected_expert_names: set[str] = set()
    for layer in range(48):
        expected_expert_names.update(
            {
                f"{layer_prefix(layer)}.mlp.gate.weight",
                f"{layer_prefix(layer)}.mlp.experts.gate_up_proj",
                f"{layer_prefix(layer)}.mlp.experts.down_proj",
            }
        )
    missing_experts = sorted(expected_expert_names - weight_map.keys())
    findings.append(
        Finding(
            "model.expert_tensors",
            not missing_experts,
            f"missing={missing_experts[:3]}{'...' if len(missing_experts) > 3 else ''} "
            f"total_missing={len(missing_experts)}",
        )
    )

    ngram_pattern = re.compile(
        r"^model\.language_model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$"
    )
    ngram_names = [name for name in weight_map if ngram_pattern.fullmatch(name)]
    findings.append(
        Finding(
            "model.ple_shards",
            len(ngram_names) == 128,
            f"expected=128 actual={len(ngram_names)}",
        )
    )

    shards = sorted(set(weight_map.values()))
    missing_shards = [name for name in shards if not (model / name).is_file()]
    findings.append(
        Finding(
            "model.shard_files",
            not missing_shards,
            f"expected={len(shards)} missing={len(missing_shards)} "
            f"first_missing={missing_shards[:2]}",
        )
    )
    transfer_markers = sorted(model.glob("*.aria2")) + sorted(model.glob("*.incomplete"))
    findings.append(
        Finding(
            "model.download_complete",
            not transfer_markers,
            f"markers={[path.name for path in transfer_markers[:3]]}",
        )
    )

    metadata = index.get("metadata", {})
    total_size = metadata.get("total_size")
    size_on_disk = sum((model / name).stat().st_size for name in shards if (model / name).is_file())
    size_ok = (
        isinstance(total_size, (int, float))
        and not isinstance(total_size, bool)
        and size_on_disk >= total_size
    )
    findings.append(
        Finding(
            "model.index_size",
            size_ok,
            f"index_total={total_size} shard_bytes_on_disk={size_on_disk}",
        )
    )
    return findings


def validate_tool(name: str, root: Path, revision: str, required: tuple[str, ...]) -> list[Finding]:
    findings: list[Finding] = [
        Finding(f"{name}.repo", root.is_dir(), f"path={root}"),
    ]
    if not root.is_dir():
        return findings

    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    actual = completed.stdout.strip()
    findings.append(
        Finding(
            f"{name}.revision",
            completed.returncode == 0 and actual == revision,
            f"expected={revision} actual={actual or completed.stderr.strip()}",
        )
    )

    missing = [str(relative) for relative in required if not (root / relative).is_file()]
    findings.append(
        Finding(
            f"{name}.entrypoints",
            not missing,
            f"missing={missing}",
        )
    )
    return findings


def validate_runtime(version: str | None) -> list[Finding]:
    findings: list[Finding] = []
    if version is None:
        findings.append(
            Finding("runtime.transformers", False, "not installed in this Python environment")
        )
        return findings
    try:
        parsed = parse_version(version)
    except ValueError as exc:
        findings.append(Finding("runtime.transformers", False, str(exc)))
        return findings
    findings.append(
        Finding(
            "runtime.transformers",
            parsed >= MIN_TRANSFORMERS,
            f"installed={version} required=>={'.'.join(map(str, MIN_TRANSFORMERS))} "
            f"rco_declared={'.'.join(map(str, RCO_DECLARED_TRANSFORMERS))}",
        )
    )
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="local Qwen3.8-Flash-Next BF16 checkpoint"
    )
    parser.add_argument("--gsq-root", type=Path, default=DEFAULT_GSQ)
    parser.add_argument("--rco-root", type=Path, default=DEFAULT_RCO)
    parser.add_argument(
        "--transformers-version",
        default=None,
        help="override the installed Transformers version for an isolated environment; X.Y.Z",
    )
    parser.add_argument(
        "--skip-tools", action="store_true", help="validate only checkpoint and runtime metadata"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings = validate_model(args.model)
    version = args.transformers_version or installed_transformers_version()
    findings.extend(validate_runtime(version))
    if not args.skip_tools:
        findings.extend(
            validate_tool(
                "gsq",
                args.gsq_root,
                GSQ_REVISION,
                ("main.py", "save_model.py", "scripts/run.sh", "pyproject.toml"),
            )
        )
        findings.extend(
            validate_tool(
                "rco",
                args.rco_root,
                RCO_REVISION,
                (
                    "rco_search_prune.py",
                    "run_build_checkpoint.py",
                    "src/search/prune.py",
                    "requirements.txt",
                ),
            )
        )

    rows = [finding.__dict__ for finding in findings]
    passed = all(finding.ok for finding in findings)
    if args.json:
        print(json.dumps({"passed": passed, "findings": rows}, indent=2))
    else:
        for finding in findings:
            mark = "PASS" if finding.ok else "FAIL"
            print(f"{mark} {finding.name}: {finding.detail}")
        print(
            f"{'PASS' if passed else 'FAIL'} compression-preflight: "
            "metadata and pinned inputs checked; no model execution claimed"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
