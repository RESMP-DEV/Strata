#!/usr/bin/env python3
"""Gate the DeepSeek-V4.1-Flash-1017 checkpoint manifest.

This host-safe static gate compares three inputs and refuses on disagreement:

* ``data/deepseek_v41_manifest.json`` -- the checked-in expected summary of the
  pinned 1017 shape contract: config invariants, checkpoint shape, and the
  tensor-family taxonomy.
* ``config.json`` from the candidate checkpoint.
* ``model.safetensors.index.json`` from the candidate checkpoint.

It reads index metadata only. It never downloads, opens, hashes, or executes a
checkpoint, never claims a tensor dtype or shape, and never proves that a
Strata backend can load the model.

The manifest contains an exact tensor-pattern taxonomy for the pinned official
1017 index. Every expected tensor family is strict, and an unknown or missing
tensor name is a refusal.

Standard library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import sys
import time
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
DEFAULT_MANIFEST = ROOT / "data" / "deepseek_v41_manifest.json"

FAMILY_STATUSES = frozenset({"published", "inferred"})
FAMILY_KINDS = frozenset({"dense", "routed"})
LAYER_GROUPS = ("backbone", "mtp", "vision", "engram", "compressing", "index")
PLACEHOLDERS = ("layer", "expert", "ratio", "slot")
SCOPE = (
    "Compares a DeepSeek-V4.1-Flash checkpoint config.json and "
    "model.safetensors.index.json against data/deepseek_v41_manifest.json on this "
    "host. It does not download, open, hash, convert, or run any checkpoint, does not "
    "verify dtypes, shapes, quantization blocks, or weight values, and does not claim "
    "that any Strata backend can load DeepSeek-V4.1-Flash."
)


class ManifestError(Exception):
    """The checked-in manifest itself is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Family:
    """One expected tensor family compiled into a matcher."""

    name: str
    kind: str
    status: str
    template: str
    regex: re.Pattern[str]
    layers: tuple[int, ...] | None
    experts: tuple[int, ...] | None
    ratios: tuple[str, ...]
    slots: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    """The compiled manifest: derived layer sets and compiled families."""

    layer_sets: dict[str, tuple[int, ...]]
    families: tuple[Family, ...]
    layer_width: int
    expert_width: int

    def family(self, name: str) -> Family:
        for entry in self.families:
            if entry.name == name:
                return entry
        raise KeyError(name)


@dataclass(frozen=True)
class CheckResult:
    """One named gate outcome with a bounded, deterministic detail sentence."""

    name: str
    passed: bool
    detail: str

    def as_json(self) -> dict[str, Any]:
        return {"check": self.name, "passed": self.passed, "detail": self.detail}


def _derive_layer_sets(manifest: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    text = manifest.get("text_config_invariants")
    vision = manifest.get("vision_config_invariants")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ManifestError(
            "manifest is missing text_config_invariants or vision_config_invariants"
        )
    try:
        ratios = [int(value) for value in text["compress_ratios"]]
        layers = int(text["num_hidden_layers"])
        vision_layers = int(vision["num_hidden_layers"])
        mtp_layers = int(text["num_nextn_predict_layers"])
        engram = [int(value) for value in text["engram_layer_ids"]]
        index = [int(value) for value in text["index_source_layer_ids"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError(f"manifest text or vision invariants are unusable: {error}") from error
    # compress_ratios is padded with one ratio per next-token prediction stage,
    # so the vector is num_hidden_layers + num_nextn_predict_layers long.
    expected = layers + mtp_layers
    if len(ratios) != expected:
        raise ManifestError(
            f"compress_ratios has {len(ratios)} entries for {layers} text layers "
            f"and {mtp_layers} DSpark stages"
        )
    return {
        "backbone": tuple(range(layers)),
        "mtp": tuple(range(mtp_layers)),
        "vision": tuple(range(vision_layers)),
        "engram": tuple(engram),
        "compressing": tuple(index for index, ratio in enumerate(ratios) if ratio > 0),
        "index": tuple(index),
    }


def _compile_pattern(
    template: str,
    layer_width: int,
    expert_width: int,
    ratios: tuple[str, ...],
    slots: tuple[str, ...],
) -> re.Pattern[str]:
    parts: list[str] = []
    for token in re.split(r"(\{[a-z_]+\})", template):
        if not token:
            continue
        if token[1:-1] in PLACEHOLDERS:
            key = token[1:-1]
            if key == "layer":
                parts.append(f"(?P<layer>[0-9]{{{layer_width}}})")
            elif key == "expert":
                parts.append(f"(?P<expert>[0-9]{{{expert_width}}})")
            elif key == "ratio":
                if not ratios:
                    raise ManifestError(f"pattern {template} uses {{ratio}} but declares no ratios")
                parts.append("(?:" + "|".join(re.escape(value) for value in ratios) + ")")
            else:
                if not slots:
                    raise ManifestError(f"pattern {template} uses {{slot}} but declares no slots")
                parts.append("(?:" + "|".join(re.escape(value) for value in slots) + ")")
        else:
            parts.append(re.escape(token))
    return re.compile("".join(parts) + r"\Z")


def build_plan(manifest: dict[str, Any]) -> Plan:
    """Compile the manifest into derived layer sets and family matchers."""
    layer_sets = _derive_layer_sets(manifest)
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict):
        raise ManifestError("manifest is missing tensors")
    try:
        layer_width = int(tensors["layer_index_width"])
        expert_width = int(tensors["expert_index_width"])
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError(f"manifest tensor index widths are unusable: {error}") from error
    entries = manifest.get("tensor_families")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("manifest tensor_families must be a non-empty list")

    routed = int(manifest["text_config_invariants"]["n_routed_experts"])
    shared = int(manifest["text_config_invariants"]["n_shared_experts"])
    mtp_routed = int(manifest["text_config_invariants"]["dspark_n_routed_experts"])

    families: list[Family] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ManifestError("every tensor family must be an object")
        name = entry.get("family")
        kind = entry.get("kind")
        status = entry.get("status")
        template = entry.get("pattern")
        if not isinstance(name, str) or not name:
            raise ManifestError("every tensor family needs a non-empty family name")
        if name in seen:
            raise ManifestError(f"duplicate family name {name}")
        seen.add(name)
        if kind not in FAMILY_KINDS:
            raise ManifestError(f"family {name} has unsupported kind {kind!r}")
        if status not in FAMILY_STATUSES:
            raise ManifestError(f"family {name} has unsupported status {status!r}")
        if not isinstance(template, str) or not template:
            raise ManifestError(f"family {name} needs a non-empty pattern")
        selector = entry.get("layers")
        if selector is None:
            layers: tuple[int, ...] | None = None
        elif isinstance(selector, str):
            if selector not in LAYER_GROUPS:
                raise ManifestError(f"family {name} names unknown layer group {selector}")
            layers = layer_sets[selector]
        elif isinstance(selector, list) and all(isinstance(value, int) for value in selector):
            layers = tuple(selector)
        else:
            raise ManifestError(f"family {name} has an unusable layers selector")
        expert_selector = entry.get("experts")
        experts: tuple[int, ...] | None = None
        if expert_selector == "routed":
            if kind != "routed":
                raise ManifestError(f"family {name} selects routed experts but is not routed")
            experts = tuple(range(routed))
        elif expert_selector == "mtp_routed":
            if kind != "routed":
                raise ManifestError(f"family {name} selects routed experts but is not routed")
            experts = tuple(range(mtp_routed))
        elif expert_selector == "shared":
            if kind != "dense":
                raise ManifestError(f"family {name} selects shared experts but is routed")
            experts = tuple(range(shared))
        elif expert_selector is None:
            pass
        else:
            raise ManifestError(f"family {name} has an unusable experts selector")
        ratios = tuple(str(value) for value in entry.get("ratios", []))
        slots = tuple(str(value) for value in entry.get("slots", []))
        families.append(
            Family(
                name=name,
                kind=kind,
                status=status,
                template=template,
                regex=_compile_pattern(template, layer_width, expert_width, ratios, slots),
                layers=layers,
                experts=experts,
                ratios=ratios,
                slots=slots,
            )
        )
    return Plan(
        layer_sets=layer_sets,
        families=tuple(families),
        layer_width=layer_width,
        expert_width=expert_width,
    )


def _load_json(path: Path, label: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise OSError(f"cannot read {label} at {path}: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} at {path} is not valid JSON: {error}") from error


def _compare(expected: Any, actual: Any, path: str, differences: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            differences.append(f"{path} is {type(actual).__name__}, expected object")
            return
        for key in sorted(expected):
            if key not in actual:
                differences.append(f"{path}.{key} is missing")
                continue
            _compare(expected[key], actual[key], f"{path}.{key}", differences)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            differences.append(f"{path} is {type(actual).__name__}, expected list")
            return
        if len(expected) != len(actual):
            differences.append(f"{path} has {len(actual)} entries, expected {len(expected)}")
        for position, (left, right) in enumerate(zip(expected, actual, strict=False)):
            _compare(left, right, f"{path}[{position}]", differences)
        return
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            same = float(expected) == float(actual)
        except (TypeError, ValueError):
            same = False
        if not same:
            differences.append(f"{path} is {actual!r}, expected {expected!r}")
        return
    if expected != actual or type(expected) is not type(actual):
        differences.append(f"{path} is {actual!r}, expected {expected!r}")


def _summary(differences: list[str], limit: int = 12) -> str:
    head = "; ".join(differences[:limit])
    remaining = len(differences) - limit
    if remaining > 0:
        return f"{head}; and {remaining} more"
    return head


def check_config(config: Any, manifest: dict[str, Any]) -> list[CheckResult]:
    """Check the required config invariants and internal consistency."""
    results: list[CheckResult] = []
    if not isinstance(config, dict):
        return [CheckResult("config-structure", False, "config.json is not a JSON object")]
    results.append(CheckResult("config-structure", True, "config.json is a JSON object"))

    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ManifestError("manifest is missing model")
    for name, path, expected in (
        ("config-model-type", "model_type", model["model_type"]),
        ("config-architectures", "architectures", model["architectures"]),
    ):
        actual = config.get(path)
        passed = actual == expected
        results.append(
            CheckResult(
                name,
                passed,
                f"{path} matches the pinned model"
                if passed
                else f"{path} is {actual!r}, expected {expected!r}",
            )
        )

    for name, key, section in (
        ("config-text-invariants", "text_config_invariants", "text_config"),
        ("config-vision-invariants", "vision_config_invariants", "vision_config"),
        ("config-quantization", "quantization_config", "quantization_config"),
    ):
        expected = manifest.get(key)
        differences: list[str] = []
        if not isinstance(expected, dict):
            differences.append(f"manifest {key} is missing")
        else:
            present = config.get(section)
            if not isinstance(present, dict):
                differences.append(f"config {section} is missing")
            else:
                _compare(expected, present, section, differences)
        results.append(
            CheckResult(
                name,
                not differences,
                f"{section} matches every manifest invariant"
                if not differences
                else _summary(differences),
            )
        )
    return results


def _ranked(names: list[str]) -> list[str]:
    def order(name: str) -> tuple[int, int, str]:
        match = re.fullmatch(r"model-(\d{5})-of-\d{5}\.safetensors", name)
        if match:
            return (0, int(match.group(1)), name)
        return (1, 0, name)

    return sorted(set(names), key=order)


def _shard_counts(weight_map: dict[str, Any], shards: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in shards}
    for shard in weight_map.values():
        if isinstance(shard, str) and shard in counts:
            counts[shard] += 1
    return counts


def _index_problems(index: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        problems.append("metadata is missing or not an object")
    total_size = metadata.get("total_size")
    if isinstance(total_size, bool) or not isinstance(total_size, int):
        problems.append("total_size is missing or not an integer")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        problems.append("weight_map is missing, empty, or not an object")
    elif any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items()
    ):
        problems.append("weight_map has non-string keys or shard file names")
    return problems


def check_index(index: dict[str, Any], manifest: dict[str, Any]) -> list[CheckResult]:
    """Check index metadata, checkpoint shape, and per-shard consistency."""
    checkpoints = manifest["checkpoint"]
    tensors = manifest["tensors"]
    expected_total_bytes = int(tensors["index_total_size_bytes"])

    results: list[CheckResult] = [
        CheckResult(
            "index-total-size",
            index["metadata"]["total_size"] == expected_total_bytes,
            f"index metadata total_size {index['metadata']['total_size']} "
            + (
                f"matches the pinned {expected_total_bytes}"
                if index["metadata"]["total_size"] == expected_total_bytes
                else f"differs from the pinned {expected_total_bytes}"
            ),
        ),
        CheckResult(
            "tensor-count",
            len(index["weight_map"]) == int(tensors["total"]),
            f"{len(index['weight_map'])} tensors, expected {tensors['total']}",
        ),
    ]

    listed: set[str] = set()
    for shard in index["weight_map"].values():
        match = re.fullmatch(
            str(checkpoints["shard_file_pattern"])
            .replace("{index:05d}", r"(\d{5})")
            .replace("{total:05d}", r"\d{5}"),
            shard,
        )
        if match:
            listed.add(shard)
    expected_shards = int(checkpoints["shards"])
    if len(listed) == expected_shards:
        results.append(
            CheckResult(
                "shard-count",
                True,
                f"{len(listed)} ranked shards, expected {expected_shards}",
            )
        )
    else:
        results.append(
            CheckResult(
                "shard-count",
                False,
                f"{len(listed)} ranked shards, expected {expected_shards}",
            )
        )

    shards = _ranked(list(index["weight_map"].values()))
    counts = _shard_counts(index["weight_map"], shards)
    low, high = (int(value) for value in tensors["shard_tensor_count_range"])
    underfilled = [name for name in shards if counts[name] < low]
    overfilled = [name for name in shards if counts[name] > high]
    if underfilled or overfilled:
        detail = []
        if underfilled:
            detail.append(f"{len(underfilled)} shards below {low} tensors ({underfilled[:2]})")
        if overfilled:
            detail.append(f"{len(overfilled)} shards above {high} tensors ({overfilled[:2]})")
        results.append(CheckResult("shard-consistency", False, _summary(detail)))
    else:
        results.append(
            CheckResult(
                "shard-consistency",
                True,
                f"every one of {len(shards)} referenced shards holds between {low} "
                f"and {high} tensors",
            )
        )
    return results


def _expand(family: Family, plan: Plan) -> list[str]:
    layer_width, expert_width = plan.layer_width, plan.expert_width

    values: list[str] = []
    layers: tuple[int | None, ...] = family.layers or (None,)
    experts: tuple[int | None, ...] = family.experts or (None,)
    ratios: tuple[str | None, ...] = family.ratios or (None,)
    slots: tuple[str | None, ...] = family.slots or (None,)
    for layer in layers:
        for expert in experts:
            for ratio in ratios:
                for slot in slots:
                    name = family.template
                    if layer is not None:
                        name = name.replace("{layer}", f"{layer:0{layer_width}d}")
                    if expert is not None:
                        name = name.replace("{expert}", f"{expert:0{expert_width}d}")
                    if ratio is not None:
                        name = name.replace("{ratio}", ratio)
                    if slot is not None:
                        name = name.replace("{slot}", slot)
                    values.append(name)
    return values


def expected_tensor_names(plan: Plan) -> dict[str, str]:
    """Return the expected tensor name to owning family mapping."""
    mapping: dict[str, str] = {}
    for family in plan.families:
        for name in _expand(family, plan):
            if name in mapping:
                raise ManifestError(f"families {mapping[name]} and {family.name} both claim {name}")
            mapping[name] = family.name
    return mapping


def check_family_totals(observed: dict[str, int], plan: Plan) -> list[CheckResult]:
    """Compare observed and expected per-family tensor counts."""
    published: list[str] = []
    inferred: list[str] = []
    for family in plan.families:
        expected = len(_expand(family, plan))
        if observed[family.name] == expected:
            continue
        if family.status == "published":
            published.append(family.name)
        else:
            inferred.append(f"{family.name}: {observed[family.name]} observed, {expected} expected")
    return [
        CheckResult(
            "family-totals",
            not published,
            "every published family holds its expected tensor count"
            if not published
            else _summary([f"published family {name} count differs" for name in published]),
        ),
        CheckResult(
            "inferred-family-report",
            True,
            _summary(inferred) if inferred else "every inferred family matches exactly too",
        ),
    ]


def check_tensor_coverage(weight_map: dict[str, Any], plan: Plan) -> list[CheckResult]:
    """Match every indexed tensor to exactly one family, or refuse."""
    expected = expected_tensor_names(plan)
    missing = sorted(name for name in expected if name not in weight_map)
    owners: dict[str, list[str]] = {}
    for tensor, shard in weight_map.items():
        owners.setdefault(tensor, []).append(shard)
    duplicated = sorted(name for name, files in owners.items() if len(files) > 1)
    unknown = sorted(name for name in owners if name not in expected)

    results: list[CheckResult] = [
        CheckResult(
            "tensor-duplicates",
            not duplicated,
            "every indexed tensor names exactly one shard"
            if not duplicated
            else _summary([f"{len(duplicated)} tensors map to several shards ({duplicated[:3]})"]),
        )
    ]
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"{len(missing)} expected tensors are absent ({missing[:3]})")
        if unknown:
            detail.append(f"{len(unknown)} tensors match no family ({unknown[:3]})")
        results.append(CheckResult("tensor-coverage", False, _summary(detail)))
    else:
        results.append(
            CheckResult(
                "tensor-coverage",
                True,
                f"all {len(expected)} expected tensors are present and classified",
            )
        )

    observed = dict.fromkeys((family.name for family in plan.families), 0)
    for tensor, family_name in expected.items():
        if tensor in owners:
            observed[family_name] += 1
    results.extend(check_family_totals(observed, plan))
    return results


def run_gate(
    manifest: dict[str, Any],
    plan: Plan,
    *,
    config: Any = None,
    index: Any = None,
) -> list[CheckResult]:
    """Run every available gate and return the results in report order."""
    results: list[CheckResult] = []
    if config is not None:
        results.extend(check_config(config, manifest))
    if index is not None:
        if not isinstance(index, dict):
            results.append(CheckResult("index-structure", False, "the index is not a JSON object"))
            return results
        problems = _index_problems(index)
        if problems:
            results.append(CheckResult("index-structure", False, _summary(problems)))
            results.extend(
                CheckResult(name, False, "not evaluated because the index metadata is invalid")
                for name in (
                    "index-total-size",
                    "tensor-count",
                    "shard-count",
                    "shard-consistency",
                    "tensor-duplicates",
                    "tensor-coverage",
                    "family-totals",
                )
            )
            return results
        results.append(
            CheckResult(
                "index-structure", True, "weight_map and metadata.total_size are well formed"
            )
        )
        results.extend(check_index(index, manifest))
        results.extend(check_tensor_coverage(index["weight_map"], plan))
    return results


def compile_manifest(path: Path, report: Callable[[str], None]) -> tuple[dict[str, Any], Plan, int]:
    """Compile the manifest and self-check it, reporting the result once.

    The manifest is the contract, so loading it is itself a check: a malformed
    manifest, an unknown family status, two families claiming the same tensor, or
    a placeholder pattern with no placeholder selector all fail before any
    candidate file is read. With no ``--config`` and no ``--index`` this is the
    only work the command does, which is why it reports and exits 0 instead of
    refusing as a usage error.
    """
    if path.is_dir():
        raise ValueError(f"manifest at {path} is a directory, not a JSON file")
    manifest = _load_json(path, "manifest")
    plan = build_plan(manifest)
    names = expected_tensor_names(plan)
    routed = sum(len(_expand(family, plan)) for family in plan.families if family.kind == "routed")
    report(
        f"PASS manifest-schema: {len(plan.families)} families own {len(names)} expected "
        f"tensors, {routed} of them routed-expert tensors"
    )
    return manifest, plan, len(names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a DeepSeek-V4.1-Flash checkpoint config.json and "
            "model.safetensors.index.json against the checked-in shape manifest."
        )
    )
    parser.add_argument(
        "--config", type=Path, help="candidate config.json; enables the config gates"
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="candidate model.safetensors.index.json; enables the index gates",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"expected manifest to compile (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON report on stdout")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the colocated synthetic fixture tests and exit",
    )
    return parser


def _run_selftest() -> int:
    """Run the colocated fixture tests, loaded by path so no package name collides."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    location = SCRIPTS / "tests" / "test_check_deepseek.py"
    specification = importlib.util.spec_from_file_location("strata_test_check_deepseek", location)
    if specification is None or specification.loader is None:
        print(f"FAIL selftest-loader: cannot load {location}")
        return 2
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    suite = unittest.TestLoader().loadTestsFromModule(module)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.selftest:
        return _run_selftest()
    report: Callable[[str], None] = (
        (lambda line: print(line, file=sys.stderr)) if arguments.json else print
    )
    try:
        manifest, plan, expected_names = compile_manifest(arguments.manifest, report)
    except (ValueError, ManifestError, KeyError) as error:
        report(f"FAIL manifest-schema: {error}")
        return 2
    try:
        config = _load_json(arguments.config, "config") if arguments.config else None
        index = _load_json(arguments.index, "index") if arguments.index else None
    except (OSError, ValueError) as error:
        report(f"FAIL input-load: {error}")
        return 2

    results = run_gate(manifest, plan, config=config, index=index)
    if config is None and index is None:
        results.append(
            CheckResult(
                "manifest-input", True, "no candidate input, so only the manifest was gated"
            )
        )
    passed = all(result.passed for result in results)
    if arguments.json:
        print(
            json.dumps(
                {
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "platform": {"system": platform.platform(), "python": sys.version},
                    "scope": SCOPE,
                    "manifest": str(arguments.manifest),
                    "inputs": {
                        "config": str(arguments.config) if arguments.config else None,
                        "index": str(arguments.index) if arguments.index else None,
                    },
                    "expected_tensor_names": expected_names,
                    "checks": [result.as_json() for result in results],
                    "aggregate_passed": passed,
                },
                indent=2,
            )
        )
        return 0 if passed else 1
    for result in results:
        print(f"{'PASS' if result.passed else 'FAIL'} {result.name}: {result.detail}")
    print(f"{'PASS' if passed else 'FAIL'} manifest-gate: {len(results)} checks executed")
    if passed:
        print("Non-claims: index metadata only; no checkpoint was opened, hashed, or run.")
        print("No dtype, shape, quantization block, or weight value was verified.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
