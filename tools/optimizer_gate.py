#!/usr/bin/env python3
"""Fail-closed planning gate for RCO/GSQ optimizer launches.

The gate is deliberately host-safe: it never imports Torch, opens a model
checkpoint, contacts a remote host, or launches an optimizer. A green result
requires every named evidence file to match its SHA-256 and every host
verification command to exit zero. It remains permission to invoke the named
implementation, never evidence that the optimizer itself passed.
"""

from __future__ import annotations

# The acceptance prose is intentionally explicit; the host-safe behavior is tested.
# ruff: noqa: E501
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

FORMAT = "strata-optimizer-gate-v1"

STAGE_REQUIREMENTS: dict[str, dict[str, str]] = {
    "qwen4exp.rco-prune": {
        "rco_qwen4exp_tiny_parity": (
            "Tiny Qwen4Exp stock-RCO parity receipt for router arity, fused experts, and mask restoration"
        ),
        "streaming_adapter_runner": (
            "Reproducible streaming runner covering Qwen4Exp topology, PLE, and hyper-connection inputs"
        ),
        "calibration_tokens": "Pinned train/validation token tensors and raw i32 streams",
        "heldout_kl_gate": "Pre-registered held-out KL gate and control mask",
        "resource_plan": "Named host, visible GPUs, CPU/RAM/disk budget, and resume/checkpoint plan",
    },
    "qwen4exp.rco-quant-aware": {
        "streaming_adapter_runner": (
            "Reproducible runner that substitutes Qwen4Exp topology functions without editing stock RCO"
        ),
        "candidate_store": (
            "RCO WeightStore directory containing every deployable Qwen4Exp candidate tensor and bitwidth"
        ),
        "exact_byte_cost_model": (
            "Candidate costs computed from exact GGUF/tensor byte sizes, not nominal bitwidth labels"
        ),
        "deployment_exporter": (
            "Assignment-to-GGUF recipe/export path with tensor-level inversion or parity proof"
        ),
        "cached_reference": (
            "Pinned full-model reference-log cache for the calibration and held-out splits"
        ),
        "candidate_smoke": (
            "Host or tiny-GPU proof that candidate interpolation preserves the target model semantics"
        ),
        "resource_plan": "Named host, visible GPUs, candidate disk layout, CPU/RAM budget, and resume plan",
    },
    "qwen4exp.gsq": {
        "model_wrapper": (
            "GSQ Qwen4Exp dispatch and layer movement for model.language_model.layers"
        ),
        "fused_3d_expert_quantization": (
            "Per-expert quantization for gate_up_proj/down_proj with per-expert group scales"
        ),
        "ple_ngram_support": (
            "Layer-1 PLE injection plus 128 ngram shard loading and state handling"
        ),
        "hyper_connection_support": (
            "hc_count-width residual streams and all hyper-connection tensor movement"
        ),
        "hybrid_attention_support": (
            "Correct linear-attention and full-attention layer forward and cache inputs"
        ),
        "non_target_ignore_contract": (
            "Shared expert, router, PLE, hyper-connection, and other non-target tensors explicitly omitted or preserved"
        ),
        "candidate_output_contract": (
            "Exact compressed-tensors/Humming or GGUF output contract and per-tensor inversion proof"
        ),
        "calibration_identity": "Pinned tokenizer and train/validation corpus with no train/held-out overlap",
        "tiny_wrapper_parity": "Tiny-model parity against the unmodified Qwen4Exp block",
        "resource_plan": "Named host, visible GPUs, layer movement budget, scratch space, and resume plan",
    },
    "deepseek-v41.rco-prune": {
        "adapter_runner": (
            "Reproducible DeepSeek adapter-injection runner bypassing stock topology and softmax assumptions"
        ),
        "faithful_router_stats": (
            "Router score/frequency paths using V4.1 weights, indices, image_mask, and correction biases"
        ),
        "calibration_tokens": "Pinned V4.1 tokenizer and train/validation RCO token tensors",
        "materialization_executor": (
            "Streaming executor for the verified 96,085-tensor omit/copy/slice plan with exact payload total"
        ),
        "resource_plan": "Named host, visible GPUs, CPU/RAM/disk budget, and resume/checkpoint plan",
    },
    "deepseek-v41.rco-quant-aware": {
        "adapter_runner": (
            "Reproducible DeepSeek adapter-injection runner bypassing stock topology and softmax assumptions"
        ),
        "candidate_store": (
            "RCO WeightStore candidates that preserve V4.1 source FP4/F8 dequantization semantics"
        ),
        "exact_byte_cost_model": (
            "Candidate costs computed from exact target tensor bytes and preserved DSpark/MTP tensors"
        ),
        "deployment_exporter": (
            "DeepSeek target-format exporter with tensor-level inversion or bounded parity proof"
        ),
        "cached_reference": "Pinned V4.1 full-model reference-log cache for both scoring splits",
        "candidate_smoke": (
            "Tiny-model proof for V4.1 routing weight-before-down-projection and shared-expert semantics"
        ),
        "resource_plan": "Named host, visible GPUs, candidate disk layout, CPU/RAM budget, and resume plan",
    },
    "deepseek-v41.gsq": {
        "model_wrapper": (
            "GSQ wrapper covering V4.1 fused experts, shared experts, image-mask router, and source quantization"
        ),
        "engram_dspark_boundary": (
            "Explicit omit/preserve/refine contract for Engram, DSpark, and MTP tensors"
        ),
        "candidate_output_contract": (
            "Exact DeepSeek target-format output contract and per-tensor inversion or parity proof"
        ),
        "calibration_identity": "Pinned V4.1 tokenizer and train/validation corpus with no train/held-out overlap",
        "tiny_wrapper_parity": "Tiny-model parity against the pinned V4.1 source implementation",
        "resource_plan": "Named host, visible GPUs, layer movement budget, scratch space, and resume plan",
    },
}


def stage_plan(stage: str) -> dict[str, Any]:
    if stage not in STAGE_REQUIREMENTS:
        raise ValueError(f"unknown optimizer stage: {stage}")
    return {
        "format": FORMAT,
        "stage": stage,
        "gpu_allowed": False,
        "requirements": [
            {"id": requirement, "acceptance": acceptance}
            for requirement, acceptance in STAGE_REQUIREMENTS[stage].items()
        ],
    }


def load_ready(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise ValueError(f"{path}: expected format {FORMAT}")
    stage = document.get("stage")
    if not isinstance(stage, str):
        raise ValueError(f"{path}: stage must be a string")
    if stage not in STAGE_REQUIREMENTS:
        raise ValueError(f"{path}: unknown or missing stage {stage!r}")
    requirements = document.get("requirements")
    if not isinstance(requirements, dict):
        raise ValueError(f"{path}: requirements must be an object")
    expected = set(STAGE_REQUIREMENTS[stage])
    observed = set(requirements)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{path}: requirement set mismatch; missing={missing}, extra={extra}")
    for requirement, evidence in requirements.items():
        if not isinstance(evidence, dict):
            raise ValueError(f"{path}:{requirement}: evidence must be an object")
        for field in ("artifact", "sha256"):
            value = evidence.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{path}:{requirement}: {field} must be a non-empty string")
        artifact = Path(evidence["artifact"])
        if artifact.is_absolute() or ".." in artifact.parts:
            raise ValueError(
                f"{path}:{requirement}: artifact must stay inside the manifest directory"
            )
        digest = evidence["sha256"]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(
                f"{path}:{requirement}: sha256 must be 64 lowercase hexadecimal characters"
            )
        command = evidence.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) and part for part in command)
        ):
            raise ValueError(f"{path}:{requirement}: command must be a non-empty argv list")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_ready(path: Path, *, command_timeout_seconds: int = 900) -> dict[str, Any]:
    document = load_ready(path)
    root = path.parent
    stage = document["stage"]
    requirements: dict[str, dict[str, Any]] = document["requirements"]

    for _requirement, evidence in requirements.items():
        artifact = root / evidence["artifact"]
        if not artifact.is_file():
            raise ValueError(f"{artifact}: required evidence artifact is not a regular file")
        observed_digest = _sha256(artifact)
        if observed_digest != evidence["sha256"]:
            raise ValueError(
                f"{artifact}: SHA-256 mismatch; expected {evidence['sha256']}, "
                f"got {observed_digest}"
            )

    for requirement, evidence in requirements.items():
        completed = subprocess.run(
            evidence["command"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=command_timeout_seconds,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(
                f"{requirement}: verification command exited {completed.returncode}: {detail}"
            )

    return {
        "format": FORMAT,
        "stage": stage,
        "passed": True,
        "gpu_allowed": True,
        "requirement_count": len(STAGE_REQUIREMENTS[stage]),
        "artifact_sha256_verified": len(requirements),
        "verification_commands_passed": len(requirements),
        "non_claim": (
            "This gate verifies named artifact bytes and host verification commands only; "
            "it does not prove optimizer quality or runtime fitness."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="verb", required=True)
    plan = subparsers.add_parser("plan", help="emit a stage's fail-closed requirements")
    plan.add_argument("--stage", required=True, choices=sorted(STAGE_REQUIREMENTS))
    check = subparsers.add_parser("check", help="check a ready manifest")
    check.add_argument("--ready", required=True, type=Path)
    check.add_argument("--command-timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verb == "plan":
        print(json.dumps(stage_plan(args.stage), indent=2, sort_keys=True))
        return 0
    try:
        result = check_ready(
            args.ready,
            command_timeout_seconds=args.command_timeout,
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        print(
            json.dumps(
                {"format": FORMAT, "passed": False, "gpu_allowed": False, "error": str(error)}
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
