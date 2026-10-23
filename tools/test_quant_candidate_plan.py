#!/usr/bin/env python3
"""Host-safe tests for exact W4 candidate and BF16 oracle planning."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "quant_candidate_plan", TOOLS / "quant_candidate_plan.py"
)
assert SPEC is not None and SPEC.loader is not None
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cost_manifest(root: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = root / f"{name}.json"
    payload = json.dumps({"format": "exact-cost-v1", "rows": rows}).encode()
    path.write_bytes(payload)
    return {
        "path": path.name,
        "sha256": digest(payload),
        "bytes": len(payload),
    }


def candidate(root: Path, identifier: str) -> dict[str, Any]:
    rows = [
        {"tensor": "layer.0.gate", "bytes": 4_096},
        {"tensor": "layer.0.down", "bytes": 8_192},
    ]
    return {
        "id": identifier,
        "weight_bits": 4,
        "activation_bits": 16 if identifier == "w4a16" else 4,
        "cost_artifact": cost_manifest(root, identifier, rows),
        "tensor_costs": rows,
    }


def input_document(root: Path) -> dict[str, Any]:
    return {
        "format": PLAN.INPUT_FORMAT,
        "model": {"id": "fixture-model", "sha256": "1" * 64},
        "splits": {
            "calibration": {
                "dataset": "private-cal",
                "sequences": 4,
                "sequence_length": 128,
                "tokens_sha256": "2" * 64,
            },
            "heldout": {
                "dataset": "private-heldout",
                "sequences": 2,
                "sequence_length": 128,
                "tokens_sha256": "3" * 64,
            },
        },
        "candidates": [candidate(root, "w4a16"), candidate(root, "w4a4")],
        "reference_oracle": {
            "format": "bf16-topk-log-probs",
            "model_precision": "BF16",
            "topk": 20,
            "calibration": {"bytes": 1234, "sha256": "4" * 64},
            "heldout": {"bytes": 617, "sha256": "5" * 64},
        },
    }


class QuantCandidatePlanTest(unittest.TestCase):
    def test_exact_candidate_and_reference_identities_produce_a_blocked_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = PLAN.build_plan(input_document(root), root=root)
            self.assertEqual([arm["id"] for arm in result["arms"]], ["w4a16", "w4a4"])
            self.assertEqual([arm["activation_bits"] for arm in result["arms"]], [16, 4])
            self.assertEqual(result["arms"][0]["deployment_weight_bytes"], 12_288)
            self.assertFalse(result["gpu_allowed"])
            self.assertIn("optimizer readiness gate", " ".join(result["non_claims"]))

    def test_nominal_cost_without_a_hashed_artifact_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = input_document(root)
            document["candidates"][0].pop("cost_artifact")
            with self.assertRaisesRegex(ValueError, "cost_artifact must be an object"):
                PLAN.build_plan(document, root=root)

    def test_wrong_activation_label_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = input_document(root)
            document["candidates"][1]["activation_bits"] = 16
            with self.assertRaisesRegex(ValueError, "w4a4 requires activation_bits=4"):
                PLAN.build_plan(document, root=root)

    def test_train_and_heldout_streams_must_differ(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = input_document(root)
            shared = document["splits"]["calibration"]["tokens_sha256"]
            document["splits"]["heldout"]["tokens_sha256"] = shared
            with self.assertRaisesRegex(ValueError, "token streams must differ"):
                PLAN.build_plan(document, root=root)

    def test_explicit_w4a16_only_mode_produces_a_single_arm_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = input_document(root)
            document["candidate_mode"] = "w4a16-only"
            document["candidates"] = [candidate(root, "w4a16")]
            result = PLAN.build_plan(document, root=root)
            self.assertEqual(result["candidate_mode"], "w4a16-only")
            self.assertEqual([arm["id"] for arm in result["arms"]], ["w4a16"])
            self.assertEqual(result["arms"][0]["activation_bits"], 16)
            self.assertFalse(result["gpu_allowed"])
            claims = " ".join(result["non_claims"])
            self.assertIn("not RCO multi-candidate optimization", claims)
            self.assertIn("W4A4", claims)

    def test_single_arm_without_an_explicit_mode_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = input_document(root)
            document["candidates"] = [candidate(root, "w4a16")]
            with self.assertRaisesRegex(ValueError, "exactly W4A16 and W4A4 arms"):
                PLAN.build_plan(document, root=root)

    def test_unknown_candidate_mode_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = input_document(root)
            document["candidate_mode"] = "w4a4-only"
            with self.assertRaisesRegex(ValueError, "candidate_mode must be one of"):
                PLAN.build_plan(document, root=root)

    def test_cost_artifact_byte_or_hash_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = input_document(root)
            document["candidates"][0]["cost_artifact"]["bytes"] += 1
            with self.assertRaisesRegex(ValueError, "declared bytes"):
                PLAN.build_plan(document, root=root)

            mismatch = input_document(root)
            mismatch["candidates"][1]["cost_artifact"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                PLAN.build_plan(mismatch, root=root)


if __name__ == "__main__":
    unittest.main()
