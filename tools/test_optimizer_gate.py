#!/usr/bin/env python3
"""Host-safe tests for the fail-closed optimizer launch gate."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("optimizer_gate", TOOLS / "optimizer_gate.py")
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def ready_document(stage: str) -> dict[str, object]:
    return {
        "format": GATE.FORMAT,
        "stage": stage,
        "requirements": {
            requirement: {
                "artifact": f"{requirement}.json",
                "sha256": "0" * 64,
                "command": ["python3", f"verify_{requirement}.py"],
            }
            for requirement in GATE.STAGE_REQUIREMENTS[stage]
        },
    }


class OptimizerGateTest(unittest.TestCase):
    def test_every_stage_plans_as_blocked_and_names_acceptance(self) -> None:
        for stage in GATE.STAGE_REQUIREMENTS:
            with self.subTest(stage=stage):
                plan = GATE.stage_plan(stage)
                self.assertFalse(plan["gpu_allowed"])
                self.assertEqual(
                    {item["id"] for item in plan["requirements"]},
                    set(GATE.STAGE_REQUIREMENTS[stage]),
                )
                self.assertTrue(all(item["acceptance"] for item in plan["requirements"]))

    def test_complete_manifest_unblocks_only_after_exact_evidence_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ready.json"
            document = ready_document("qwen4exp.rco-quant-aware")
            path.write_text(json.dumps(document), encoding="utf-8")
            result = GATE.check_ready(GATE.load_ready(path))
            self.assertTrue(result["passed"])
            self.assertTrue(result["gpu_allowed"])
            self.assertIn("not optimizer quality", result["non_claim"])

            first = next(iter(document["requirements"]))
            incomplete = json.loads(json.dumps(document))
            del incomplete["requirements"][first]["sha256"]
            path.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256 must be a non-empty string"):
                GATE.load_ready(path)

    def test_unknown_or_incomplete_requirement_set_is_refused(self) -> None:
        document = ready_document("deepseek-v41.rco-prune")
        missing = document["requirements"].pop("resource_plan")
        self.assertIsInstance(missing, dict)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ready.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing=\\['resource_plan'\\]"):
                GATE.load_ready(path)


if __name__ == "__main__":
    unittest.main()
