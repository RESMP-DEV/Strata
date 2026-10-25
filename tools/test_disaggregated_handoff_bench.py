#!/usr/bin/env python3
"""Host-safe tests for the cross-engine handoff benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "disaggregated_handoff_bench",
    Path(__file__).resolve().parent / "disaggregated_handoff_bench.py",
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def reply(worker: int, completion: int = 4) -> dict:
    return {
        "wall_s": 0.1 + worker * 0.01,
        "first_token_s": 0.02,
        "prompt_tokens": 100,
        "cached_tokens": 80,
        "fresh_tokens": 20,
        "cache_ratio": 0.8,
        "completion_tokens": completion,
        "timings": {},
        "text": f"ok-{worker}",
        "error": None,
    }


class DisaggregatedHandoffBenchTest(unittest.TestCase):
    def test_context_and_requests_are_deterministic(self) -> None:
        context = BENCH.shared_context(64)
        self.assertEqual(context, BENCH.shared_context(64))
        first = BENCH.request_messages(context, 2, 0)
        self.assertEqual(first, BENCH.request_messages(BENCH.shared_context(64), 2, 0))
        self.assertIn("Worker 2, round 0", first[1]["content"])

    def test_request_summary_counts_only_valid_rows(self) -> None:
        records = [reply(0), reply(1, 6), {**reply(2), "error": "timeout"}]
        summary = BENCH.summarize_requests(records)
        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["prompt_tokens"], 200)
        self.assertEqual(summary["cached_tokens"], 160)
        self.assertEqual(summary["completion_tokens"], 10)
        self.assertAlmostEqual(summary["cache_ratio"], 0.8)

    def test_split_summary_reports_lanes_and_slot_costs(self) -> None:
        save = {
            "wall_s": 0.5,
            "action": "save",
            "filename": "a.bin",
            "error": None,
            "response": {"n_saved": 100, "n_written": 200, "timings": {"save_ms": 480.0}},
        }
        restore = {
            "wall_s": 0.25,
            "action": "restore",
            "filename": "a.bin",
            "error": None,
            "response": {"n_restored": 100, "n_read": 200, "timings": {"restore_ms": 240.0}},
        }
        row = {
            "worker": 0,
            "round": 0,
            "messages": [],
            "index": 0,
            "filename": "a.bin",
            "prefill": reply(0, 1),
            "save": save,
            "restore": restore,
            "decode": reply(0),
        }
        summary = BENCH.summarize_split([row], 10.0, 12.0, 1.5, 1.0)
        self.assertEqual(summary["decode_summary"]["requests"], 1)
        self.assertAlmostEqual(summary["makespan_s"], 2.0)
        self.assertAlmostEqual(summary["aggregate_completion_tok_s"], 2.0)
        self.assertAlmostEqual(summary["save_summary"]["bytes_sum"], 200)
        self.assertAlmostEqual(summary["restore_summary"]["bytes_sum"], 200)
        self.assertAlmostEqual(summary["lane_overlap_s"], 0.5)

    def test_split_pipeline_runs_prefill_before_decode(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            output = Path(tmp) / "split.json"
            args = argparse.Namespace(
                mode="split",
                prefill_url="http://prefill",
                decode_url="http://decode",
                control_url=None,
                model="test",
                workers=2,
                rounds=1,
                context_words=32,
                max_tokens=4,
                timeout=1,
                reasoning_effort="none",
                session_prefix="safe",
                output=output,
            )
            order: list[str] = []

            def fake_chat(url: str, _model: str, _messages: list, max_tokens: int, *_args):
                order.append(f"{url}-{max_tokens}")
                return reply(0, max_tokens)

            def fake_slot(url: str, action: str, filename: str, _timeout: float):
                order.append(f"{url}-{action}")
                ms = 10.0 if action == "save" else 5.0
                return {
                    "wall_s": 0.01,
                    "action": action,
                    "filename": filename,
                    "error": None,
                    "response": {
                        f"n_{action}d": 10,
                        "n_written" if action == "save" else "n_read": 20,
                        "timings": {f"{action}_ms": ms},
                    },
                }

            with (
                patch.object(BENCH, "stream_chat", fake_chat),
                patch.object(BENCH, "slot_action", fake_slot),
            ):
                result = BENCH.run_split(args)
            self.assertTrue(output.is_file())
            self.assertEqual(result["summary"]["decode_summary"]["errors"], 0)
            positions = {name: index for index, name in enumerate(order)}
            self.assertLess(positions["http://prefill-1"], positions["http://prefill-save"])
            self.assertLess(positions["http://prefill-save"], positions["http://decode-restore"])
            self.assertLess(positions["http://decode-restore"], positions["http://decode-4"])


if __name__ == "__main__":
    unittest.main()
