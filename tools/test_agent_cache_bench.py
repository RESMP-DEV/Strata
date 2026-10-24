#!/usr/bin/env python3
"""Host-safe tests for the parallel agent-cache benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "agent_cache_bench", Path(__file__).resolve().parent / "agent_cache_bench.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class AgentCacheBenchTest(unittest.TestCase):
    def test_context_is_deterministic_and_worker_scoped(self) -> None:
        first = BENCH.long_context(3, 128)
        self.assertEqual(first, BENCH.long_context(3, 128))
        self.assertNotEqual(first, BENCH.long_context(4, 128))
        self.assertIn("Worker 3", first)

    def test_round_summary_uses_original_prompt_and_cache_counts(self) -> None:
        records = [
            {
                "error": None,
                "prompt_tokens": 1000,
                "cached_tokens": 960,
                "fresh_tokens": 40,
                "completion_tokens": 20,
                "first_token_s": 0.2,
                "prompt_ms": 10.0,
                "predicted_per_second": 50.0,
            },
            {
                "error": None,
                "prompt_tokens": 1000,
                "cached_tokens": 980,
                "fresh_tokens": 20,
                "completion_tokens": 20,
                "first_token_s": 0.4,
                "prompt_ms": 20.0,
                "predicted_per_second": 40.0,
            },
        ]
        summary = BENCH.summarize_round(2, records, 2.0)
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["prompt_tokens"], 2000)
        self.assertEqual(summary["cached_tokens"], 1940)
        self.assertEqual(summary["fresh_tokens"], 60)
        self.assertAlmostEqual(summary["cache_ratio"], 0.97)
        self.assertAlmostEqual(summary["aggregate_completion_tok_s"], 20.0)
        self.assertAlmostEqual(summary["engine_fresh_prompt_tok_s"], 2000.0)

    def test_benchmark_writes_json_and_warm_summary(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            output = Path(tmp) / "result.json"
            args = argparse.Namespace(
                url="http://127.0.0.1:1",
                model="test",
                workers=2,
                rounds=2,
                context_words=32,
                max_tokens=1,
                timeout=1,
                inter_round_delay=0,
                fixed_replies=False,
                shared_prefix=False,
                reasoning_effort="none",
                output=output,
                transcripts=None,
            )

            def fake_request(_url, _model, worker, _messages, *_args):
                return {
                    "worker": worker,
                    "wall_s": 0.1,
                    "first_token_s": 0.01,
                    "prompt_tokens": 100,
                    "cached_tokens": 90,
                    "fresh_tokens": 10,
                    "cache_ratio": 0.9,
                    "completion_tokens": 1,
                    "prompt_per_second": 100.0,
                    "predicted_per_second": 10.0,
                    "prompt_ms": 0.1,
                    "predicted_ms": 0.1,
                    "draft_n": 0,
                    "draft_n_accepted": 0,
                    "text": f"ok {worker}",
                    "error": None,
                }

            with patch.object(BENCH, "one_request", fake_request):
                result = BENCH.run_benchmark(args)
            self.assertTrue(output.is_file())
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["format"], "strata-agent-cache-bench-v1")
            self.assertEqual(loaded["warm_summary"]["requests"], 4)
            self.assertEqual(result["warm_summary"]["errors"], 0)
            self.assertAlmostEqual(result["warm_summary"]["cache_ratio"], 0.9)


if __name__ == "__main__":
    unittest.main()
