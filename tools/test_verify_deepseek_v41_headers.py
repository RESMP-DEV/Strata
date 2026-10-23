#!/usr/bin/env python3
"""Host-safe tests for the DeepSeek safetensors header verifier."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class VerifyDeepseekV41HeadersTest(unittest.TestCase):
    def test_missing_indexed_shard_is_recorded_as_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            index_path = root / "index.json"
            output_path = root / "result.json"
            index_path.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 4},
                        "weight_map": {"a": "model-00001-of-00002.safetensors"},
                    }
                ),
                encoding="utf-8",
            )

            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("verify_deepseek_v41_headers.py")),
                    str(checkpoint),
                    str(index_path),
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            result = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(process.returncode, 1)
            self.assertEqual(result["checkpoint"], "checkpoint")
            self.assertEqual(result["failure_count"], 1)
            self.assertEqual(result["file_size_failures"], 1)
            self.assertTrue(
                result["failures"][0].startswith(
                    "model-00001-of-00002.safetensors: cannot stat shard:"
                )
            )


if __name__ == "__main__":
    unittest.main()
