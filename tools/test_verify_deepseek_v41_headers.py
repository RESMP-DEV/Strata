#!/usr/bin/env python3
"""Host-safe tests for the DeepSeek safetensors header verifier."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class VerifyDeepseekV41HeadersTest(unittest.TestCase):
    def write_safetensors(self, path: Path, entries: dict[str, object]) -> None:
        header = json.dumps(entries, separators=(",", ":")).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"0" * 8)

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

    def test_every_interval_gap_and_overlap_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            shard = checkpoint / "model-00001-of-00001.safetensors"
            self.write_safetensors(
                shard,
                {
                    "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                    "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
                    "c": {"dtype": "U8", "shape": [2], "data_offsets": [4, 6]},
                },
            )
            index_path = root / "index.json"
            output_path = root / "result.json"
            index_path.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 6},
                        "weight_map": {name: shard.name for name in "abc"},
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
            self.assertEqual(result["interval_failures"], 2)
            self.assertIn("b: interval gap/overlap at 1", result["failures"][0])
            self.assertIn("c: interval gap/overlap at 4", result["failures"][1])


if __name__ == "__main__":
    unittest.main()
