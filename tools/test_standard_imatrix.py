#!/usr/bin/env python3
"""Host-safe tests for the pinned standard iMatrix token path."""

from __future__ import annotations

import importlib.util
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("standard_imatrix", ROOT / "tools" / "standard_imatrix.py")
SI = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = SI
SPEC.loader.exec_module(SI)


class StandardIMatrixTest(unittest.TestCase):
    def test_npy_export_is_little_endian_i32(self) -> None:
        values = [0, 1, 0x7FFFFFFF, 248046]
        header = (
            b"\x93NUMPY\x01\x00"
            + (76).to_bytes(2, "little")
            + b"{'descr': '<i8', 'fortran_order': False, 'shape': (256, 2048), }".ljust(76)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npy = root / "tokens.npy"
            # Patch the expected shape count only for this synthetic fixture.
            npy.write_bytes(header + b"".join(value.to_bytes(8, "little", signed=True) for value in values * 131072))
            output = root / "tokens.i32"
            SI.EXPECTED_SHAPE = (256, 2048)
            SI.TOKEN_COUNT = len(values) * 131072
            try:
                digest = SI.export_tokens(npy, output)
            finally:
                SI.EXPECTED_SHAPE = (256, 2048)
                SI.TOKEN_COUNT = 256 * 2048
            expected = b"".join(value.to_bytes(4, "little") for value in values * 131072)
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(len(digest), 64)

    def test_prepare_verifies_and_exports_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            token_path = root / SI.NPY_PATH
            token_path.parent.mkdir(parents=True)
            token_path.write_bytes(b"fake")
            token_file = root / SI.TOKEN_FILE_NAME
            token_file.write_bytes(b"0" * 4)
            tokenizer_json = root / "qwen3.8-27b-tokenizer" / "tokenizer.json"
            tokenizer_json.parent.mkdir(parents=True)
            tokenizer_json.write_text("{}")
            fake_hash = "a" * 64
            fake_max = 3
            command = subprocess.CompletedProcess(["hf"], 0)
            with (
                mock.patch.object(SI.subprocess, "run", return_value=command) as run,
                mock.patch.object(SI, "verify"),
                mock.patch.object(SI, "sha256", return_value=fake_hash),
                mock.patch.object(SI, "max_token_id", return_value=fake_max),
                mock.patch.object(SI, "export_tokens", return_value=fake_hash),
            ):
                result = SI.prepare(root)
            self.assertEqual(result.token_file, token_file)
            self.assertEqual(result.token_sha256, fake_hash)
            self.assertEqual(result.max_token_id, fake_max)
            self.assertEqual(run.call_count, 2)
            self.assertIn("--revision", run.call_args_list[0].args[0])
            self.assertIn(SI.DATASET_REVISION, run.call_args_list[0].args[0])
            self.assertIn(SI.TOKENIZER_REVISION, run.call_args_list[1].args[0])

    def test_command_uses_exact_tokens_and_checkpoints(self) -> None:
        command = SI.command_for(
            Path("llama-imatrix"),
            Path("model.gguf"),
            Path("out.gguf"),
            Path("tokens.i32"),
            extra_arguments=["-t", "24"],
        )
        self.assertIn("--token-file", command)
        self.assertIn("-c", command)
        self.assertEqual(command[command.index("-c") + 1], "2048")
        self.assertEqual(command[command.index("--chunks") + 1], "256")
        self.assertIn("--save-frequency", command)
        self.assertEqual(command[-2:], ["-t", "24"])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            SI.command_for(Path("x"), Path("m"), Path("o"), Path("t"), chunks=257)

    def test_resume_loads_recorded_chunk_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "standard.imatrix.gguf"
            with mock.patch.object(SI, "imatrix_chunk_count", return_value=40):
                command = SI.command_for(
                    Path("llama-imatrix"),
                    Path("model.gguf"),
                    Path("out.gguf"),
                    Path("tokens.i32"),
                    resume_output=checkpoint,
                )
            self.assertEqual(command[command.index("--in-file") + 1], str(checkpoint))
            self.assertEqual(command[command.index("--chunk") + 1], "40")


if __name__ == "__main__":
    unittest.main()
