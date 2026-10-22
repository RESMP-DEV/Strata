#!/usr/bin/env python3
"""Tests for deterministic RCO token export."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("build_rco_tokens", TOOLS / "build_rco_tokens.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    vocab_size = 10
    eos_token_id = 9
    alphabet = "abcdefgh"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, Any]:
        assert add_special_tokens is False
        return {"input_ids": [self.alphabet.index(character) for character in text]}


class BuildRcoTokensTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_rows(self, name: str, rows: list[dict[str, object]]) -> Path:
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_auto_prefers_private_text_and_chunks_deterministically(self) -> None:
        first = self.write_rows(
            "private.jsonl",
            [
                {"text": "abc", "passage": "wrong"},
                {"text": "de"},
            ],
        )
        second = self.write_rows("public.jsonl", [{"passage": "fgh"}])
        result = MODULE.build_sequences(
            [first, second],
            FakeTokenizer(),
            sequence_length=5,
            max_records=None,
            max_sequences=None,
            field="auto",
        )
        self.assertEqual(result.sequences, [[0, 1, 2, 9, 3], [4, 9, 5, 6, 7], [9, 9, 9, 9, 9]])
        self.assertEqual(result.records_read, 3)
        self.assertEqual(result.records_used, 3)
        self.assertEqual(result.full_sequences, 2)
        self.assertEqual(result.padded_sequences, 1)
        self.assertEqual(result.padding_tokens, 4)
        self.assertEqual(
            [(source.rows, source.used_rows) for source in result.sources],
            [(2, 2), (1, 1)],
        )

    def test_limits_stop_before_more_input_is_read(self) -> None:
        path = self.write_rows("rows.jsonl", [{"text": "abc"}, {"text": "de"}])
        result = MODULE.build_sequences(
            [path],
            FakeTokenizer(),
            sequence_length=4,
            max_records=1,
            max_sequences=None,
            field="text",
        )
        self.assertEqual(result.records_read, 1)
        self.assertEqual(result.sequences, [[0, 1, 2, 9]])

    def test_non_positive_limits_are_rejected(self) -> None:
        path = self.write_rows("rows.jsonl", [{"text": "abc"}])
        for keyword in ("max_records", "max_sequences"):
            with self.subTest(keyword=keyword):
                limits = {"max_records": None, "max_sequences": None}
                limits[keyword] = 0
                with self.assertRaisesRegex(ValueError, f"--{keyword.replace('_', '-')}"):
                    MODULE.build_sequences(
                        [path],
                        FakeTokenizer(),
                        sequence_length=4,
                        **limits,
                        field="text",
                    )

    def test_cli_limits_and_remote_code_opt_in_are_validated(self) -> None:
        arguments = [
            "build",
            "rows.jsonl",
            "--tokenizer",
            "tokenizer",
            "--out",
            "out.pt",
            "--dataset-name",
            "test",
        ]
        with patch("sys.argv", arguments):
            args = MODULE.parse_args()
        self.assertIsNone(args.max_records)
        self.assertIsNone(args.max_sequences)
        self.assertFalse(args.trust_remote_code)

        with (
            patch("sys.argv", [*arguments, "--max-records", "0"]),
            self.assertRaises(SystemExit),
        ):
            MODULE.parse_args()
        with (
            patch("sys.argv", [*arguments, "--max-sequences", "-1"]),
            self.assertRaises(SystemExit),
        ):
            MODULE.parse_args()
        with patch("sys.argv", [*arguments, "--trust-remote-code"]):
            self.assertTrue(MODULE.parse_args().trust_remote_code)

    def test_padding_is_reported_without_inventing_source_tokens(self) -> None:
        path = self.write_rows("rows.jsonl", [{"text": "ab"}])
        result = MODULE.build_sequences(
            [path],
            FakeTokenizer(),
            sequence_length=5,
            max_records=None,
            max_sequences=None,
            field="text",
        )
        self.assertEqual(result.sequences, [[0, 1, 9, 9, 9]])
        self.assertEqual(result.full_sequences, 0)
        self.assertEqual(result.padded_sequences, 1)
        self.assertEqual(result.padding_tokens, 2)
        self.assertEqual(result.non_padding_tokens, 3)

    def test_exact_sequence_limit_discards_an_unrequested_partial_tail(self) -> None:
        path = self.write_rows("rows.jsonl", [{"text": "abcdefgh"}])
        result = MODULE.build_sequences(
            [path],
            FakeTokenizer(),
            sequence_length=4,
            max_records=None,
            max_sequences=2,
            field="text",
        )
        self.assertEqual(result.sequences, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(result.full_sequences, 2)
        self.assertEqual(result.padded_sequences, 0)
        self.assertEqual(result.padding_tokens, 0)

    def test_invalid_token_is_rejected_against_tokenizer_vocabulary(self) -> None:
        class OutOfRange(FakeTokenizer):
            def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, Any]:
                return {"input_ids": [10]}

        path = self.write_rows("bad.jsonl", [{"text": "x"}])
        with self.assertRaisesRegex(ValueError, "token id outside"):
            MODULE.build_sequences(
                [path],
                OutOfRange(),
                sequence_length=2,
                max_records=None,
                max_sequences=None,
                field="text",
            )

    def test_non_object_jsonl_is_rejected(self) -> None:
        path = self.root / "bad.jsonl"
        path.write_text("[1]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected a JSON object"):
            MODULE.build_sequences(
                [path],
                FakeTokenizer(),
                sequence_length=2,
                max_records=None,
                max_sequences=None,
                field="text",
            )


if __name__ == "__main__":
    unittest.main()
