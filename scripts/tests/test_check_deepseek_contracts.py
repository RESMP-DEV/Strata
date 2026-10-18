#!/usr/bin/env python3
"""Tests for the DeepSeek contract command surface."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_deepseek_contracts


class CheckDeepSeekContractsTest(unittest.TestCase):
    def test_manifest_and_test_targets_are_pinned(self) -> None:
        with (
            mock.patch.object(
                check_deepseek_contracts.shutil,
                "which",
                side_effect=["/cmake", "/ninja", "/ctest"],
            ),
        ):
            planned = check_deepseek_contracts.commands()
        self.assertIn("check_deepseek.py", " ".join(planned[0]))
        self.assertIn("data/deepseek_v41_manifest.json", " ".join(planned[0]))
        configure = " ".join(planned[1])
        self.assertIn("--fresh", configure)
        self.assertIn("src/deepseek", configure)
        build = " ".join(planned[2])
        self.assertIn("deepseek_geometry_test", build)
        self.assertIn("deepseek_candidate_select_test", build)
        self.assertIn("deepseek_geometry_test|deepseek_candidate_select_test", " ".join(planned[3]))

    def test_a_failing_command_stops_the_gate(self) -> None:
        with mock.patch.object(
            check_deepseek_contracts.shutil,
            "which",
            side_effect=["/cmake", "/ninja", "/ctest"],
        ):
            check_deepseek_contracts.commands()
        runs: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: object) -> object:
            runs.append(command)
            return mock.Mock(returncode=1)

        with (
            mock.patch.object(check_deepseek_contracts.subprocess, "run", side_effect=fake_run),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(check_deepseek_contracts.main(), 1)
        self.assertEqual(len(runs), 1)

    def test_missing_native_tool_is_a_failure(self) -> None:
        with (
            mock.patch.object(check_deepseek_contracts.shutil, "which", return_value=None),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(check_deepseek_contracts.main(), 1)


if __name__ == "__main__":
    unittest.main()
