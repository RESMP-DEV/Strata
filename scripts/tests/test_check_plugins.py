#!/usr/bin/env python3
"""Tests for the accelerator-plugin check domain."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_plugins


class CheckPluginsTest(unittest.TestCase):
    def test_commands_are_argv_lists_and_stop_on_first_failure(self) -> None:
        results = [
            subprocess.CompletedProcess(["first"], 0),
            subprocess.CompletedProcess(["second"], 2),
            subprocess.CompletedProcess(["third"], 0),
        ]
        with (
            mock.patch.object(
                check_plugins.shutil, "which", side_effect=lambda name: f"/opt/{name}"
            ),
            mock.patch.object(check_plugins, "run", side_effect=results) as runs,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(check_plugins.main(), 2)
        self.assertEqual(runs.call_count, 2)
        for call in runs.call_args_list:
            self.assertIsInstance(call.args[0], list)

    def test_missing_cmake_is_an_explicit_failure(self) -> None:
        with (
            mock.patch.object(check_plugins.shutil, "which", return_value=None),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(check_plugins.main(), 1)


if __name__ == "__main__":
    unittest.main()
