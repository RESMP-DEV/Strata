#!/usr/bin/env python3
"""Tests for the selected setup-regression domain."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_setup


class CheckSetupTest(unittest.TestCase):
    def test_selected_modules_exist_and_are_unique(self) -> None:
        self.assertEqual(
            len(check_setup.SELECTED_SETUP_TESTS),
            len(set(check_setup.SELECTED_SETUP_TESTS)),
        )
        for module in check_setup.SELECTED_SETUP_TESTS:
            self.assertTrue((Path(__file__).parents[2] / "tools" / module).is_file(), module)

    def test_each_module_is_a_separate_bounded_unittest_process(self) -> None:
        calls: list[subprocess.CompletedProcess[str]] = []

        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            call = subprocess.CompletedProcess(command, 0)
            calls.append(call)
            return call

        with mock.patch.object(check_setup.subprocess, "run", side_effect=fake_run):
            results = check_setup.run_setup_modules()
        self.assertEqual([code for _, code in results], [0] * len(check_setup.SELECTED_SETUP_TESTS))
        self.assertEqual(len(calls), len(check_setup.SELECTED_SETUP_TESTS))
        self.assertTrue(all("-q" in call.args for call in calls))


class CheckSetupFailureTest(unittest.TestCase):
    def test_a_failing_module_is_reported_and_fails_the_domain(self) -> None:
        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            module = command[-2]
            code = 1 if module == check_setup.SELECTED_SETUP_TESTS[-1] else 0
            return subprocess.CompletedProcess(command, code)

        with mock.patch.object(check_setup.subprocess, "run", side_effect=fake_run):
            results = check_setup.run_setup_modules()
        self.assertEqual(results[-1], (check_setup.SELECTED_SETUP_TESTS[-1], 1))
        with (
            mock.patch.object(check_setup, "run_setup_modules", return_value=results),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(check_setup.main(), 1)


if __name__ == "__main__":
    unittest.main()
