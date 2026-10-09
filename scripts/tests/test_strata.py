#!/usr/bin/env python3
"""Tests for Strata's receipt-writing dispatcher."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import strata


class ReceiptSafetyTest(unittest.TestCase):
    def test_write_receipt_replaces_a_symlinked_final_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "outside-receipt.json"
            target.write_text("do not touch", encoding="utf-8")
            report = root / "receipt.json"
            report.symlink_to(target)
            with mock.patch.object(strata, "REPORT", report):
                strata.write_receipt(
                    passed=True,
                    checks=[],
                    state={"source_revision": "a" * 40, "source_dirty": True},
                )
            self.assertFalse(report.is_symlink())
            self.assertEqual(json.loads(report.read_text())["aggregate_passed"], True)
            self.assertEqual(target.read_text(), "do not touch")

    def test_quality_removes_a_stale_pass_before_starting(self) -> None:
        events: list[str] = []
        state = {"source_revision": "a" * 40, "source_dirty": True}
        write_receipt = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "receipt.json"
            report.write_text("stale pass", encoding="utf-8")

            def fail_unlink(path: Path, missing_ok: bool = False) -> None:
                events.append("unlink")
                raise RuntimeError("startup interrupted")

            with (
                mock.patch.object(strata, "REPORT", report),
                mock.patch.object(strata.Path, "unlink", fail_unlink),
                mock.patch.object(strata, "git_state", return_value=state),
                mock.patch.object(strata, "write_receipt", write_receipt),
                self.assertRaisesRegex(RuntimeError, "startup interrupted"),
            ):
                strata.quality()
        self.assertEqual(events, ["unlink"])
        self.assertFalse(report.exists())
        write_receipt.assert_called_once_with(
            passed=False,
            checks=[],
            state={},
            startup_failure="RuntimeError: startup interrupted",
        )


if __name__ == "__main__":
    unittest.main()
