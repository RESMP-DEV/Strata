#!/usr/bin/env python3
"""Host-safe contract tests for the IQ3 QDQ bridge wrapper."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("iq3_qdq", TOOLS / "iq3_qdq.py")
assert SPEC is not None and SPEC.loader is not None
IQ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IQ)


class Iq3QdqTest(unittest.TestCase):
    def test_kind_constants_match_the_c_bridge(self) -> None:
        self.assertEqual((IQ.IQ3_S, IQ.IQ3_XXS, IQ.IQ4_NL, IQ.IQ4_XS), (0, 1, 2, 3))

    def test_missing_library_is_reported(self) -> None:
        with self.assertRaisesRegex(IQ.Iq3Error, "not found"):
            IQ.load_library(Path("/nonexistent/libiq_qdq.so"))


if __name__ == "__main__":
    unittest.main()
