#!/usr/bin/env python3
"""Host-safe tests for exact-COPY recipe generation."""

from __future__ import annotations

import importlib.util
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exact_copy_recipe", ROOT / "tools" / "exact_copy_recipe.py"
)
EC = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = EC
SPEC.loader.exec_module(EC)


def write_minimal_f32_gguf(path: Path, names: list[str]) -> None:
    """Write the tiny F32 GGUF needed by this test without optional dependencies."""

    alignment = 32
    body = bytearray()
    body += struct.pack("<IIQQ", 0x46554747, 3, len(names), 0)
    offsets = []
    offset = 0
    for name in names:
        encoded = name.encode("utf-8")
        body += struct.pack("<Q", len(encoded)) + encoded
        body += struct.pack("<IQQIQ", 2, 2, 2, 0, offset)
        offsets.append(offset)
        offset = (offset + 16 + alignment - 1) // alignment * alignment
    body += b"\0" * ((-len(body)) % alignment)
    for _ in offsets:
        body += b"\0" * 16
        body += b"\0" * ((-len(body)) % alignment)
    path.write_bytes(body)


class ExactCopyRecipeTest(unittest.TestCase):
    def test_anchored_recipe_preserves_and_targets_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gguf = root / "model.gguf"
            write_minimal_f32_gguf(
                gguf,
                [
                    "blk.0.ssm_a",
                    "blk.0.ssm_alpha.weight",
                    "blk.0.ffn_gate_exps.weight",
                    "blk.0.ffn_up_exps.weight",
                    "blk.0.ffn_down_exps.weight",
                ],
            )
            recipe = root / "recipe.txt"
            summary = EC.write_recipe(gguf, recipe)
            lines = recipe.read_text(encoding="utf-8").splitlines()
            self.assertEqual(summary.rows, 5)
            self.assertEqual(summary.gate_up_rows, 2)
            self.assertEqual(summary.down_rows, 1)
            self.assertEqual(summary.preserved_rows, 2)
            self.assertIn(r"^blk\.0\.ssm_alpha\.weight$=F32", lines)
            self.assertIn(r"^blk\.0\.ffn_gate_exps\.weight$=IQ2_XS", lines)
            self.assertIn(r"^blk\.0\.ffn_down_exps\.weight$=Q2_0", lines)
            for line in lines:
                pattern, target = line.split("=", 1)
                self.assertTrue(pattern.startswith("^") and pattern.endswith("$"))
            ssm_alpha_pattern = r"^blk\.0\.ssm_alpha\.weight$"
            ssm_a_pattern = next(
                line.split("=", 1)[0] for line in lines if line.endswith("=F32") and "ssm_a" in line
            )
            self.assertIsNotNone(re.search(ssm_alpha_pattern, "blk.0.ssm_alpha.weight"))
            self.assertIsNone(re.search(ssm_a_pattern, "blk.0.ssm_alpha.weight"))


if __name__ == "__main__":
    unittest.main()
