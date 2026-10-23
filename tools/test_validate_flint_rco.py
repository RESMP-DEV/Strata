#!/usr/bin/env python3
"""Host-safe tests for the tiny Flint/RCO integration probe."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False

if torch_available:
    SPEC = importlib.util.spec_from_file_location(
        "validate_flint_rco", TOOLS / "validate_flint_rco.py"
    )
    MODULE = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    sys.modules[SPEC.name] = MODULE
    SPEC.loader.exec_module(MODULE)


@unittest.skipIf(not torch_available, "torch is not installed")
class ValidateFlintRcoTest(unittest.TestCase):
    def test_tiny_model_matches_adapter_geometry(self) -> None:
        model = MODULE.TinyFlintLM()
        self.assertEqual(MODULE.adapter.get_moe_info(model), (3, 4, 2))
        output = model(torch.tensor([[1, 2, 3]]))
        self.assertEqual(tuple(output.logits.shape), (1, 3, 32))


if __name__ == "__main__":
    unittest.main()
