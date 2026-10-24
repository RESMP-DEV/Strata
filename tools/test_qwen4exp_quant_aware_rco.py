#!/usr/bin/env python3
"""Host-safe tests for the layer-scope quant-aware RCO prototype."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "qwen4exp_quant_aware_rco", TOOLS / "qwen4exp_quant_aware_rco.py"
)
assert SPEC is not None and SPEC.loader is not None
RCO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RCO)

BYTES = {"2": 1_536_000, "3": 2_150_400, "4": 2_764_800}


def errors(w2: float, w3: float, w4: float) -> dict[str, list[float]]:
    return {"2": [w2], "3": [w3], "4": [w4]}


class QuantAwareRcoTest(unittest.TestCase):
    def test_packed_bytes_round_up_to_int32_words(self) -> None:
        self.assertEqual(RCO.packed_bytes(640, 2560, 4), 640 * 320 * 4)
        self.assertEqual(RCO.packed_bytes(2560, 640, 3), 2560 * 60 * 4)
        self.assertEqual(RCO.packed_bytes(8, 40, 2), 8 * 3 * 4)

    def test_candidate_bytes_add_group_scales(self) -> None:
        expected = 1280 * 320 * 4 + 1280 * 80 * 2
        self.assertEqual(RCO.candidate_bytes(1280, 2560, 4), expected)

    def test_low_budget_stays_at_the_cheapest_width(self) -> None:
        assignment = RCO.greedy_assignment(
            errors(0.01, 0.005, 0.001),
            bits_list=(4, 3, 2),
            byte_budget_total=BYTES["2"],
            per_expert_bytes=BYTES,
        )
        self.assertEqual(assignment["bits"], [2])

    def test_high_budget_upgrades_to_the_best_width(self) -> None:
        assignment = RCO.greedy_assignment(
            errors(0.01, 0.005, 0.001),
            bits_list=(4, 3, 2),
            byte_budget_total=BYTES["4"],
            per_expert_bytes=BYTES,
        )
        self.assertEqual(assignment["bits"], [4])

    def test_mid_budget_buys_the_best_upgrade(self) -> None:
        assignment = RCO.greedy_assignment(
            errors(0.01, 0.005, 0.001),
            bits_list=(4, 3, 2),
            byte_budget_total=BYTES["2"] + (BYTES["3"] - BYTES["2"]),
            per_expert_bytes=BYTES,
        )
        self.assertEqual(assignment["bits"], [3])

    def test_all_width_assignment_is_fixed_rtn(self) -> None:
        assignment = RCO.all_width_assignment(
            512,
            bits=4,
            per_expert_bytes=BYTES,
        )
        self.assertEqual(assignment["bits"], [4] * 512)
        self.assertEqual(assignment["histogram"], {"4": 512})
        self.assertEqual(assignment["total_bytes"], BYTES["4"] * 512)
        self.assertEqual(assignment["mean_bytes_per_expert"], BYTES["4"])
        with self.assertRaisesRegex(ValueError, "unsupported fixed width"):
            RCO.all_width_assignment(1, bits=5, per_expert_bytes=BYTES)


if __name__ == "__main__":
    unittest.main()
