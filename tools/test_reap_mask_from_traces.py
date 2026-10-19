#!/usr/bin/env python3
"""Tests for the trace-derived REAP pruning control."""

from __future__ import annotations

import importlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "reap_mask_from_traces", TOOLS / "reap_mask_from_traces.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def trace(path: Path, rows: list[tuple[int, list[int], list[float]]]) -> None:
    with path.open("wb") as handle:
        for layer, ids, weights in rows:
            handle.write(struct.pack("<ii", layer, len(ids)))
            handle.write(struct.pack(f"<{len(ids)}i", *ids))
            handle.write(struct.pack(f"<{len(ids)}f", *weights))


class ReapMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_weighted_mask_exact_and_deterministic(self) -> None:
        path = self.root / "trace.bin"
        trace(path, [
            (0, [1, 2, 3], [0.1, 0.5, 0.5]),
            (1, [4, 5], [0.2, 0.1]),
        ])
        scores, records = MODULE.read_trace_scores([path], n_layers=2, n_expert=6, weighted=True)
        keep, fills, selected = MODULE.choose_mask(scores, 2, 6, 2)
        self.assertEqual(records, 2)
        self.assertEqual(keep, [[2, 3], [4, 5]])
        self.assertEqual(fills, [[], []])
        self.assertEqual(selected[0], [0.5, 0.5])
        self.assertAlmostEqual(selected[1][0], 0.2, places=6)
        self.assertAlmostEqual(selected[1][1], 0.1, places=6)

    def test_zero_serve_weights_count_as_unit_activations(self) -> None:
        path = self.root / "serve.bin"
        trace(path, [(0, [1, 1, 2], [0.0, 0.0, 0.0])])
        scores, records = MODULE.read_trace_scores([path], n_layers=1, n_expert=4, weighted=True)
        self.assertEqual(records, 1)
        self.assertEqual(scores[(0, 1)], 2.0)
        self.assertEqual(scores[(0, 2)], 1.0)

    def test_sparse_layers_are_filled_to_exact_budget(self) -> None:
        path = self.root / "sparse.bin"
        trace(path, [(0, [3], [0.1]), (1, [5], [0.1])])
        scores, _ = MODULE.read_trace_scores([path], n_layers=2, n_expert=6, weighted=True)
        keep, fills, _ = MODULE.choose_mask(scores, 2, 6, 3)
        self.assertEqual(keep[0], [3, 0, 1])
        self.assertEqual(keep[1], [5, 0, 1])
        self.assertEqual(fills, [[0, 1], [0, 1]])

    def test_profile_round_trip_and_cli(self) -> None:
        path = self.root / "trace.bin"
        rows = []
        for layer in range(48):
            ids = [1, 2] if layer == 0 else [3, 4]
            weights = [0.1, 0.4] if layer == 0 else [0.3, 0.1]
            rows.append((layer, ids, weights))
        trace(path, rows)
        out = self.root / "mask.json"
        profile = self.root / "profile.bin"
        script = TOOLS / "reap_mask_from_traces.py"
        result = subprocess_run(
            [
                sys.executable,
                str(script),
                str(path),
                "--out",
                str(out),
                "--profile",
                str(profile),
                "--n-expert",
                "6",
                "--keep",
                "2",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        mask = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(mask["keep_ids"][0], [2, 1])
        self.assertEqual(mask["keep_ids"][1], [3, 4])
        ranked = importlib_start_profile(profile)
        self.assertEqual(len(ranked), 96)
        self.assertIn((0, 2), ranked)
        self.assertIn((1, 3), ranked)

    def test_truncated_trace_is_rejected(self) -> None:
        path = self.root / "bad.bin"
        path.write_bytes(struct.pack("<ii", 0, 2) + struct.pack("<2i", 1, 2))
        with self.assertRaisesRegex(ValueError, "invalid routing count"):
            MODULE.read_trace_scores([path], n_layers=1, n_expert=4, weighted=True)


def subprocess_run(argv: list[str]):
    import subprocess

    return subprocess.run(argv, capture_output=True, text=True, check=False)


def importlib_start_profile(path: Path) -> list[tuple[int, int]]:
    # make_profile is already importable from the same tools directory.
    return importlib.import_module("make_profile").read_profile(path, 6)


if __name__ == "__main__":
    unittest.main()
