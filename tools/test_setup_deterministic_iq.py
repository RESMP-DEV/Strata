"""Tests for the opt-in deterministic llama.cpp IQ2_XS patch."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("strata_setup", ROOT / "setup.py")
setup = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(setup)


class DeterministicIQPatchTest(unittest.TestCase):
    def write_source(self, root: Path, source: str) -> Path:
        llama = root / "llama.cpp"
        target = llama / "ggml" / "src" / "ggml-quants.c"
        target.parent.mkdir(parents=True)
        target.write_text(
            "static void quantize_row_iq2_xs_impl() {\n"
            + source
            + "\nsize_t quantize_iq2_xs_end_marker() {}\n",
            encoding="utf-8",
        )
        return llama

    def test_patch_is_absent_without_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama = self.write_source(Path(temporary), setup.LLAMA_IQ2_XS_WEIGHT_ORIGINAL)
            with mock.patch.dict("os.environ", {"STRATA_DETERMINISTIC_IQ": "0"}):
                self.assertFalse(setup.apply_optional_llama_patches(llama))
            source = (llama / "ggml" / "src" / "ggml-quants.c").read_text(encoding="utf-8")
            self.assertIn(setup.LLAMA_IQ2_XS_WEIGHT_ORIGINAL, source)
            self.assertNotIn(setup.LLAMA_IQ2_XS_WEIGHT_DETERMINISTIC, source)

    def test_patch_is_idempotent_with_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama = self.write_source(Path(temporary), setup.LLAMA_IQ2_XS_WEIGHT_ORIGINAL)
            with (
                mock.patch.dict("os.environ", {"STRATA_DETERMINISTIC_IQ": "1"}),
                mock.patch.object(setup, "ok") as say,
            ):
                self.assertTrue(setup.apply_optional_llama_patches(llama))
                self.assertFalse(setup.apply_optional_llama_patches(llama))
            patched = (llama / "ggml" / "src" / "ggml-quants.c").read_text(encoding="utf-8")
            self.assertIn(setup.LLAMA_IQ2_XS_WEIGHT_DETERMINISTIC, patched)
            self.assertNotIn(setup.LLAMA_IQ2_XS_WEIGHT_ORIGINAL, patched)
            say.assert_called_once()

    def test_patch_refuses_a_moved_upstream_expression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama = self.write_source(Path(temporary), "static void moved(void) {}\n")
            with mock.patch.dict("os.environ", {"STRATA_DETERMINISTIC_IQ": "1"}):
                with self.assertRaisesRegex(RuntimeError, "found 0"):
                    setup.apply_optional_llama_patches(llama)


if __name__ == "__main__":
    unittest.main()
