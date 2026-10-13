"""Tests for the opt-in exact-COPY llama.cpp research patch."""

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


class ExactCopyPatchTest(unittest.TestCase):
    def write_source(self, root: Path, source: str) -> Path:
        llama = root / "llama.cpp"
        target = llama / "src" / "llama-quant.cpp"
        target.parent.mkdir(parents=True)
        target.write_text(source, encoding="utf-8")
        return llama

    def source(self, resolver: str | None = None, preflight: str | None = None) -> str:
        return (
            (resolver or setup.LLAMA_EXACT_COPY_RESOLVER_ORIGINAL)
            + "\nvoid between_blocks() {}\n"
            + (preflight or setup.LLAMA_EXACT_COPY_PREFLIGHT_ORIGINAL)
        )

    def test_patch_is_absent_without_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama = self.write_source(Path(temporary), self.source())
            with mock.patch.dict("os.environ", {"STRATA_EXACT_COPY": "0"}):
                self.assertFalse(setup.apply_optional_llama_patches(llama))
            source = (llama / "src" / "llama-quant.cpp").read_text(encoding="utf-8")
            self.assertNotIn(setup.LLAMA_EXACT_COPY_MARKER, source)

    def test_patch_is_idempotent_and_reverts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama = self.write_source(Path(temporary), self.source())
            with (
                mock.patch.dict("os.environ", {"STRATA_EXACT_COPY": "1"}),
                mock.patch.object(setup, "ok") as say,
            ):
                self.assertTrue(setup.apply_optional_llama_patches(llama))
                self.assertFalse(setup.apply_optional_llama_patches(llama))
            source = (llama / "src" / "llama-quant.cpp").read_text(encoding="utf-8")
            self.assertIn(setup.LLAMA_EXACT_COPY_RESOLVER_PATCHED, source)
            self.assertIn(setup.LLAMA_EXACT_COPY_PREFLIGHT_PATCHED, source)
            say.assert_called_once()
            with (
                mock.patch.dict("os.environ", {"STRATA_EXACT_COPY": "0"}),
                mock.patch.object(setup, "ok"),
            ):
                self.assertFalse(setup.apply_optional_llama_patches(llama))
            source = (llama / "src" / "llama-quant.cpp").read_text(encoding="utf-8")
            self.assertIn(setup.LLAMA_EXACT_COPY_RESOLVER_ORIGINAL, source)
            self.assertIn(setup.LLAMA_EXACT_COPY_PREFLIGHT_ORIGINAL, source)

    def test_patch_refuses_a_moved_upstream_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama = self.write_source(Path(temporary), "static void moved(void) {}\n")
            with (
                mock.patch.dict("os.environ", {"STRATA_EXACT_COPY": "1"}),
                self.assertRaisesRegex(RuntimeError, "found 0"),
            ):
                setup.apply_exact_copy_patch(llama)


if __name__ == "__main__":
    unittest.main()
