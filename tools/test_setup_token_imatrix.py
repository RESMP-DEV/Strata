#!/usr/bin/env python3
"""Tests for the opt-in llama.cpp token-file iMatrix patch."""

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


SOURCE = """#include <regex>
#include <numeric>

static void print_usage(int, char ** argv) {
__USAGE__
}

static bool compute_imatrix(llama_context * ctx, const common_params & params, const int32_t n_ctx) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
__TOKENIZATION__
    return true;
}

int main(int argc, char ** argv) {
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }
    if (params.prompt.empty()) {
        return 1;
    }
}
"""


class TokenIMatrixPatchTest(unittest.TestCase):
    def test_reference_patch_contains_the_same_helper_functions(self) -> None:
        patch = (ROOT / "tools" / "patches" / "llama-imatrix-token-file.patch").read_text(encoding="utf-8")
        functions = (ROOT / "tools" / "patches" / "llama-imatrix-token-functions.cpp").read_text(encoding="utf-8")
        for line in functions.splitlines():
            self.assertIn(line, patch)

    def write_source(self, root: Path, source: str = SOURCE) -> Path:
        llama = root / "llama.cpp"
        target = llama / "tools" / "imatrix"
        target.mkdir(parents=True)
        source = source.replace("__USAGE__", setup.LLAMA_IMATRIX_USAGE_ORIGINAL)
        source = source.replace("__TOKENIZATION__", setup.LLAMA_IMATRIX_TOKENIZATION_ORIGINAL)
        (target / "imatrix.cpp").write_text(source, encoding="utf-8")
        return llama

    def test_patch_is_absent_without_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama = self.write_source(Path(temporary))
            with mock.patch.dict("os.environ", {"STRATA_TOKEN_IMATRIX": "0"}):
                self.assertFalse(setup.apply_optional_llama_patches(llama))
            source = (llama / "tools" / "imatrix" / "imatrix.cpp").read_text(encoding="utf-8")
            self.assertNotIn("--token-file", source)

    def test_patch_is_idempotent_with_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama = self.write_source(Path(temporary))
            with (
                mock.patch.dict("os.environ", {"STRATA_TOKEN_IMATRIX": "1"}),
                mock.patch.object(setup, "ok") as say,
            ):
                self.assertTrue(setup.apply_optional_llama_patches(llama))
                self.assertFalse(setup.apply_optional_llama_patches(llama))
            source = (llama / "tools" / "imatrix" / "imatrix.cpp").read_text(encoding="utf-8")
            self.assertEqual(source.count("static std::string g_token_file;"), 1)
            self.assertIn("--token-file tokens.i32", source)
            self.assertIn("static bool read_token_file", source)
            self.assertIn("params.prompt.empty() && g_token_file.empty()", source)
            say.assert_called_once()

    def test_patch_refuses_a_moved_upstream_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama = self.write_source(Path(temporary), "int main() {}\n")
            with mock.patch.dict("os.environ", {"STRATA_TOKEN_IMATRIX": "1"}):
                with self.assertRaisesRegex(RuntimeError, "moved llama.cpp block"):
                    setup.apply_optional_llama_patches(llama)

    def test_patch_reverts_when_the_gate_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            llama = self.write_source(Path(temporary))
            original = (llama / "tools" / "imatrix" / "imatrix.cpp").read_text(encoding="utf-8")
            with (
                mock.patch.dict("os.environ", {"STRATA_TOKEN_IMATRIX": "1"}),
                mock.patch.object(setup, "ok"),
            ):
                self.assertTrue(setup.apply_optional_llama_patches(llama))
            with (
                mock.patch.dict("os.environ", {"STRATA_TOKEN_IMATRIX": "0"}),
                mock.patch.object(setup, "ok"),
            ):
                self.assertFalse(setup.apply_optional_llama_patches(llama))
            self.assertEqual(
                (llama / "tools" / "imatrix" / "imatrix.cpp").read_text(encoding="utf-8"),
                original,
            )


if __name__ == "__main__":
    unittest.main()
