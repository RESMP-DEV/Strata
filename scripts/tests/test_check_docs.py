#!/usr/bin/env python3
"""Tests for the documentation index contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_docs


class CheckDocsTest(unittest.TestCase):
    def test_missing_local_link_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "docs" / "reference").mkdir(parents=True)
            (root / "README.md").write_text("[missing](docs/missing.md)\n", encoding="utf-8")
            with (
                mock.patch.object(check_docs, "ROOT", root),
                mock.patch.object(check_docs, "README", root / "README.md"),
                mock.patch.object(check_docs, "INDEX", root / "docs/README.md"),
            ):
                result = check_docs._links_resolve()
        self.assertFalse(result.passed)
        self.assertIn("docs/missing.md", result.detail)

    def test_missing_markdown_anchor_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "docs"
            docs.mkdir()
            (docs / "architecture.md").write_text("## Real heading\n", encoding="utf-8")
            (root / "README.md").write_text(
                "[status](docs/architecture.md#missing)", encoding="utf-8"
            )
            with (
                mock.patch.object(check_docs, "ROOT", root),
                mock.patch.object(check_docs, "README", root / "README.md"),
                mock.patch.object(check_docs, "INDEX", docs / "README.md"),
            ):
                result = check_docs._links_resolve()
        self.assertFalse(result.passed)
        self.assertIn("#missing", result.detail)

    def test_unindexed_top_level_document_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "docs").mkdir()
            (root / "docs" / "INSTALL.md").write_text("install\n", encoding="utf-8")
            (root / "docs" / "README.md").write_text("[install](INSTALL.md)\n", encoding="utf-8")
            with (
                mock.patch.object(check_docs, "ROOT", root),
                mock.patch.object(check_docs, "INDEX", root / "docs/README.md"),
            ):
                result = check_docs._index_covers_docs()
        self.assertFalse(result.passed)
        self.assertIn("INSTALL.md", result.detail)

    def test_oversized_readme_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "[index](docs/README.md)",
                        "[status](docs/architecture.md#project-status-and-governance)",
                        "## Project status and governance",
                        "| a | b |",
                        "| --- | --- |",
                        "| one | two |",
                        "## After",
                        *(f"line {number}" for number in range(150)),
                    ]
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(check_docs, "ROOT", root),
                mock.patch.object(check_docs, "README", readme),
            ):
                result = check_docs._readme_contract()
        self.assertFalse(result.passed)
        self.assertIn("exceeds", result.detail)

    def test_third_duplicate_heading_uses_github_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "document.md"
            path.write_text("## Same\n## Same\n## Same\n", encoding="utf-8")
            self.assertIn("same-2", check_docs._heading_slugs(path))

    def test_missing_readme_status_heading_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "[index](docs/README.md)",
                        "[status](docs/architecture.md#project-status-and-governance)",
                        "| a | b |",
                        "| --- | --- |",
                    ]
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(check_docs, "ROOT", root),
                mock.patch.object(check_docs, "README", readme),
            ):
                result = check_docs._readme_contract()
        self.assertFalse(result.passed)
        self.assertIn("missing README status heading", result.detail)

    def test_parallel_status_page_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parallel = root / "docs" / "reference" / "PROJECT_STATUS.md"
            parallel.parent.mkdir(parents=True)
            parallel.write_text("parallel status\n", encoding="utf-8")
            with mock.patch.object(check_docs, "ROOT", root):
                result = check_docs._authority_contract()
        self.assertFalse(result.passed)
        self.assertIn("PROJECT_STATUS.md", result.detail)


if __name__ == "__main__":
    unittest.main()
