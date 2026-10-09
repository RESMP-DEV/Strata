#!/usr/bin/env python3
"""Tests for repository structure checks."""

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

import check_structure


class CheckStructureTest(unittest.TestCase):
    def test_required_authorities_are_unique(self) -> None:
        self.assertEqual(
            len(check_structure.REQUIRED_FILES), len(set(check_structure.REQUIRED_FILES))
        )

    def test_rejects_non_relative_claude_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "CLAUDE.md").symlink_to(root / "AGENTS.md")
            with mock.patch.object(check_structure, "ROOT", root):
                result = check_structure._claude_symlink()
        self.assertFalse(result.passed)
        self.assertIn("relative", result.detail)

    def test_rejects_evidence_without_a_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "evidence").mkdir()
            (root / "evidence" / "project-standup.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(check_structure, "ROOT", root):
                result = check_structure._evidence_schema()
        self.assertFalse(result.passed)
        self.assertIn("fields", result.detail)


class AgentContractTest(unittest.TestCase):
    def test_a_document_without_the_contract_tags_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "AGENTS.md").write_text("nothing here", encoding="utf-8")
            with mock.patch.object(check_structure, "ROOT", root):
                result = check_structure._agent_contract()
        self.assertFalse(result.passed)

    def test_evidence_rejects_a_missing_non_claim_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "evidence").mkdir()
            (root / "evidence" / "project-standup.json").write_text(
                json.dumps(
                    {
                        "recorded_at": "2026-10-09T00:00:00Z",
                        "agent": "Codex",
                        "source_revision": "a" * 40,
                        "source_dirty": True,
                        "scope": "test",
                        "checks": [{"command": "true", "exit": 0}],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(check_structure, "ROOT", root):
                result = check_structure._evidence_schema()
        self.assertFalse(result.passed)
        self.assertIn("non_claims", result.detail)


class ArchitectureContractTest(unittest.TestCase):
    def test_a_parallel_roadmap_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "docs"
            docs.mkdir()
            (docs / "architecture.md").write_text(
                "\n".join(heading for heading in check_structure.REQUIRED_ARCHITECTURE_HEADINGS),
                encoding="utf-8",
            )
            (docs / "ROADMAP.md").write_text("parallel", encoding="utf-8")
            with mock.patch.object(check_structure, "ROOT", root):
                result = check_structure._architecture_contract()
        self.assertFalse(result.passed)
        self.assertIn("ROADMAP", result.detail)


class IgnoreAndWorkflowTest(unittest.TestCase):
    def test_missing_ignore_entries_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".gitignore").write_text("*.log\n", encoding="utf-8")
            with mock.patch.object(check_structure, "ROOT", root):
                result = check_structure._ignore_contract()
        self.assertFalse(result.passed)

    def test_a_workflow_directory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".github" / "workflows").mkdir(parents=True)
            with mock.patch.object(check_structure, "ROOT", root):
                result = check_structure._workflow_contract()
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
