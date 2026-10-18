#!/usr/bin/env python3
"""Validate the repository's Lapis-style authority contracts."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "README.md",
    "docs/architecture.md",
    "evidence/project-standup.json",
    "evidence/unlink-safety-check.json",
    "justfile",
    "scripts/strata.py",
    "scripts/check_structure.py",
    "scripts/check_setup.py",
    "scripts/check_deepseek.py",
    "scripts/check_deepseek_contracts.py",
    "scripts/tests/test_check_deepseek.py",
    "scripts/tests/test_check_deepseek_contracts.py",
    "data/deepseek_v41_manifest.json",
    "scripts/tests/test_check_structure.py",
    "scripts/tests/test_check_setup.py",
)
REQUIRED_AGENT_TAGS = (
    "<purpose>",
    "<establish_context>",
    "<ownership_map>",
    "<quality_and_receipts>",
    "<project_boundaries>",
    "<coordination>",
)
REQUIRED_ARCHITECTURE_HEADINGS = (
    "## Product philosophy",
    "## Ownership boundaries",
    "## Open choices",
    "## Contracts to preserve",
    "## Work log",
)
REQUIRED_IGNORES = (".worktrees/", ".sindexer/", "build*/")
EVIDENCE_FIELDS = {
    "recorded_at",
    "agent",
    "source_revision",
    "source_dirty",
    "scope",
    "checks",
    "non_claims",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _required_files() -> CheckResult:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    return CheckResult(
        "required-authorities",
        not missing,
        "all governance and check files exist" if not missing else f"missing: {', '.join(missing)}",
    )


def _claude_symlink() -> CheckResult:
    path = ROOT / "CLAUDE.md"
    passed = path.is_symlink() and Path.readlink(path) == Path("AGENTS.md")
    return CheckResult(
        "claude-symlink",
        passed,
        "CLAUDE.md -> AGENTS.md" if passed else "CLAUDE.md must relatively link to AGENTS.md",
    )


def _agent_contract() -> CheckResult:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    missing_tags = [tag for tag in REQUIRED_AGENT_TAGS if tag not in text]
    required_terms = (
        "docs/architecture.md",
        "CONTRIBUTING.md",
        "receipt",
        "GitHub Actions",
        "127.0.0.1",
    )
    missing_terms = [term for term in required_terms if term not in text]
    problems = [f"tag {tag}" for tag in missing_tags]
    problems.extend(f"term {term}" for term in missing_terms)
    return CheckResult(
        "agent-contract",
        not problems,
        "instruction sections and durable terms are present"
        if not problems
        else f"missing {'; '.join(problems)}",
    )


def _architecture_contract() -> CheckResult:
    text = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    missing = [heading for heading in REQUIRED_ARCHITECTURE_HEADINGS if heading not in text]
    forbidden_roots = [
        path for path in ("docs/ROADMAP.md", "docs/STATUS.md") if (ROOT / path).exists()
    ]
    passed = not missing and not forbidden_roots
    problems = [f"heading {heading}" for heading in missing]
    problems.extend(f"parallel authority {path}" for path in forbidden_roots)
    return CheckResult(
        "architecture-contract",
        passed,
        "single architecture authority has stable front matter and work log"
        if passed
        else "; ".join(problems),
    )


def _ignore_contract() -> CheckResult:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    missing = [entry for entry in REQUIRED_IGNORES if entry not in lines]
    return CheckResult(
        "local-state-ignored",
        not missing,
        "worktrees, semantic-index state, and build output are ignored"
        if not missing
        else f"missing .gitignore entries: {', '.join(missing)}",
    )


def _workflow_contract() -> CheckResult:
    workflows = ROOT / ".github" / "workflows"
    return CheckResult(
        "github-actions-disabled",
        not workflows.exists(),
        "no workflow directory exists" if not workflows.exists() else "workflows are forbidden",
    )


def _evidence_schema() -> CheckResult:
    path = ROOT / "evidence" / "project-standup.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return CheckResult("durable-evidence", False, f"cannot read evidence: {error}")
    missing = sorted(EVIDENCE_FIELDS - data.keys())
    valid_revision = isinstance(data.get("source_revision"), str) and bool(
        re.fullmatch(r"[0-9a-f]{40}", data["source_revision"])
    )
    has_checks = isinstance(data.get("checks"), list) and len(data["checks"]) > 0
    has_non_claims = isinstance(data.get("non_claims"), list) and len(data["non_claims"]) > 0
    valid_time = isinstance(data.get("recorded_at"), str) and bool(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", data["recorded_at"])
    )
    passed = not missing and valid_revision and has_checks and has_non_claims and valid_time
    problems = []
    if missing:
        problems.append(f"fields {missing}")
    if not valid_revision:
        problems.append("source_revision must be 40 lowercase hex characters")
    if not has_checks:
        problems.append("checks must be a non-empty list")
    if not has_non_claims:
        problems.append("non_claims must be a non-empty list")
    if not valid_time:
        problems.append("recorded_at must be ISO-8601 UTC with a Z suffix")
    return CheckResult(
        "durable-evidence",
        passed,
        "stand-up evidence binds revision, checks, and non-claims"
        if passed
        else "; ".join(problems),
    )


def check_all() -> list[CheckResult]:
    return [
        _required_files(),
        _claude_symlink(),
        _agent_contract(),
        _architecture_contract(),
        _ignore_contract(),
        _workflow_contract(),
        _evidence_schema(),
    ]


def main() -> int:
    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))
    failed = False
    for result in check_all():
        print(f"{'PASS' if result.passed else 'FAIL'} {result.name}: {result.detail}")
        failed = failed or not result.passed
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
