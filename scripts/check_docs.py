#!/usr/bin/env python3
"""Validate the README reference split and documentation index."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
INDEX = ROOT / "docs" / "README.md"
MAX_README_LINES = 140
FORBIDDEN_PARALLEL_AUTHORITIES = (
    "docs/ROADMAP.md",
    "docs/STATUS.md",
    "docs/reference/PROJECT_STATUS.md",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _local_targets(path: Path) -> list[str]:
    return [target for target, _fragment in _local_references(path)]


def _local_references(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    references = []
    for value in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target, separator, fragment = value.partition("#")
        references.append((target if separator else target, fragment if separator else ""))
    references.extend(
        (target, "")
        for target in re.findall(r"<(?:a|img)\b[^>]*?(?:href|src)=\"([^\"#]+)(?:#[^\"]*)?\"", text)
    )
    return [
        reference
        for reference in references
        if reference[0] and not re.match(r"^[a-z][a-z0-9+.-]*:", reference[0])
    ]


def _heading_slugs(path: Path) -> set[str]:
    slugs: list[str] = []
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", path.read_text(encoding="utf-8"), re.M):
        slug = re.sub(r"[^\w\s-]", "", heading.lower(), flags=re.UNICODE)
        slug = re.sub(r"\s", "-", slug)
        if slugs.count(slug):
            slug = f"{slug}-{slugs.count(slug)}"
        slugs.append(slug)
    return set(slugs)


def _links_resolve() -> CheckResult:
    missing: list[str] = []
    for path in (README, INDEX):
        if not path.is_file():
            missing.append(str(path.relative_to(ROOT)))
            continue
        for target, fragment in _local_references(path):
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")
            elif fragment and resolved.suffix == ".md" and fragment not in _heading_slugs(resolved):
                missing.append(f"{path.relative_to(ROOT)} -> {target}#{fragment}")
    return CheckResult(
        "docs-links",
        not missing,
        "README and index links resolve" if not missing else f"missing: {', '.join(missing)}",
    )


def _index_covers_docs() -> CheckResult:
    if not INDEX.is_file():
        return CheckResult("docs-index-coverage", False, "docs/README.md is absent")
    docs_root = ROOT / "docs"
    expected = {
        path.relative_to(docs_root).as_posix() for path in docs_root.rglob("*.md") if path != INDEX
    }
    indexed: list[str] = []
    for target in _local_targets(INDEX):
        resolved = (INDEX.parent / target).resolve()
        try:
            relative = resolved.relative_to(docs_root).as_posix()
        except ValueError:
            continue
        if resolved.suffix == ".md":
            indexed.append(relative)
    duplicates = sorted(name for name in set(indexed) if indexed.count(name) != 1)
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)
    if duplicates:
        missing.extend(f"{name} (duplicate)" for name in duplicates)
    missing.extend(extra)
    return CheckResult(
        "docs-index-coverage",
        not missing,
        f"{len(expected)} nested documents are indexed exactly once"
        if not missing
        else f"not indexed: {', '.join(missing)}",
    )


def _readme_contract() -> CheckResult:
    if not README.is_file():
        return CheckResult("readme-reference-split", False, "README.md is absent")
    lines = README.read_text(encoding="utf-8").splitlines()
    text = "\n".join(lines)
    required = (
        "docs/README.md",
        "docs/architecture.md#project-status-and-governance",
    )
    missing = [link for link in required if f"]({link})" not in text]
    problems = [f"missing link {link}" for link in missing]
    if len(lines) > MAX_README_LINES:
        problems.append(f"{len(lines)} lines exceeds {MAX_README_LINES}")
    status_section = text.split("## Project status and governance", 1)[-1].split("##", 1)[0]
    status_rows = status_section.count("\n| ") - 1
    if status_rows > 6:
        problems.append(f"README status table has {status_rows} rows; keep it compact")
    return CheckResult(
        "readme-reference-split",
        not problems,
        f"compact {len(lines)}-line entry page routes status and reference detail"
        if not problems
        else "; ".join(problems),
    )


def _authority_contract() -> CheckResult:
    parallel = [path for path in FORBIDDEN_PARALLEL_AUTHORITIES if (ROOT / path).exists()]
    return CheckResult(
        "docs-authorities",
        not parallel,
        "README and architecture retain their status/plan authorities"
        if not parallel
        else f"parallel authority: {', '.join(parallel)}",
    )


def check_all() -> list[CheckResult]:
    return [_readme_contract(), _authority_contract(), _index_covers_docs(), _links_resolve()]


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
