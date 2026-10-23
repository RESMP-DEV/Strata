#!/usr/bin/env python3
"""Run pinned stock RCO expert pruning with the Strata DeepSeek adapter injected.

Stock RCO (``~/.local/share/strata-tools/RCO``, pinned at
``9a1e09c07d468109cbe60a1b87d5036034a79d10``) prunes experts on Qwen3-Next /
OLMoE style MoE blocks.  DeepSeek-V4.1 breaks four of that pipeline's
assumptions, and ``tools/deepseek_rco_adapter.py`` implements them faithfully.
Until now that substitution was applied ad hoc, in one process, on one host.

This runner makes it reproducible and host-safe:

* the pinned tree is opened read-only.  Nothing in it is written, patched on
  disk, or checked out, and bytecode writing is disabled for the duration of
  the run so the tree does not even collect fresh ``__pycache__`` entries;
* the stock entry ``rco_search_prune.py`` is imported as a module and the
  stock ``search.prune`` module is patched in memory.  The entry's
  ``from search.prune import get_moe_info, evaluate_with_mask`` therefore
  binds the adapter, and ``search.prune.optimize`` calls the adapter's
  ``install_wrappers`` / ``remove_wrappers`` through its module globals;
* the four stock functions are restored when the run ends, whether it ends
  normally, raises, or exits early;
* the explicit ``qwen4exp`` mode is a pass-through: stock RCO runs with no
  injection at all, because Qwen4Exp already fits stock RCO's pruning wrapper.

The runner itself imports only the standard library.  Torch, NumPy, Datasets
and Transformers are needed by the stock tree, not by this file, so
``--dry-run`` and the manifest work on a host that has none of them.

Usage::

    python3 tools/rco_adapter_runner.py --dry-run --json
    python3 tools/rco_adapter_runner.py --mode deepseek -- \\
        --model /path/to/model --sparsity 0.5 --save-mask /tmp/mask.pt
    python3 tools/rco_adapter_runner.py --mode qwen4exp -- \\
        --model /path/to/qwen4exp --sparsity 0.25

Everything after a bare ``--`` is forwarded verbatim to the stock entry's own
argument parser.  Without a ``--``, whatever this runner does not recognize is
forwarded instead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

FORMAT = "strata-rco-adapter-runner-v1"

DEFAULT_RCO_ROOT = Path.home() / ".local/share/strata-tools/RCO"
PINNED_RCO_REVISION = "9a1e09c07d468109cbe60a1b87d5036034a79d10"

STOCK_ENTRY_NAME = "rco_search_prune.py"
STOCK_PRUNE_RELATIVE = ("src", "search", "prune.py")
STOCK_SEARCH_PACKAGE = "search"
STOCK_PRUNE_MODULE = "search.prune"
STOCK_ENTRY_MODULE = "strata_rco_stock_entry"

ADAPTER_FILE_NAME = "deepseek_rco_adapter.py"
ADAPTER_MODULE = "strata_deepseek_rco_adapter"

MODE_DEEPSEEK = "deepseek"
MODE_QWEN4EXP = "qwen4exp"
MODES = (MODE_DEEPSEEK, MODE_QWEN4EXP)

INJECTED_FUNCTIONS = (
    "evaluate_with_mask",
    "get_moe_info",
    "install_wrappers",
    "remove_wrappers",
)

MODE_SEMANTICS = {
    MODE_DEEPSEEK: (
        "Injects the Strata DeepSeek-V4.1 adapter into the stock search.prune "
        "module for the duration of the run, then restores the stock functions."
    ),
    MODE_QWEN4EXP: (
        "Explicit pass-through: stock RCO runs unmodified, with no injection. "
        "Qwen4Exp already fits stock RCO's pruning wrapper, so no adapter is "
        "substituted."
    ),
}

NON_CLAIMS = (
    "The runner performs no CUDA work, loads no model weights, and contacts no host.",
    "A dry run proves the pinned tree is present and hash-stable; it proves no RCO "
    "search, pruning quality, or KL number.",
    "Injection swaps four in-process attributes on the stock search.prune module; "
    "it modifies no file in the pinned RCO tree.",
    "The pass-through mode runs stock RCO unchanged and claims nothing about any "
    "other model topology.",
    "A skipped optional test is not a pass; the dependency-light suite is the host-safe evidence.",
    "No released checkpoint was opened, and no pruning budget, quality, or runtime "
    "claim follows from this runner.",
)


class RunnerError(RuntimeError):
    """Raised when the pinned tree or the injection contract is not satisfied."""


def file_digest(path: Path) -> str | None:
    """SHA-256 of a file, or ``None`` when the file does not exist."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    """Read the tree's HEAD locally, without contacting any host."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def qualify(obj: Any) -> str:
    """Qualified ``module.name`` of a function, for the manifest."""
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    return repr(obj)


def load_module_from_path(name: str, path: Path) -> ModuleType:
    """Import ``path`` under ``name`` without importing it as part of a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RunnerError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def import_stock_prune_module(root: Path) -> ModuleType:
    """Import the pinned tree's ``search.prune`` under its stock module name.

    The stock entry imports ``from search.prune import ...``, so the patched
    module has to be the one registered as ``search.prune`` in ``sys.modules``.
    A ``search`` package left over from a different tree is evicted first, so
    a stale module can never be patched in place of the pinned one.
    """
    src = (root / "src").resolve()
    prune_path = src.joinpath("search", "prune.py")
    if not prune_path.is_file():
        raise RunnerError(f"stock prune module not found: {prune_path}")
    sys.path.insert(0, str(src))
    package = sys.modules.get(STOCK_SEARCH_PACKAGE)
    if package is not None:
        paths = getattr(package, "__path__", ())
        resident = any(Path(entry).resolve() == src / "search" for entry in paths)
        if not resident:
            sys.modules.pop(STOCK_PRUNE_MODULE, None)
            sys.modules.pop(STOCK_SEARCH_PACKAGE, None)
    try:
        return importlib.import_module(STOCK_PRUNE_MODULE)
    except ImportError as error:
        raise RunnerError(f"could not import the stock prune module: {error}") from error


def inject_adapter_functions(
    prune_module: ModuleType, adapter_module: ModuleType
) -> dict[str, Any]:
    """Replace the four stock functions with the Strata adapter's.

    Returns the stock originals so they can be restored exactly.
    """
    originals: dict[str, Any] = {}
    for name in INJECTED_FUNCTIONS:
        replacement = getattr(adapter_module, name, None)
        if not callable(replacement):
            raise RunnerError(f"the Strata adapter has no callable '{name}'")
        if not hasattr(prune_module, name):
            raise RunnerError(f"stock {STOCK_PRUNE_MODULE} has no '{name}' to replace")
        originals[name] = getattr(prune_module, name)
    for name in INJECTED_FUNCTIONS:
        setattr(prune_module, name, getattr(adapter_module, name))
    return originals


def restore_stock_functions(prune_module: ModuleType, originals: dict[str, Any]) -> bool:
    """Put the stock functions back and report whether every one is the original."""
    for name, value in originals.items():
        setattr(prune_module, name, value)
    return all(getattr(prune_module, name) is value for name, value in originals.items())


def build_parser() -> argparse.ArgumentParser:
    """The runner's own options.  Stock RCO options are forwarded, not parsed here."""
    parser = argparse.ArgumentParser(
        prog="rco_adapter_runner.py",
        description=(
            "Run the pinned stock RCO entry with the Strata DeepSeek adapter "
            "injected, or as an explicit Qwen4Exp pass-through."
        ),
        epilog=(
            "Everything after a bare -- is forwarded verbatim to the stock "
            "rco_search_prune.py argument parser."
        ),
    )
    parser.add_argument(
        "--rco-root",
        type=Path,
        default=DEFAULT_RCO_ROOT,
        help="pinned stock RCO tree (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=MODE_DEEPSEEK,
        help=(
            "deepseek injects the Strata adapter; qwen4exp is an explicit "
            "pass-through that runs stock RCO unmodified"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="emit the manifest and stop; the stock entry is not imported or run",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="write the run manifest as JSON to this path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the run manifest as JSON on stdout",
    )
    return parser


def split_argv(argv: list[str]) -> tuple[list[str], list[str], bool]:
    """Split the runner's own options from the argv forwarded to stock RCO.

    A bare ``--`` is the explicit separator.  Without one, the runner's parser
    collects what it does not recognize and that is forwarded instead, so
    ``--model X`` still reaches stock RCO.
    """
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1 :], True
    return argv, [], False


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse the runner's options and return the argv destined for stock RCO."""
    parser = build_parser()
    own, forwarded, separated = split_argv(argv)
    if separated:
        return parser.parse_args(own), forwarded
    args, extra = parser.parse_known_args(own)
    return args, extra


def _injection_block(
    mode: str,
    functions: list[dict[str, Any]],
    applied: bool | None,
    restored: bool | None,
    entry_bindings: dict[str, bool],
) -> dict[str, Any]:
    """The manifest's injection record."""
    return {
        "kind": "inject" if mode == MODE_DEEPSEEK else "pass-through",
        "applied": applied,
        "planned": mode == MODE_DEEPSEEK,
        "target_module": STOCK_PRUNE_MODULE,
        "functions": functions,
        "restored": restored,
        "entry_bindings": entry_bindings,
    }


def build_manifest(root: Path, mode: str, stock_argv: list[str]) -> dict[str, Any]:
    """Describe what a run would do, without importing or running stock RCO."""
    root = root.expanduser().resolve()
    entry = root / STOCK_ENTRY_NAME
    prune_path = root.joinpath(*STOCK_PRUNE_RELATIVE)
    revision = git_revision(root)
    if revision != PINNED_RCO_REVISION:
        raise RunnerError(
            f"RCO tree {root} is at {revision or 'an unreadable revision'}, "
            f"expected {PINNED_RCO_REVISION}"
        )
    functions = [
        {
            "name": name,
            "stock": f"{STOCK_PRUNE_MODULE}.{name}",
            "injected": f"{ADAPTER_MODULE}.{name}",
        }
        for name in INJECTED_FUNCTIONS
    ]
    return {
        "format": FORMAT,
        "mode": mode,
        "mode_semantics": MODE_SEMANTICS[mode],
        "dry_run": True,
        "rco_root": str(root),
        "rco_revision": revision,
        "rco_revision_pinned": PINNED_RCO_REVISION,
        "rco_revision_matches_pinned": ((revision == PINNED_RCO_REVISION) if revision else None),
        "stock_entry": str(entry),
        "stock_entry_present": entry.is_file(),
        "stock_entry_sha256": file_digest(entry),
        "stock_prune_module": str(prune_path),
        "stock_prune_module_present": prune_path.is_file(),
        "stock_prune_module_sha256": file_digest(prune_path),
        "bytecode_writes_disabled": True,
        "injection": _injection_block(mode, functions, None, None, {}),
        "stock_argv": list(stock_argv),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "exit_code": None,
        "error": None,
        "non_claims": list(NON_CLAIMS),
    }


def run_stock(root: Path, mode: str, stock_argv: list[str]) -> dict[str, Any]:
    """Run the pinned stock entry under ``mode`` and return the run manifest."""
    if mode not in MODES:
        raise RunnerError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    root = root.expanduser().resolve()
    entry = root / STOCK_ENTRY_NAME
    prune_path = root.joinpath(*STOCK_PRUNE_RELATIVE)
    if not entry.is_file():
        raise RunnerError(f"stock RCO entry not found: {entry}")
    revision = git_revision(root)
    if revision != PINNED_RCO_REVISION:
        raise RunnerError(
            f"RCO tree {root} is at {revision or 'an unreadable revision'}, "
            f"expected {PINNED_RCO_REVISION}"
        )

    before = {
        "stock_entry_sha256": file_digest(entry),
        "stock_prune_module_sha256": file_digest(prune_path),
    }

    saved_path = list(sys.path)
    saved_bytecode = sys.dont_write_bytecode
    saved_modules = {
        name: sys.modules.get(name)
        for name in (STOCK_SEARCH_PACKAGE, STOCK_PRUNE_MODULE, STOCK_ENTRY_MODULE)
    }
    prune_module: ModuleType | None = None
    originals: dict[str, Any] = {}
    functions: list[dict[str, Any]] = [
        {"name": name, "stock": f"{STOCK_PRUNE_MODULE}.{name}", "injected": None}
        for name in INJECTED_FUNCTIONS
    ]
    entry_bindings: dict[str, bool] = {}
    exit_code = 0
    restored = True
    sys.dont_write_bytecode = True
    try:
        if mode == MODE_DEEPSEEK:
            prune_module = import_stock_prune_module(root)
            adapter_module = load_module_from_path(
                ADAPTER_MODULE, Path(__file__).resolve().parent / ADAPTER_FILE_NAME
            )
            originals = inject_adapter_functions(prune_module, adapter_module)
            functions = [
                {
                    "name": name,
                    "stock": qualify(originals[name]),
                    "injected": qualify(getattr(prune_module, name)),
                }
                for name in INJECTED_FUNCTIONS
            ]
        entry_module = load_module_from_path(STOCK_ENTRY_MODULE, entry)
        if prune_module is not None:
            entry_bindings = {
                name: getattr(entry_module, name, None) is getattr(prune_module, name, None)
                for name in ("get_moe_info", "evaluate_with_mask")
            }
        exit_code = _call_stock_entry(entry_module, list(stock_argv))
    finally:
        if prune_module is not None and originals:
            restored = restore_stock_functions(prune_module, originals)
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_bytecode
        for name, value in saved_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    after = {
        "stock_entry_sha256": file_digest(entry),
        "stock_prune_module_sha256": file_digest(prune_path),
    }
    return {
        "format": FORMAT,
        "mode": mode,
        "mode_semantics": MODE_SEMANTICS[mode],
        "dry_run": False,
        "rco_root": str(root),
        "rco_revision": revision,
        "rco_revision_pinned": PINNED_RCO_REVISION,
        "rco_revision_matches_pinned": ((revision == PINNED_RCO_REVISION) if revision else None),
        "stock_entry": str(entry),
        "stock_entry_present": True,
        "stock_entry_sha256": before["stock_entry_sha256"],
        "stock_prune_module": str(prune_path),
        "stock_prune_module_present": prune_path.is_file(),
        "stock_prune_module_sha256": before["stock_prune_module_sha256"],
        "tree_unmodified_after_run": before == after,
        "bytecode_writes_disabled": True,
        "injection": _injection_block(
            mode, functions, mode == MODE_DEEPSEEK, restored, entry_bindings
        ),
        "stock_argv": list(stock_argv),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "exit_code": exit_code,
        "error": None,
        "non_claims": list(NON_CLAIMS),
    }


def _call_stock_entry(entry_module: ModuleType, stock_argv: list[str]) -> int:
    """Call the stock entry's ``main`` and normalize its exit code."""
    try:
        return int(entry_module.main(stock_argv) or 0)
    except SystemExit as stop:
        code = stop.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1


def summarize(report: dict[str, Any]) -> str:
    """One compact human-readable line about a run."""
    injection = report.get("injection") or {}
    parts = [
        f"rco-adapter-runner mode={report.get('mode')}",
        f"root={report.get('rco_root')}",
        f"revision={report.get('rco_revision') or 'unknown'}",
        f"injection={injection.get('kind')}",
        f"applied={injection.get('applied')}",
        f"restored={injection.get('restored')}",
        f"tree_unmodified={report.get('tree_unmodified_after_run')}",
        f"exit_code={report.get('exit_code')}",
    ]
    if report.get("error"):
        parts.append(f"error={report['error']}")
    return " ".join(parts)


def emit(report: dict[str, Any], args: argparse.Namespace) -> None:
    """Write the manifest file and/or print the manifest."""
    if args.manifest is not None:
        path = args.manifest.expanduser()
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(summarize(report))


def failure_report(
    args: argparse.Namespace,
    stock_argv: list[str],
    exit_code: int,
    error: Exception,
) -> dict[str, Any]:
    """The manifest recorded when a run could not start or could not finish."""
    root = args.rco_root.expanduser().resolve()
    revision = git_revision(root)
    return {
        "format": FORMAT,
        "mode": args.mode,
        "mode_semantics": MODE_SEMANTICS[args.mode],
        "dry_run": bool(args.dry_run),
        "rco_root": str(root),
        "rco_revision": revision,
        "rco_revision_pinned": PINNED_RCO_REVISION,
        "rco_revision_matches_pinned": ((revision == PINNED_RCO_REVISION) if revision else None),
        "injection": _injection_block(args.mode, [], None, None, {}),
        "stock_argv": list(stock_argv),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "exit_code": exit_code,
        "error": f"{type(error).__name__}: {error}",
        "non_claims": list(NON_CLAIMS),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns the stock entry's exit code."""
    raw = list(sys.argv[1:] if argv is None else argv)
    args, stock_argv = parse_args(raw)
    try:
        if args.dry_run:
            report = build_manifest(args.rco_root, args.mode, stock_argv)
        else:
            report = run_stock(args.rco_root, args.mode, stock_argv)
    except RunnerError as error:
        emit(failure_report(args, stock_argv, 2, error), args)
        return 2
    except Exception as error:  # the manifest records the failure before it propagates
        emit(failure_report(args, stock_argv, 1, error), args)
        raise
    emit(report, args)
    return int(report.get("exit_code") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
