#!/usr/bin/env python3
"""Host-safe tests for the stock-RCO adapter-injection runner.

Most of this suite runs against a dependency-free stand-in for the pinned
stock RCO tree, so it needs neither torch nor a checkpoint.  One optional
test drives the real pinned tree and is skipped when its dependencies are
absent; a skip is not a pass.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent

SPEC = importlib.util.spec_from_file_location("rco_adapter_runner", TOOLS / "rco_adapter_runner.py")
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

PINNED_ROOT = Path.home() / ".local/share/strata-tools/RCO"
PINNED_DEPENDENCIES = ("torch", "numpy", "tqdm", "datasets")

FAKE_PRUNE = '''"""Stand-in for the pinned stock RCO src/search/prune.py."""


def get_moe_info(model):
    return (1, 2, 1)


def build_ref_cache(model, data, masks, batch_size=4):
    return [], []


def compute_frequency(model, data, n_layers, n_experts, top_k, batch_size=4):
    return [[0.0] * n_experts for _ in range(n_layers)]


def compute_router_scores(model, data, n_layers, n_experts, batch_size=4):
    return [[0.0] * n_experts for _ in range(n_layers)]


def frequency_uniform_mask(freq, n_layers, n_experts, target_budget):
    return [[False] * n_experts for _ in range(n_layers)]


def optimize(
    model,
    cal_data,
    ref_log_probs,
    ref_masks,
    n_layers,
    n_experts,
    target_budget,
    **kwargs,
):
    return None, None, []


def install_wrappers(model, n_layers, n_experts, shared_state):
    return ["stock-wrapper"]


def remove_wrappers(wrappers):
    return None


def evaluate_with_mask(
    model,
    prune_mask,
    cal_data,
    ref_log_probs,
    ref_masks,
    batch_size=4,
):
    return 0.0
'''

FAKE_ENTRY = '''#!/usr/bin/env python3
"""Stand-in for the pinned stock RCO entry point: no torch, no model, no GPU."""

import argparse
import json
import os
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))

from search import prune as _prune_module
from search.prune import (
    build_ref_cache,
    compute_frequency,
    compute_router_scores,
    evaluate_with_mask,
    frequency_uniform_mask,
    get_moe_info,
    optimize,
)


class _Config:
    num_hidden_layers = 1
    num_experts = 2
    num_experts_per_tok = 1


class _Block:
    def forward(self, hidden_states):
        return hidden_states


class _Layer:
    def __init__(self):
        self.mlp = _Block()


class _Inner:
    def __init__(self):
        self.layers = [_Layer()]


class _Model:
    def __init__(self):
        self.config = _Config()
        self.model = _Inner()


def _describe(function):
    return "%s.%s" % (function.__module__, function.__qualname__)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fake-model")
    parser.add_argument("--sparsity", type=float, default=0.25)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--boom", action="store_true")
    return parser


def main(argv=None):
    args, extra = build_parser().parse_known_args(argv)
    model = _Model()
    observation = {
        "argv": list(argv or []),
        "extra": list(extra),
        "prune_module_file": _prune_module.__file__,
        "functions": {
            "get_moe_info": _describe(get_moe_info),
            "evaluate_with_mask": _describe(evaluate_with_mask),
            "install_wrappers": _describe(_prune_module.install_wrappers),
            "remove_wrappers": _describe(_prune_module.remove_wrappers),
        },
        "calls": {},
    }
    for label, function, arguments in (
        ("get_moe_info", get_moe_info, (model,)),
        (
            "install_wrappers",
            _prune_module.install_wrappers,
            (model, 1, 2, {"ste_masks": []}),
        ),
        ("remove_wrappers", _prune_module.remove_wrappers, ([],)),
        ("evaluate_with_mask", evaluate_with_mask, (model, None, None, None, None)),
    ):
        try:
            observation["calls"][label] = {"value": repr(function(*arguments))}
        except Exception as error:
            observation["calls"][label] = {
                "raised": type(error).__name__,
                "message": str(error),
            }
    path = os.environ.get("STRATA_FAKE_RCO_OBSERVATION")
    if path:
        _Path(path).write_text(json.dumps(observation, indent=2), encoding="utf-8")
    if args.boom:
        raise RuntimeError("the fake stock entry failed on purpose")
    return int(args.exit_code)
'''


def build_fake_tree(root: Path) -> Path:
    """Write a dependency-free stand-in for the pinned stock RCO tree."""
    search = root / "src" / "search"
    search.mkdir(parents=True)
    (search / "__init__.py").write_text("", encoding="utf-8")
    (search / "prune.py").write_text(FAKE_PRUNE, encoding="utf-8")
    (root / "rco_search_prune.py").write_text(FAKE_ENTRY, encoding="utf-8")
    return root


def dependencies_available(names: tuple[str, ...]) -> bool:
    """True when every named module can be located without importing it."""
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


class FakeStockTreeTest(unittest.TestCase):
    """The dependency-light contract, driven against a fake stock tree."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name).resolve()
        self.root = build_fake_tree(self.temp / "RCO")
        original_revision = RUNNER.git_revision
        self.addCleanup(lambda: setattr(RUNNER, "git_revision", original_revision))
        RUNNER.git_revision = lambda _root: RUNNER.PINNED_RCO_REVISION

        self.observation_path = self.temp / "observation.json"
        os.environ["STRATA_FAKE_RCO_OBSERVATION"] = str(self.observation_path)
        self.addCleanup(os.environ.pop, "STRATA_FAKE_RCO_OBSERVATION", None)

        self.saved_path = list(sys.path)
        self.addCleanup(self._restore_path)
        sys.path.insert(0, str(self.root / "src"))

        saved_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            self.stock_prune = importlib.import_module("search.prune")
        finally:
            sys.dont_write_bytecode = saved_bytecode
        self.stock_functions = {
            name: getattr(self.stock_prune, name) for name in RUNNER.INJECTED_FUNCTIONS
        }

    def test_an_unpinned_tree_is_refused_before_import(self) -> None:
        RUNNER.git_revision = lambda _root: "0" * 40
        with self.assertRaisesRegex(RUNNER.RunnerError, "expected 9a1e09c"):
            RUNNER.run_stock(self.root, "deepseek", [])

    def _restore_path(self) -> None:
        sys.path[:] = self.saved_path
        sys.modules.pop("search.prune", None)
        sys.modules.pop("search", None)

    def observation(self) -> dict:
        return json.loads(self.observation_path.read_text(encoding="utf-8"))

    def tree_files(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in sorted(self.root.rglob("*"))
            if path.is_file()
        }

    def test_deepseek_mode_injects_the_adapter_functions(self) -> None:
        report = RUNNER.run_stock(self.root, RUNNER.MODE_DEEPSEEK, ["--model", "fake-model"])
        observed = self.observation()
        for name in RUNNER.INJECTED_FUNCTIONS:
            with self.subTest(function=name):
                self.assertEqual(
                    observed["functions"][name],
                    f"{RUNNER.ADAPTER_MODULE}.{name}",
                )
                self.assertEqual(
                    report["injection"]["functions"][
                        [item["name"] for item in report["injection"]["functions"]].index(name)
                    ]["injected"],
                    f"{RUNNER.ADAPTER_MODULE}.{name}",
                )
        self.assertTrue(report["injection"]["applied"])
        self.assertTrue(report["injection"]["restored"])
        self.assertEqual(
            report["injection"]["entry_bindings"],
            {
                "get_moe_info": True,
                "evaluate_with_mask": True,
            },
        )
        self.assertEqual(report["exit_code"], 0)

    def test_deepseek_mode_calls_the_injected_functions(self) -> None:
        RUNNER.run_stock(self.root, RUNNER.MODE_DEEPSEEK, ["--model", "fake-model"])
        calls = self.observation()["calls"]
        self.assertEqual(
            calls["get_moe_info"]["value"], "MoeCounts(n_layers=1, n_experts=2, top_k=1)"
        )
        self.assertNotIn("raised", calls["install_wrappers"])
        self.assertNotIn("raised", calls["remove_wrappers"])

    def test_stock_functions_are_restored_after_the_run(self) -> None:
        RUNNER.run_stock(self.root, RUNNER.MODE_DEEPSEEK, ["--model", "fake-model"])
        for name, original in self.stock_functions.items():
            with self.subTest(function=name):
                self.assertIs(getattr(self.stock_prune, name), original)

    def test_stock_functions_are_restored_when_the_stock_entry_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            RUNNER.run_stock(self.root, RUNNER.MODE_DEEPSEEK, ["--boom"])
        for name, original in self.stock_functions.items():
            with self.subTest(function=name):
                self.assertIs(getattr(self.stock_prune, name), original)

    def test_qwen4exp_mode_is_an_explicit_pass_through(self) -> None:
        report = RUNNER.run_stock(self.root, RUNNER.MODE_QWEN4EXP, ["--model", "fake-model"])
        observed = self.observation()
        for name in RUNNER.INJECTED_FUNCTIONS:
            with self.subTest(function=name):
                self.assertEqual(observed["functions"][name], f"search.prune.{name}")
        self.assertFalse(report["injection"]["applied"])
        self.assertEqual(report["injection"]["kind"], "pass-through")
        self.assertTrue(report["injection"]["restored"])

    def test_stock_argv_is_forwarded_verbatim(self) -> None:
        forwarded = ["--model", "fake-model", "--sparsity", "0.5", "--unknown-flag", "value"]
        report = RUNNER.run_stock(self.root, RUNNER.MODE_DEEPSEEK, forwarded)
        self.assertEqual(self.observation()["argv"], forwarded)
        self.assertEqual(report["stock_argv"], forwarded)

    def test_the_stock_exit_code_is_reported(self) -> None:
        report = RUNNER.run_stock(self.root, RUNNER.MODE_DEEPSEEK, ["--exit-code", "3"])
        self.assertEqual(report["exit_code"], 3)

    def test_the_pinned_tree_is_left_byte_identical(self) -> None:
        before = self.tree_files()
        RUNNER.run_stock(self.root, RUNNER.MODE_DEEPSEEK, ["--model", "fake-model"])
        self.assertEqual(self.tree_files(), before)
        self.assertFalse(list(self.root.rglob("__pycache__")))

    def test_the_run_writes_no_bytecode_into_the_pinned_tree(self) -> None:
        self.assertFalse(sys.dont_write_bytecode)
        RUNNER.run_stock(self.root, RUNNER.MODE_DEEPSEEK, ["--model", "fake-model"])
        self.assertFalse(list(self.root.rglob("__pycache__")))
        self.assertFalse(sys.dont_write_bytecode)

    def test_dry_run_emits_a_manifest_without_running_the_stock_entry(self) -> None:
        report = RUNNER.build_manifest(self.root, RUNNER.MODE_DEEPSEEK, ["--model", "fake-model"])
        self.assertTrue(report["dry_run"])
        self.assertFalse(self.observation_path.exists())
        self.assertTrue(report["stock_entry_present"])
        self.assertTrue(report["stock_prune_module_present"])
        self.assertEqual(len(report["stock_entry_sha256"]), 64)
        self.assertIsNone(report["injection"]["applied"])
        self.assertTrue(report["injection"]["planned"])

    def test_cli_dry_run_writes_the_manifest_file(self) -> None:
        manifest = self.temp / "manifest.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = RUNNER.main(
                [
                    "--rco-root",
                    str(self.root),
                    "--dry-run",
                    "--manifest",
                    str(manifest),
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        document = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(document["format"], RUNNER.FORMAT)
        self.assertTrue(document["dry_run"])
        self.assertEqual(json.loads(stdout.getvalue())["format"], RUNNER.FORMAT)
        self.assertFalse(self.observation_path.exists())

    def test_the_bare_separator_forwards_the_rest_of_the_command_line(self) -> None:
        args, forwarded = RUNNER.parse_args(["--mode", "qwen4exp", "--", "--model", "fake-model"])
        self.assertEqual(args.mode, "qwen4exp")
        self.assertEqual(forwarded, ["--model", "fake-model"])

    def test_unrecognized_options_are_forwarded_without_a_separator(self) -> None:
        args, forwarded = RUNNER.parse_args(
            ["--mode", "qwen4exp", "--model", "fake-model", "--sparsity", "0.5"]
        )
        self.assertEqual(args.mode, "qwen4exp")
        self.assertEqual(forwarded, ["--model", "fake-model", "--sparsity", "0.5"])

    def test_a_missing_stock_tree_is_refused(self) -> None:
        with self.assertRaises(RUNNER.RunnerError) as caught:
            RUNNER.run_stock(self.temp / "absent", RUNNER.MODE_DEEPSEEK, [])
        self.assertIn("stock RCO entry not found", str(caught.exception))

    def test_an_unknown_mode_is_refused(self) -> None:
        with self.assertRaises(RUNNER.RunnerError) as caught:
            RUNNER.run_stock(self.root, "not-a-mode", [])
        self.assertIn("unknown mode", str(caught.exception))


@unittest.skipUnless(
    dependencies_available(PINNED_DEPENDENCIES),
    "the pinned stock-RCO integration needs torch, numpy, tqdm and datasets",
)
class PinnedStockTreeTest(unittest.TestCase):
    """The optional integration against the real pinned stock RCO tree."""

    def test_the_pinned_tree_accepts_and_releases_the_adapter(self) -> None:
        entry = PINNED_ROOT / "rco_search_prune.py"
        if not entry.is_file():
            self.skipTest(f"the pinned stock RCO tree is not present at {PINNED_ROOT}")
        revision = RUNNER.git_revision(PINNED_ROOT)
        if revision != RUNNER.PINNED_RCO_REVISION:
            self.skipTest(f"the pinned tree is at {revision}, not {RUNNER.PINNED_RCO_REVISION}")

        saved_path = list(sys.path)
        saved_modules = {name: sys.modules.get(name) for name in ("search", "search.prune")}
        saved_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            prune = RUNNER.import_stock_prune_module(PINNED_ROOT)
            adapter = RUNNER.load_module_from_path(
                RUNNER.ADAPTER_MODULE, TOOLS / RUNNER.ADAPTER_FILE_NAME
            )
            originals = RUNNER.inject_adapter_functions(prune, adapter)
            try:
                for name in RUNNER.INJECTED_FUNCTIONS:
                    with self.subTest(function=name):
                        self.assertIs(getattr(prune, name), getattr(adapter, name))
            finally:
                self.assertTrue(RUNNER.restore_stock_functions(prune, originals))
            for name, original in originals.items():
                with self.subTest(function=name):
                    self.assertIs(getattr(prune, name), original)
        finally:
            sys.path[:] = saved_path
            sys.dont_write_bytecode = saved_bytecode
            for name, value in saved_modules.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


if __name__ == "__main__":
    unittest.main()
