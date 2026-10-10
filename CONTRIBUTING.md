# Contributing to Strata

This is the canonical procedure for human and agent contributors. Start with
[README.md](README.md) for product status and
[architecture](docs/architecture.md) for ownership and open decisions. Agent
session rules live in [AGENTS.md](AGENTS.md).

## Baseline and work records

Start from a committed baseline and record `git rev-parse HEAD` plus any dirty
work before editing. For an isolated change:

```sh
git fetch origin
git worktree add -b work/my-change .worktrees/my-change origin/main
```

Keep substantial execution state in an ignored `build/<task>/work.md`. Record
the objective, exclusions, source revision, temporary file ownership, checks,
findings, handoff, and non-claims. Update it instead of creating a parallel
roadmap.

## Code standards

- Preserve the surrounding style. Python is 3.10+; maintained scripts use
  modern typing, `pathlib`, f-strings, and standard-library subprocess argv
  lists. `ruff.toml` owns mechanical style for `scripts/`; do not reformat the
  historical `tools/` or `serve/` tree incidentally.
- C++ and backend changes follow the existing backend conventions and build
  files. Keep CUDA, HIP, and SYCL-specific dependencies behind their existing
  boundaries. Build every backend touched by a file.
- New behavior is opt-in. Preserve the default path byte-for-byte unless a
  measured architecture decision changes it.
- Validate sizes, paths, external IDs, GPU responses, and protocol state before
  allocation or mutation. Preserve original failures in cleanup paths and bound
  downloads, queues, logs, and subprocess work.
- Tests should observe a public contract and its failure recovery, not an
  implementation detail. Isolate temporary files and endpoints, await completion
  for positive async assertions, and label negative observation windows.
- Scripts use `#!/usr/bin/env bash`; keep portable bash semantics and avoid
  shell text assembled into commands for untrusted input.

## Required checks

Run `just quality` first on a host without GPU toolchains. It writes
`build/reports/quality/receipt.json` and covers maintained structure, selected
setup regressions, the host-safe accelerator-plugin ABI test, script tests,
Ruff, ShellCheck warning severity, and Git whitespace. It intentionally does
not compile an engine.

| Changed area | Required evidence |
| --- | --- |
| `AGENTS.md`, `CONTRIBUTING.md`, governance docs, receipt runner | `just quality` |
| `scripts/` | `just quality`; add or update the colocated test for each check domain |
| Host-safe plugin ABI and probe | `python3 scripts/check_plugins.py` (included in `just quality`) |
| `setup.py` or installer behavior | Affected `python3 tools/test_setup_<name>.py`, plus a fresh installer smoke test appropriate to the platform |
| `serve/` | Affected `serve/test_*.py` and an API/request lifecycle check that exercises the changed boundary |
| Engine source shared by CUDA and HIP | CUDA and HIP builds for the affected target; no SYCL claim unless SYCL also builds |
| SYCL-only file | SYCL build and the affected test or documented probe |
| Quantization, routing, batching, or memory behavior | Same-day hardware control on the named card and model; exact workload, revision, cache state, and non-claims in the receipt |
| Public speed or hardware support claim | Measured result on named hardware; compare with the previous build in the same session |

The selected setup modules in `scripts/check_setup.py` are a fast regression
set, not the complete installer matrix. Run them directly with
`python3 scripts/check_setup.py`, or run a named module with
`python3 tools/test_setup_<name>.py`. The full current tree also contains tests
that expect native build toolchains or the upstream release endpoint; if one is
not runnable, record the exact failure rather than broadening a green claim.

Reuse a recent receipt only when its recorded revision, dirty state, command
scope, platform, and inputs still match. Link the receipt; do not copy it into
conversation and call that verification. Re-run the affected checks after
merging concurrent branches.

## Receipts

Every non-trivial unit of work leaves a persistent receipt with the date, agent,
source revision and dirty state, commands and exit codes, measurements, changed
contracts, and non-claims. The runner invalidates its prior PASS before work,
publishes JSON atomically, and retains logs under ignored `build/`.

For hand-picked durable evidence, add a sanitized JSON or Markdown file under
`evidence/` and link it from `docs/architecture.md` or README status. Machine
state and large logs stay ignored.

## Pull requests and reports

Use one change per PR. Say what changes, what stays untouched, and exactly how
it was checked. Search existing issues and PRs first. Community hardware
reports follow [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md)
and include a same-day comparison against the build being tested.
