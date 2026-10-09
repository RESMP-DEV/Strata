# Strata agent instructions

<purpose>
Strata runs Qwen3.8-Flash-Next and its Coder, Swift 1.5 and Unsloth variants
on an ordinary PC with one NVIDIA, AMD, or supported Intel graphics card plus
system RAM. The project has a C++/CUDA/HIP/SYCL engine in `src/`, `include/`,
and `sycl/`; a Python server and web app in `serve/`; and a one-click installer
in `setup.py`, `setup.sh`, and the Windows launchers.

The canonical repository is `RESMP-DEV/Strata`. `AGENTS.md` is the only
instruction file; `CLAUDE.md` is a relative symlink to it. Do not add another
agent-specific instruction file.
</purpose>

<establish_context>
1. Read this file, inspect `git status --short`, and identify the requested
   scope before editing. Preserve unrelated dirty work and active worktrees.
2. Read `docs/architecture.md` for ownership and open decisions. Read
   `CONTRIBUTING.md` for checks, evidence, and backend requirements.
3. Treat existing technical documentation as observations, not live proof.
   Probe the actual installer, API, engine binary, accelerator, or service
   before making an operational claim.
4. The README status table owns current implementation and exercise status.
   Dated evidence and work-log entries record observations at a revision; they
   never upgrade a proposed feature into an implemented feature.
</establish_context>

<ownership_map>
| Concern | Canonical owner |
| --- | --- |
| Agent and contributor contract | `AGENTS.md` |
| Product status and entry point | `README.md` |
| Architecture, ownership, open choices, append-only work log | `docs/architecture.md` |
| Procedure, code standards, check matrix | `CONTRIBUTING.md` |
| Engine and backend build | `src/`, `include/`, `CMakeLists.txt`, `sycl/` |
| Installer | `setup.py`, `setup.sh`, `START-HERE.bat`, `SETUP.bat`, `UPDATE.bat` |
| Local server and API | `serve/` |
| Durable sanitized receipts | `evidence/` |
| Generated receipts and logs | ignored `build/reports/` |

Specialized docs under `docs/` are allowed when they contain operational detail
that does not belong in `docs/architecture.md`; do not create another roadmap,
status summary, or architecture file.
</ownership_map>

<remote_execution_hosts>
The main machine remains the sole authority for Git, credentials, planning, task
state, and acceptance. Two SSH aliases identify the two remote execution hosts;
both currently report the hostname `compute`, so never select either host by
hostname. Verify address and hardware before each operational claim.

| Alias | Identity | Boundary |
| --- | --- | --- |
| `B550` (also `compute`) | 192.168.1.171; dual RTX 3090 Ti | Bounded CUDA/heavy-GPU execution only; no authoritative project state |
| `strix` | 192.168.1.182; ROG Strix G615LP with RTX 5070 Laptop, Arrow Lake NPU, and AVX2/AVX-VNNI CPU | Bounded heterogeneous validation only; no authoritative project state. The host is intentionally modified |

For Strix, use the SSH alias rather than its reported hostname. The NVIDIA path
is the existing CUDA engine. Intel NPU and CPU execution are available through
the installed Level Zero/OpenVINO runtime, but are not Strata engine backends
until an opt-in integration is implemented and measured. The `kearm` account is
a member of `render` for `/dev/accel/accel0`; preserve that access boundary.
Do not download a model on either remote host without current-task
authorization.
</remote_execution_hosts>

<quality_and_receipts>
Use `just quality` for the host-safe repository gate. It checks project
structure, selected setup regressions, script unit tests, Ruff on maintained
scripts, ShellCheck at warning severity, whitespace, and writes
`build/reports/quality/receipt.json`. The runner removes an old receipt before
starting and writes a failure receipt on startup errors.

That gate does not compile a backend or prove an engine change. Apply the
affected rows in `CONTRIBUTING.md`: build every backend a file touches, test
the affected setup or server behavior, and use real hardware for performance
claims. Compare performance against a same-day control and name the hardware,
revision, model, quantization, workload, and what was not measured.

Every non-trivial state change or trusted finding leaves a receipt. Prefer
`build/reports/<mode>/receipt.json`; copy a sanitized durable result into
`evidence/` when later work must be able to find it. Record command, exit,
revision, dirty state, scope, platform, and non-claims. A worker completion is
evidence, not acceptance.
</quality_and_receipts>

<project_boundaries>
- User setup follows `docs/AI_SETUP.md`; check the PC, choose by RAM, install
  non-interactively, verify the server, and connect applications. Never expose
  the server beyond `127.0.0.1` without `--api-key`.
- A new feature is opt-in. The default install and default runtime path stay
  byte-identical unless a measured decision recorded in architecture changes
  them.
- Published claims follow the existing docs style: plain words, measured
  numbers, named hardware, and explicit limits. No unmeasured speed or support
  claim enters documentation.
- Never create GitHub Actions or author `.github/workflows/`. Local commands
  and receipts are the verification path.
- Use the existing CLI/server interfaces before desktop automation. The MCP
  interface in `docs/MCP_SERVER.md` remains an alternative surface, not a
  reason to build another one.
- Preserve upstream technical content while this private RESMP-DEV mirror
  evolves. Do not push to `upstream`; upstream contribution requires an explicit
  instruction in the active task.
</project_boundaries>

<coordination>
For multi-agent implementation, create one ignored worktree per writer under
`.worktrees/`, define file ownership and one build owner, and have the parent
integrate and verify the combined result. Green tests in a worker checkout do
not establish integration acceptance. Hand off changed files, exact source
revision and dirty state, commands and results, evidence paths, unresolved
findings, and limits.
</coordination>
