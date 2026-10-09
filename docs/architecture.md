# Strata architecture and plan

This is the single implementation plan and decision record for the RESMP-DEV
Strata checkout. Technical references under `docs/` retain their specialized
procedures and measurements; this file owns how those concerns fit together.
README owns current product status. Nothing here makes an unmeasured hardware
claim.

## Product philosophy

Correctness first: a quantization, routing, cache, or installer optimization is
not accepted because it is clever, but because its outputs and failure paths are
correct on the affected backend. Measured performance comes second: changes that
claim speed need a same-day control on named hardware. Usability comes third,
but it is a requirement for the supported default path: setup, start, app
connection, and recovery must remain operations an ordinary PC owner can
complete. A default stays predictable unless evidence recorded below changes it.

The current product scope is inherited from the upstream Strata code at
`fb58e0d`: local Qwen3.8-Flash-Next inference on supported NVIDIA, AMD, and
experimental Intel hardware, with the local API, web app, installer, and
specialized backend documentation already present. The 2026-10-09 governance
change did not modify runtime behavior.

## Ownership boundaries

| Concern | Owner | Current position | Decision gate |
| --- | --- | --- | --- |
| Agent contract | `AGENTS.md` | Canonical instructions; `CLAUDE.md` symlinks to it | Keep one instruction authority and a passing structural check |
| Product status | `README.md` | Status table distinguishes inherited implementation from locally exercised work | Update only with a receipt or live command evidence |
| Architecture and decisions | `docs/architecture.md` | Stable ownership/open-choice front matter plus append-only work log | A lasting decision needs claim, evidence, revision, and non-claims |
| Procedure and checks | `CONTRIBUTING.md` | Host-safe common gate plus affected-backend matrix | Expand only after a command is runnable and scoped |
| Engine | `src/`, `include/`, `CMakeLists.txt` | CUDA and HIP are primary; SYCL remains separately qualified | Compile every backend touched; measure behavior on real hardware |
| Installer | `setup.py`, shell and Windows launchers | Existing installer tests cover many choices without downloads | Run affected setup tests and a platform-appropriate smoke test |
| Local API and web app | `serve/` | OpenAI-, Anthropic-, and Responses-compatible surfaces are implemented in the inherited tree | Exercise the changed protocol and lifecycle boundary |
| Verification state | `build/reports/`, `evidence/` | Generated receipts are transient; durable evidence is tracked | Receipts must bind revision, dirty state, scope, commands, and non-claims |

## Open choices

| Topic | Current position | Decision gate |
| --- | --- | --- |
| Common quality mode | `just quality` checks host-safe structure and selected regressions | Add a check only with a test and an explicit non-claim |
| Full setup matrix | Existing full discovery is not accepted as a gate on macOS Python 3.14 because native build and upstream-release cases fail | A dedicated setup change runs its named modules and the affected platform smoke test |
| Backend acceptance | No local backend build is claimed on this Mac | Require named compiler/driver/backend evidence on the target host |
| Performance defaults | Preserve upstream defaults byte-identically | Same-day matched control and exact-output/failure-path evidence |
| Public artifact policy | Private RESMP-DEV mirror; no GitHub Actions or upstream push | Explicit current-task authorization for anything outward-facing |

## Contracts to preserve

- `AGENTS.md` is the only agent instruction file, and `CLAUDE.md` is its
  relative symlink.
- No GitHub Actions workflow may be authored or enabled; verification is local
  with durable receipts.
- A common green gate never substitutes for compiling and testing every backend
  touched by an engine file.
- New features are opt-in, and defaults remain byte-identical until a measured
  architecture decision changes them.
- Documentation distinguishes measured values from hypotheses and names the
  hardware, model, revision, workload, and limits.
- The local server is loopback by default; wider exposure requires an API key
  and explicit user intent.
- Receipts bind evidence to revision and scope, state dirty provenance, and say
  what they do not prove.

## Work log

### Project stand-up, 2026-10-09

Converted the clean checkout from a minimal upstream agent note to the Lapis
style without changing runtime code: added the instruction, architecture,
procedure, status, receipt, and `just` authorities; added tested structural and
setup check domains; added ignore rules for local worktrees and semantic-index
state. Baseline `fb58e0d` was clean. Full setup discovery was probed before the
change and failed on this macOS Python 3.14 host (418 tests: 2 failures, 4
errors, 1 skip), so the common gate uses six deterministic setup modules and
explicitly does not claim the full installer matrix. Final validation is linked
from [project stand-up evidence](../evidence/project-standup.json).

The receipt runner was also tested as code. A focused startup-failure test
found that receipt invalidation was outside the guarded region, so an
interruption at startup could leave no failure receipt. Invalidation now happens
inside the guarded region, a startup failure writes a failing receipt, and
publication atomically replaces even a symlinked final path. The focused
contract result is recorded in
[unlink safety evidence](../evidence/unlink-safety-check.json).
