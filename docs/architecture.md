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
| Product status and entry point | `README.md` | README owns compact user-facing status and links this file's detailed implementation/exercise matrix; architecture owns the decisions and evidence record | Update only with a receipt or live command evidence; the README table must stay compact and link here |
| Architecture and decisions | `docs/architecture.md` | Stable ownership/open-choice front matter plus append-only work log | A lasting decision needs claim, evidence, revision, and non-claims |
| Procedure and checks | `CONTRIBUTING.md` | Host-safe common gate plus affected-backend matrix | Expand only after a command is runnable and scoped |
| Engine | `src/`, `include/`, `CMakeLists.txt` | CUDA and HIP are primary; SYCL remains separately qualified | Compile every backend touched; measure behavior on real hardware |
| Accelerator plugin seam | `include/strata/runtime/`, `src/runtime/`, `plugins/` | Backend-neutral ABI v1, reference CPU plugin, probe, and external OpenVINO/NPU router plugin; no default engine call site | An accelerator becomes a routing candidate only after real-operation parity, explicit opt-in configuration, failure tests, and same-day transfer/latency accounting |
| Quantization calibration | `tools/standard_imatrix.py`, opt-in llama.cpp token patch | Pins the RESMP-DEV Qwen3.8 `[256,2048]` token view and preserves its selected windows; the public mradermacher matrix remains a separate dated input | A standard-calibrated artifact needs deterministic quantization plus quality and same-day throughput/memory controls |
| DeepSeek V4.1 quantization and port | exact 1017 tensor-taxonomy gate, geometry contract, and deterministic CPU candidate-selection reference are implemented host-side; RESMP-owned converter/quantizer remains to be built, with Lucebox commit `5ada2a5` as a read-only semantic reference | External prequantized artifacts, including Lucebox ROCMFP files, are research inputs only; runnable Strata artifacts must be produced and hashed by an RESMP-owned pipeline from a pinned source checkpoint | Require exact pack inversion, deterministic conversion, backend parity, DSpark parity, quality controls, and same-day memory/throughput evidence before an executable path is claimed |
| Installer | `setup.py`, shell and Windows launchers | Existing installer tests cover many choices without downloads | Run affected setup tests and a platform-appropriate smoke test |
| Local API and web app | `serve/` | OpenAI-, Anthropic-, and Responses-compatible surfaces are implemented in the inherited tree | Exercise the changed protocol and lifecycle boundary |
| Verification state | `build/reports/`, `evidence/` | Generated receipts are transient; durable evidence is tracked | Receipts must bind revision, dirty state, scope, commands, and non-claims |
| Remote hardware | SSH aliases `B550`/`compute` and `strix` | B550 owns bounded heavy-CUDA execution; Strix owns bounded heterogeneous NVIDIA/NPU/CPU validation | The main machine remains authoritative; select remotes by alias and verify identity live |
| Strix host watcher | deployed `strata-watch.service`, source `tools/strix_watch.py` | Persistent PSI/disk/SSH/journald sampling is enabled on Strix even when interactive SSH is unavailable | A hardware failure claim requires watcher evidence plus live SMART/controller/journal state |
| Flash-Next phase-specialized prefiller | [prefiller architecture](PREFILLER.md), planner `tools/prefiller_plan.py` | The 27B checkpoint/paper are templates only; a host-only planner binds exact Flash-Next dense and hot-expert MXFP4 sizes to deployed pack metadata, with no trainer or runtime path | Training and runtime promotion require real routing traces, phase-masked QADD controls, state-handoff parity, opt-in engine integration, and many-agent mixed-hit serving evidence |

## Current quantization surface

The setup labels describe whole releases, not always one on-disk GGUF type. A
model file commonly mixes gate/up and down formats by tensor and layer.

| Setup choice | Effective expert storage | Current boundary |
| --- | --- | --- |
| `Q2_0` | Strata GSQ-RCO 2-bit blocks in the canonical pack | Fastest normal path; `tools/strata_pack.py` owns the relayout and exact inverse checks |
| `IQ2_XS` | Native mixed i-quant blocks, around 2 bits overall | Preserved in raw GGUF form by `tools/iq_pack.py`; the label does not mean every tensor is IQ2_XS |
| `IQ3_XXS` | Native mixed i-quant blocks, around 3 bits overall | Slower CPU work per token than the 2-bit packs |
| `IQ3_S` | Native mixed i-quant blocks, around 3.5 bits overall | Best-quality normal setup choice; the published full-model comparison belongs to upstream evidence |
| Coder `IQ1_M` | Half the routed experts, stored in mixed IQ2/IQ3/IQ4-like forms | The historical name is misleading: this is not a globally 1-bit pack |
| `UD-Q4_K_XL` | Mixed Q4_K/Q5_K gate-up with Q5_1/Q8_0 down paths | Experimental, NVIDIA-only in setup, and may stream experts from SSD under RAM pressure |
| `UD-IQ4_XS` | IQ3_S-style gate/up with IQ4_NL and Q8_0 down paths | Regular setup choice; still uses an expert RAM budget on smaller systems |

The CPU path has ggml-backed native rows for the mixed formats plus Strata
AVX2/AVX-VNNI kernels for Q2_0, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS, Q4_K, Q5_1,
and Q8_0. IQ1_M and the remaining K-quants stay on the ggml-cpu dot. The Strix Core Ultra
9 275HX has AVX2 and AVX-VNNI but no AVX-512 or AMX, so its relevant fast path
is the AVX2/AVX-VNNI family, not the AVX-512 kernels.

`tools/quantscope.py` and `tools/gguf_reader.py` can identify additional GGUF
types such as IQ1_S, TQ1_0, TQ2_0, MXFP4, and NVFP4. Recognition is not
execution support; the normal engine and setup choices above remain the
executable set. An NVFP4 research fork is referenced by inherited benchmarks
but is not a current setup path.

The first implementation seam for substantially smaller routed experts is
experimental, not a default-path replacement: add the pack type in
`tools/strata_pack.py` with an exact inverse, preserve mixed native slices in
`tools/iq_pack.py`, then extend CPU dispatch in
`include/strata/kernels/cpu/native_expert.hpp`,
`src/kernels/cpu/native_expert.cpp`, and `src/kernels/cpu/iq_avx2*.cpp`, plus
the corresponding CUDA/HIP implementations under `src/kernels/` and
`src/prefill/`. Acceptance starts with the existing parity binaries
(`iq_avx2_parity`, `q2_bitplane_parity`, and backend parity), then adds bounded
reference differences and a real-model output comparison. A proposal that uses
the NPU must separately prove a representative expert-shaped operation through
Level Zero/OpenVINO before any Strata routing claim.
Calibration-source identities, exact-token handling, and resume behavior are in
[calibration](CALIBRATION.md).

## Open choices

| Topic | Current position | Decision gate |
| --- | --- | --- |
| Common quality mode | `just quality` checks host-safe structure, README/docs navigation, selected regressions, the plugin ABI, dependency-free DeepSeek host contracts, the trace-derived REAP and prefiller-planner tests | Add a check only with a test and an explicit non-claim |
| Full setup matrix | Existing full discovery is not accepted as a gate on macOS Python 3.14 because native build and upstream-release cases fail | A dedicated setup change runs its named modules and the affected platform smoke test |
| Backend acceptance | No local backend build is claimed on this Mac | Require named compiler/driver/backend evidence on the target host |
| Eight-GB NVIDIA runtime | Documentation and setup accept an 8 GB NVIDIA card as slow; an sm_120 engine built and launched on the Strix laptop | Require a real model run and same-day performance measurement before claiming usable throughput |
| Strix Intel NPU integration | llama.cpp's whole qwen4exp IQ graph fails on unsupported IQ matmuls plus an NPU layout mismatch, but the explicit Strata plugin ABI now runs a representative one-token `2560x512` router on `Intel(R) AI Boost` through OpenVINO, translating BF16 ABI tensors to F16 because this driver rejects BF16 inference precision; the model engine does not load the plugin | Integration requires explicit opt-in CLI/config, real-model parity and failure tests, plus same-day CPU/CUDA/NPU latency and transfer accounting |
| Strix heterogeneous routing | NVIDIA dGPU, Intel NPU, and CPU are independently visible; Strata currently routes only its existing CUDA/CPU paths | A routing proposal needs per-part correctness, scheduling, and same-day throughput/latency controls before it can replace a default |
| iMatrix low-bit routed experts | The cross-host IQ2_XS difference is isolated to the derived-weight square-root expression before codebook selection; `STRATA_DETERMINISTIC_IQ=1` applies the validated double-precision expression to downloaded llama.cpp source while the default pinned path stays unchanged; an exact-token path now consumes the RESMP-DEV standard `[256,2048]` Qwen view | A full standard-calibrated patched rerun, perplexity/output quality, and same-day throughput/memory controls remain required before replacing a normal quantization choice |
| DeepSeek V4.1 backend order | CUDA/CPU extraction on B550 is the initial implementation path; Lucebox HIP is reference-only | A HIP-first path is justified only if an AMD target becomes primary or an unmodified Lucebox oracle is required; either way, named AMD hardware and parity evidence are mandatory |
| DeepSeek V4.1 artifact source | Start from the pinned official 1017 checkpoint already transferring to B550, then quantize through an RESMP-owned tool | Do not promote the Lucebox ROCMFP artifact as a Strata runtime dependency; a different source or borrowed quantized artifact needs an explicit dated architecture decision |
| Speculative-head/QAT direction | The native one-layer Qwen MTP is the immediate executable QAT lane; a DSpark-style rank-256 Markov plus confidence head over the promoted DFlash2 G3 full head is the next research architecture | A candidate needs exact target-controlled parity and improved paired live acceptance before integration; ordinary KL, offline replay, compression, or throughput alone is not acceptance |
| GSQ/RCO expert-pruned Qwen variant | The IST-DASLab Coder release (256 of 512 experts per layer, 3.5 bpw retained, 1.89 bpw effective) is registered in `setup.py` as the `coder` family at pinned revision `5348543e` and consumes `--n-expert 256` in `tools/make_profile.py`; a Strata-owned prune, heal, quantize, verify variant is unbuilt except for the host-safe input/runtime preflight, `tools/reap_mask_from_traces.py`, and the tiny Qwen4-Exp/RCO CPU plus Modal-L4 compatibility probes. GSQ and RCO are Apache-2.0 tools, not upstreams, so consuming them does not touch the no-upstream rule | A candidate needs a confirmed GSQ or self-written Flash-Next wrapper, a REAP-from-trace control arm, held-out KL and a heal-over-unhealed delta, bit-exact `strata-pack verify`, and a same-day runtime control against the vendor artifact on named hardware |
| Flash-Next pruning prototype | Use the current vendor Coder GGUF only as a measured as-is control while designing a Strata-owned 256/288/384-expert higher-capacity artifact; the first full-model runtime sweep and 48-layer decode routing trace are recorded in [pruning](PRUNING.md) | Promote only with current shard hashes, representative prompt+decode traces, held-out quality, exact pack inversion, and same-day full-model controls |
| Flash-Next auxiliary prefiller | Target is a Qwen3.8-Flash-Next prompt-state model trained against the frozen served low-bit decoder; MXFP4 is acceptable and the dense 27B NVFP4 artifact is reference-only. The metadata planner and architecture are implemented host-side, and opt-in literal batched `Prefill::run` STRP v2 capture is implemented and exercised on B550; training, artifact generation, and runtime swapping remain unbuilt | Start with dense/shared adapters plus measured prompt-and-decode hot experts; require phase-masked QADD, GDN/QSA/PLE handoff parity, default-path byte identity when disabled, and many-agent mixed-hit controls before any runtime or quality claim |
| Disaggregated KV tier | Target design is [KV disaggregation](KV_DISAGGREGATION.md): a coordinator owns L0 VRAM, L1 pinned host slots, L2 local fast NVMe groups (B550's HPE P4800X), and optional L3 shared storage through one `KvBlockStore` ABI. The current `--kv-disk` prototype is quarantined experimental code, not serving-path acceptance | Phase A must move append/read hooks into the actual verify path and support all KV formats; live acceptance requires zero argmax mismatches, mean NLL within `1e-4`, nonzero forced cold reads/evictions, then 600K and 800K matched controls |
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
- A future model port does not become a runtime dependency on somebody else's
  prequantized artifact: Strata's executable pack must be generated by an
  RESMP-owned quantizer with an exact inverse and recorded source hash.
- Documentation distinguishes measured values from hypotheses and names the
  hardware, model, revision, workload, and limits.
- The local server is loopback by default; wider exposure requires an API key
  and explicit user intent.
- Receipts bind evidence to revision and scope, state dirty provenance, and say
  what they do not prove.

## Project status and governance

This is the detailed implementation/exercise matrix behind the compact README
entry status. Claims remain bound to the linked evidence and do not transfer to
later revisions.

| Area | Current status |
| --- | --- |
| Engine, installer, local API, web app | Implemented and documented by the inherited upstream code at `fb58e0d`; not rebuilt or measured locally by the governance change on 2026-10-09 |
| Local project governance | Lapis-style authority structure in place: [AGENTS.md](../AGENTS.md), [architecture](architecture.md), [CONTRIBUTING.md](../CONTRIBUTING.md), and [stand-up evidence](../evidence/project-standup.json) |
| Common verification | `just quality` runs on this macOS host; it does not claim a CUDA, HIP, or SYCL build |
| Strix hardware evidence | An 8 GB RTX 5070 engine build launched; the Intel NPU also passed an external, opt-in one-token router probe through the new plugin ABI. Neither is a Strata model-run or speed claim. See [accelerator evidence](../evidence/accelerator-plugin-router-probe.json) |
| DeepSeek V4.1 port preparation | Exact 1017 checkpoint metadata taxonomy, model geometry, and deterministic CPU candidate selection are implemented and exercised host-side on both this machine and the B550 compute host. B550 also built the regular Qwen engine and passed its AVX2 parity targets; its DeepSeek-specific AVX2 ceiling is measured but no DeepSeek pack or backend exists. See [first host contracts](../evidence/deepseek-v41-first-host-contracts.json), [B550 bring-up](../evidence/deepseek-v41-b550-bringup.json), and [AVX2 ceiling](../evidence/deepseek-v41-b550-avx2-ceiling.json) |
| Qwen B550 storage placement | The HDD source store sustains 202 MiB/s sequential reads but only 106 MiB/s for Strata-sized random expert-slot misses; the root NVMe is 10.8x faster at queue depth 1 and 18.3x at depth 8. Stage runnable Qwen packs on NVMe scratch. See [storage baseline](../evidence/qwen-b550-storage-read-baseline.json) |
| Qwen B550 calibrated IQ2_XS execution | The staged split GGUF now has a shard-aware pack, the model loaded and executed on both cards and through the two-GPU server, and the missing Q2_0 MTP drafter was fetched, verified, packed, and exercised. First observations put two GPUs around 1.9x GPU 1 alone; with the embedding replicas stopped, all 24,576 experts fit resident and a five-request short-prompt control reached 227.4 tok/s median decode. Broader performance and quality remain unclaimed. See [B550 baseline](../evidence/qwen-b550-iq2xs-bringup-and-mtp-baseline.json) and [clean control](../evidence/qwen-b550-full-vram-clean-control.json) |
| Speculative-head/QAT target research | The next high-upside research head is DSpark-style DFlash2 (rank-256 Markov plus calibrated confidence), while the already-running native Qwen MTP is the first practical QAT lane. No QAT backend or trained artifact is claimed yet. See [head research](../evidence/qwen-dspark-mtp-qat-target-research.json) |
| DSpark2 Qwen 27B parameter plan | The local 11,273-cycle DFlash2 corpus, prior G2/G3 searches, official DeepSpec defaults, and a live B550 smoke define a staged block-7 Markov/confidence training plan. Existing top-64 captures cannot train the official TV/confidence loss unchanged. See [parameter evidence](../evidence/dspark2-qwen27b-training-parameters.json) |
| DSpark2-Qwen full-logit distillation | The MLX DFlash2 prediction-head template now captures full-vocab F32 logits and trains DSpark-weighted hybrid loss (CE 0.1 / TV 0.9) with an MXFP4 QAT path. Prime RTX 6000 Ada is the selected lane: a real A100 80GB control was 1.53x slower and 2.51x costlier per step. This trains the Qwen3.8-27B prototype, not Flash-Next, and no candidate is promoted or live-accepted. See [cloud training state](../evidence/dspark2-qwen-cloud-training-state.json) and [GPU selection](../evidence/dspark2-gpu-provider-selection.json) |
| Qwen B550 concurrency and embedding tiers | Strata serves concurrent agent requests via `--batch` slots; measured on one short coding-prompt family over 3 rounds, MTP activity inside concurrent batch slots is unconfirmed. On this launch-bound workload concurrency buys latency fairness, not aggregate throughput (two-GPU full-residency: 149.6/126.3/158.1 tok/s aggregate at C=1/2/4; per-request 199.8 to 49.7). CPU-only embedding of the small Qwen3-0.6B is reasonable for light use (0.020 s single-request, faster than the 0.081 s GPU round-trip) but 4.7x slower on bulk (38.8 vs 183.1 items/s); the sindexer MXFP4 Jina embedder is CUDA-only, not a CPU path. See [concurrency evidence](../evidence/qwen-b550-concurrency-fallback.json) |
| Strix MXFP4 embedding tier | The sindexer Jina MXFP4 embedder runs on Strix's RTX 5070 Laptop (SM120): the Triton MXFP4 kernel executes there, and the 1536-dim output matches the B550 reference at cosine 0.9999. It is the low-latency 1536-dimensional MXFP4 tier (0.036 s single-request versus 0.081 s on B550) but far weaker on bulk (15.9 versus 183.1 items/s). It runs standalone on :1236 and is not in the sindexer routing path; the NPU path is untested. See [Strix tier evidence](../evidence/qwen-strix-mxfp4-embed-tier.json) |
| B550 layer-split topology | B550's GPU0 is chipset-attached at Gen3 x4 (2.9 GB/s) while GPU1 is CPU-root Gen4 x16 (25.4 GB/s). On a verified-clean GPU, an explicit `--layer-split 30` with the fast card first measured 236.0 to 243.6 tok/s decode against the auto control's 226.9 to 227.3 (about 4 to 7%), and 165.9 versus 151.8 tok/s prompt. Reversing card order alone gains about 1%, and `STRATA_SPLIT_MISS_MS` does not move the placement. No deployed configuration was changed. See [topology evidence](../evidence/qwen-b550-layer-split-topology.json) |
| B550 two-request serving policy | C=2 batch slots lose aggregate throughput even with topology-aware groups: 127.0 versus 142.7 tok/s at C=1, and split-30 falls to 106.3. Independent per-GPU servers and a shared CPU arena are worse. For the measured exact short-prompt shape, queue serially on split-30: ten requests reached 222.5 tok/s aggregate and about 252.6 tok/s per request; varied-prompt serial C=2 and longer prompts remain unmeasured. No deployed routing tier was changed. See [queueing evidence](../evidence/qwen-b550-c2-queue-topology.json) |
| B550 long-session serving | At 256K context with KV streaming (32,768 hot cells per QSA layer in VRAM, the engine's measured 1.80 GiB pinned host allocation, against a 3.09 GiB full-context payload upper bound), a 112,626-token real-source prompt prefilled in 27.743 s (4059.5 tok/s) and decoded at 117.1 tok/s with a 99.5% expert-cache hit rate and 92% of KV reads from VRAM. RAM is not the constraint: `--ple-io ram` was 2.5x slower to prefill and 1.9x slower to decode than direct SSD reads on this 91 GiB host. See [long-session evidence](../evidence/qwen-b550-longrun-256k.json) |
| B550 fast-NVMe KV sessions | Native session files can park and restore whole long conversations on the P4800X: a real 112,626-token state became a 1.956 GB file in 1.463 s and restored into a restarted single-GPU process in 2.232 s. This is explicit session parking, not live per-token NVMe paging; layer-split, peer-device, and batch modes remain incompatible, and Mooncake is reserved for a future multi-host shared pool. See [session-store evidence](../evidence/qwen-b550-optane-session-store.json) |
| Disaggregated KV tier | The opt-in `--kv-tier` path owns per-layer stores, per-layer selections, explicit background store operations, a Mooncake-style shared pool reference, and restore into the engine host copy consumed by `kv_stream_resolve`. On B550 it now runs against a real model on the designated 750 GB device: a matched control with the tier off and on produced bit-identical output and a mean NLL delta of exactly 0.0 over 8,191 scored positions, with 756,560 real cold reads, 794,873 engine host restores and zero checksum errors. The tier was about 14.8x slower on the prompt read for that workload, the full pinned host allocation remains resident, and no RAM or speed benefit is claimed. See [KV disaggregation](KV_DISAGGREGATION.md), [cold-tier evidence](../evidence/b550-optane-cold-kv-2026-10-20.json), and [sm86 build evidence](../evidence/b550-sm86-build-2026-10-20.json) |
| DeepInfer portable scheduling intake | The fetch-only DeepInfer audit at `774ce244` ranked confidence-scheduled variable-length MTP, zero-overhead scheduling, incremental metadata assembly, and sync guards above model-specific Engram/Hopper/Ascend work. The first host-only confidence/varlen draft-row planner is implemented and passes focused tests on macOS and in the B550 sm_86 tree; it is not wired into runtime MTP and claims no speedup. See [audit evidence](../evidence/deepinfer-portable-optimizations-2026-10-20.json) and [implementation evidence](../evidence/deepinfer-confidence-varlen-2026-10-20.json) |
| Flash-Next auxiliary prefiller | The 27B NVFP4 checkpoint and paper are templates only. A host-only planner now binds the deployed Flash-Next pack metadata, accepts 256/288/384/512 physical expert packs through validated explicit physical-to-source maps (with identity only for a complete 512-slot pack), selects dense prompt-state tensors plus per-layer hot experts in parallel physical/source namespaces, and consumes legacy or batched-prefill STRP v2 routing traces. A real 13,963-position literal prefill trace gives 64 experts/layer 60.8184% aggregate router-weight coverage and 256 experts/layer 94.0132%. No trainer, runtime prefiller, or quality/speed effect is claimed. See [architecture](PREFILLER.md), [planning evidence](../evidence/flash-next-prefiller-plan-2026-10-20.json), and [runtime acceptance](../evidence/prefill-routing-v2-runtime-2026-10-21.json) |
| Flash-Next pruning baseline | A real-source 13.96K-token B550 sweep measured 3,355.9 tok/s median cold prompt reads and 99.9211% decode expert-cache hits; a held 13,943-token prefix cut median fresh-read time from 4,159.5 ms to 192 ms. A clean-main refresh at `66067e30` measured 3,491.6 tok/s steady cold prompt reads and reduced a 47-token shared-prefix fresh read to about 216 ms. A 48-layer decode trace shows 64 retained experts/layer cover 92.08% of routed activations in-sample. The current vendor Coder prototype is 9.7% faster on cold prompt reads and uses an 8.81 GiB smaller arena, but is 13.6% slower on the primed concurrent-4 shared-prefix burst; those Coder deltas are against the same-day 2026-10-20 control, not the clean-main refresh. See [pruning](PRUNING.md), [full-model evidence](../evidence/flash-next-agent-runtime-stats-2026-10-20.json), [clean-main evidence](../evidence/flash-next-current-main-runtime-2026-10-21.json), and [Coder evidence](../evidence/coder-current-runtime-stats-2026-10-20.json) |
| Flash-Next prompt routing diagnostic | The 2026-10-20 forced-short-read B550 capture recorded 14,020 prompt positions per layer with unit activations; 64 experts/layer covered 55.61% and 256 covered 92.31%. Its direct normal-prefill negative control exposed the missing literal seam at that revision. The opt-in STRP v2 seam and real literal capture now close that diagnostic gap without changing the default path. See [prompt-routing evidence](../evidence/flash-next-prompt-routing-2026-10-20.json) and [runtime acceptance](../evidence/prefill-routing-v2-runtime-2026-10-21.json) |
| Pruning/quantization publication prep | A model-card and artifact-file contract now defines the future private Hugging Face release: provenance, hashes, licenses, physical/source pruning map, quantization allocation, evaluation, and runtime evidence are mandatory before promotion. No pruned or custom-quantized artifact exists yet. See [model cards](MODEL_CARDS.md) |

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

### Strix VRAM boundary, 2026-10-09

The SSH host `strix` is a ROG Strix G16 G615LP at 192.168.1.182 (although its
reported hostname remains `compute`) with an 8 GB RTX 5070 Laptop GPU, 91 GB
RAM, CUDA 13.4, and driver 595.91.07. `setup.py --check` accepts it while
warning that most experts stay on the CPU. Source review confirms that a
supported single GPU has no hard 12 GB gate; 12 GB is the documented
performance recommendation. The separate 8 GB requirement applies to a card
sharing a model, and `SPLIT_PROMPT_VRAM_GB = 12` is only a split-stage prompt
buffer recommendation. The engine built 135/135 for sm_120 and its binary
launched `--help`, but no real-model inference was measured. Full evidence and
side effects are in [Strix VRAM boundary evidence](../evidence/strix-vram-boundary.json).

### Remote access, Intel NPU/CPU, and quantization inventory, 2026-10-09

The two SSH execution aliases were verified live and are now contractual in
`AGENTS.md`: `B550`/`compute` at 192.168.1.171 is the dual-RTX-3090-Ti heavy
GPU host, while `strix` at 192.168.1.182 is the heterogeneous validation host.
Both report hostname `compute`, so aliases, not hostnames, select them. The
main machine remains the authority.

On Strix, the Arrow Lake NPU was already bound to `intel_vpu` at PCI
`8086:ad1d` with `/dev/accel/accel0`, but a plain SSH session was outside the
seat-based udev ACL. Intel's signed v1.38.0 Ubuntu 26.04 user-space stack and
Level Zero loader 1.32.0 were installed after archive SHA-256 and per-package
GPG verification, and `kearm` was added to the documented `render` group. A
fresh SSH session can open the device read/write. OpenVINO 2026.3.1 in an
isolated Python 3.12 environment enumerates CPU, the NVIDIA dGPU, and
`Intel(R) AI Boost`, and a small elementwise graph compiled and executed on the
NPU with the expected output. The host currently booted the February 2026
firmware even though the August 2026 package is installed on disk; no reboot
was forced. Its CPU has AVX2 and AVX-VNNI, but no AVX-512 or AMX. The source
inventory and research gates above were derived from `setup.py`, the pack
tools, and CPU/GPU format dispatch. Full commands, package identities,
signatures, side effects, and non-claims are in
[Strix accelerator access evidence](../evidence/strix-accelerator-access.json).

### iMatrix expert mix and llama.cpp NPU boundary, 2026-10-09

A pinned public iMatrix for this exact model (319 chunks of 512 tokens, 902
tensor entries, and per-expert coverage for all but 0.33% of expert rows) was
used to requantize the routed experts of the 98.4 GB IQ4_XS source. Gate and
up experts became IQ2_XS; down experts became Q2_0 because their 640-value
inner dimension is not a whole IQ2_S block. Every non-target tensor kept its
source type. The result is 66,649,935,616 bytes (llama.cpp reports 63,551.82
MiB of quantized model data, 3.01 BPW), and its 48 gate, 48 up, and 48 down
expert tensors passed exact type checks. The standalone GGUF retains F32
routers because llama.cpp hard-skips `ffn_gate_inp` before manual overrides;
the Strata pack then converts all 96 router tensors to exact BF16. The pack
maps 34,603,008,000 bytes of native experts.

The experimental artifact was transferred to Strix with matching SHA-256 and
loaded by engine 0.1.41 with the shipped expert profile. The engine loaded
32.23 GiB of experts at 18.26 GiB/s, selected 2,233 expert-cache slots on the
8 GB RTX 5070, completed a three-token prompt, and emitted greedy token 198.
That one-token run measured 24.78 tokens/s decode and 5.47 tokens/s prefill,
but it does not establish general throughput or quality.

The same full quantization was then rerun on Strix from a hash-verified local
copy of the IQ4_XS source. It passed in 1,713.87 seconds with 29,314,752 KiB
peak RSS, kept the same size and every target tensor type, and loaded through
the same pack. It produced a different whole-file SHA-256 and greedy token
(220 rather than 198), so parallel IQ quantization is not currently
byte-reproducible across the two hosts. The Strix-native run measured 25.88
tokens/s decode and 6.66 tokens/s prefill for its one-token prompt.

llama.cpp's pinned `ggml-openvino` backend is present even though Strata does
not link it. Forced to NPU, it executes a tiny F32 llama model and produces
the same text as CPU. On the real qwen4exp artifact it assigns all layers to
OPENVINO0 but reports thousands of unsupported IQ4_XS/IQ4_NL MUL_MAT and
IQ2_XS/Q2_0 MUL_MAT_ID operations, leaves most weights CPU-mapped, splits the
graph 660 ways, and finally fails in Intel's NPU infer request with input
shape `[1,1,3,2560]` where the compiled graph expects `[1,3,1,2560]`. Thus
the obstacle is not that NPUs cannot run graphs in general; this specific
llama.cpp/OpenVINO path does not run the target quantized qwen4exp graph
correctly. Full hashes, sampled reconstruction errors, commands, timings,
and non-claims are in
[IQ2/iMatrix and NPU evidence](../evidence/iq2-imatrix-npu-experiment.json).

### IQ2_XS reproducibility and accelerator plugin seam, 2026-10-10

The whole-artifact and per-tensor comparison isolated the previous cross-host
difference to IQ2_XS gate/up tensors; every Q2_0 `ffn_down_exps` tensor matched.
With bit-identical layer-0 gate-expert-0 input and iMatrix data, both hosts were
internally repeatable but produced different IQ2_XS bytes. Equal block variances
and unequal derived `w = qw * sqrtf(sigma2 + x*x)` bit patterns place the root
cause in host single-precision square-root/contraction arithmetic before codebook
selection. Promoting that add, square, and square root to double precision while
compiling the quantizer with `-ffp-contract=off` made Apple Clang and Strix GCC
produce the same 640-row expert bytes,
`e5b46bc8dbee76ad6c6975755436c77ee55be859b9d08f2d4fa9a4c0c19ffe2a`. Setup now
applies this only when `STRATA_DETERMINISTIC_IQ=1`, refuses a moved upstream
expression, and leaves the default third-party path unchanged. Full quantization
and quality controls were not rerun. Full hashes and non-claims are in
[IQ2 reproducibility evidence](../evidence/iq2-xs-reproducibility.json).

K-Transformers at `6160538c81fc1d2cbc1952302a6d4b77a8adcd2e` informed the seam
shape without being imported: its archived YAML optimizer imports an explicit
replacement class for each matched module, while current KT-Kernel validates a
method name and maps it to a specific Python wrapper/backend. Strata's analogous
first boundary is a C ABI for one router operation. The loader/reference test
passed on macOS and Linux, and an OpenVINO plugin compiled on Strix. The first
NPU attempt exposed two real contracts: this driver accepts F16/I8 inference
precision but rejects BF16, and its compiler requires a static one-token shape
rather than an unbounded dynamic token dimension. The BF16-to-F16 plugin then
passed a representative `2560x512`, one-token router probe with max absolute
error `4.5975e-09`; execution was `25.189 ms`, correctness evidence rather than
a performance claim. No model-generation path loads a plugin yet. Commands and
outputs are in
[accelerator plugin evidence](../evidence/accelerator-plugin-router-probe.json).
Operational build and probe instructions live in
[accelerator plugins](ACCELERATOR_PLUGINS.md).

### Standard calibration corpus and exact-token iMatrix, 2026-10-11

The earlier 3.01-BPW experiment consumed a public mradermacher matrix generated
from `imatrix-training-full-3` (319 chunks of 512 tokens), not the RESMP-DEV
standard corpus. The standard current Qwen3.8 view is
`RESMP-DEV/ptq-calibration-corpus@6ef2bbf...`, tokenized by the pinned
Qwen3.8-27B tokenizer and represented as 256 seeded, disjoint windows of 2,048
IDs (524,288 tokens). Its token and merge tables match Flash-Next; the GGUF's
extra tokenizer rows are explicit padding.

Because raw-text `-f` would retokenize a contiguous stream and lose the
standard view's selected windows, `STRATA_TOKEN_IMATRIX=1` now adds an opt-in
`llama-imatrix --token-file` input for nonnegative little-endian int32 IDs.
The helper downloads and hash-verifies the view, exports
`calibration_256x2048.i32`, validates the target tokenizer, builds the patched
tool, checkpoints every ten chunks, and resumes from
`imatrix.chunk_count`. Disabling the gate restores the pinned llama.cpp source.

A real full run over the IQ4_XS Flash-Next GGUF completed all 256 standard
windows on the Apple-silicon main machine. The resulting importance matrix is
cached as
`~/.cache/strata-quant/standard-imatrix/Qwen3.8-Flash-Next.standard-256x2048.imatrix.gguf`,
SHA-256 `92c6a742027b569907a2ce3cd2bf12b3ae5ea037c86105ef7ba4ae244a43fc89`,
with 902 count tensors and 144 routed-expert tensors. Thirty-three expert
count rows remain zero. Strix was intended for this run, but SSH key exchange
repeatedly reset and its NVIDIA NVML reported a driver/library mismatch, so no
remote execution or reboot was claimed. Full identities, commands, checkpoint
hashes, and non-claims are in
[standard calibration evidence](../evidence/standard-calibration-token-imatrix.json);
operational details are in [calibration](CALIBRATION.md).

### Strix I/O failure and persistent host watcher, 2026-10-13

Strix returned after a manual reboot, but a fresh hash of its 92 GB IQ4_XS
source advanced an approximately 43 GiB read, failed with `Input/output error`,
and the prior boot ended abruptly. The host came back about two and a half
minutes later; journald reported both system and user journals corrupted or
uncleanly shut down. The persisted prior-boot kernel log contains no NVMe or
EXT4 error before the cutoff.

A root `strata-watch.service` is now installed and enabled on Strix. It samples
PSI, NVMe counters, process/SSH state, service transitions, NVIDIA procfs, and
filtered journald messages every 20 seconds, writes a one-minute heartbeat, and
bounds event storage to 30 days or 512 MiB. It survived the observed reboot.
Its first design called `nvidia-smi`, which entered an uninterruptible state and
blocked past an eight-second timeout; the final watcher never invokes that
command.

SMART is available and currently passes: the CT4000T700SSD3 reports critical
warning 0, 100% spare, 3% endurance used, 46 C, zero media errors, and zero
error-log entries. Historical short and extended self-tests passed. This means
the drive health counters are clean, not that the Linux I/O path is healthy.
The failing source file was not retried and no Strix calibration run was
started. Full sequence and SMART values are in
[Strix I/O watcher evidence](../evidence/strix-io-watch.json); inspection
commands are in [Strix watcher](STRIX_WATCH.md).

### Direct-read stress reproduced the Strix wedge, 2026-10-13

After the clean SMART snapshot, a bounded 1 GiB direct read around the previous
42 GiB boundary passed. A detached transient systemd service then began two
sequential passes over the 98.4 GB source GGUF, using 1 GiB aligned direct-read
checkpoints, per-chunk hashes, and a 180-second chunk timeout. The first 29 GiB
completed at roughly one GiB per three seconds with zero recorded errors, zero
D-state processes, zero measured I/O pressure, and zero inflight NVMe commands
at the last retrieved sample.

Soon after that sample, SSH began resetting every pre-auth connection while
ICMP and TCP port 22 continued answering. The condition persisted for more
than eleven minutes. This reproduces a host/storage access wedge under the
read workload, but the final checkpoint and kernel evidence remain on Strix
until console or SSH recovery. No write test was run and the source remains
quarantined. Details are in
[direct-read stress evidence](../evidence/strix-direct-read-stress.json).

### Raw NVMe test identified the thermal pathway, 2026-10-13

After recovery, the watcher was extended to record nonblocking NVMe/board
temperatures and battery/AC state. A 64 GiB raw `/dev/nvme0n1` direct-read test
then bypassed ext4 and the source GGUF entirely. It completed at least 60 GiB,
but the physical signature was clear: NVMe temperature rose from 26.85 °C to
85.85 °C, per-GiB time increased from 3 to 12 seconds, full I/O PSI rose to
54.35%, and SSH then reproduced its pre-auth reset. The drive's warning and
critical thresholds are 87 °C and 89 °C.

SMART has remained clean with zero media errors and zero controller error-log
entries. The leading diagnosis is therefore sustained-read thermal
throttling/controller or platform protection, not GGUF/ext4 damage or SSH load.
Insufficient Gen5-SSD cooling/contact or a platform power response remains the
likely physical cause. Details are in
[raw NVMe thermal evidence](../evidence/strix-raw-nvme-thermal-reset.json).

### Desktop-size NVMe heatsink removed the thermal boundary, 2026-10-13

A desktop-size heatsink was then installed on the T700 while leaving the
platform profile at `balanced`. The same detached 64 GiB raw `/dev/nvme0n1`
direct-read protocol completed all 64 checkpoints in 109 seconds. Every chunk
took the recorded one-second interval, all SHA-256 values were present, and no
checkpoint returned nonzero. Watcher samples peaked at 40.85 °C, full I/O PSI
peaked at 0.69%, and both D-state and NVMe inflight counts stayed zero. SMART
remained clean and the warning-priority kernel window was empty.

Compared with the failing run, peak observed NVMe temperature fell by 45.00 °C,
throughput improved from 3-12 seconds per GiB to one second per GiB, and the
host no longer reset. This makes inadequate NVMe cooling/contact/airflow the
dominant physical cause of the immediate wedge. The exact firmware, controller,
or EC protection action remains unobserved. Details are in
[heatsink mitigation evidence](../evidence/strix-heatsink-mitigation.json).

### Strix source GGUF requalified, 2026-10-13

After the raw-drive mitigation result, the previously quarantined
98,421,788,160-byte IQ4_XS source GGUF was read twice with its normal detached
checkpoint protocol. Both passes covered the same 92 logical offsets, including
the 711,282,176-byte EOF chunk; all exit codes were zero, stderr stayed empty,
and every cross-pass hash matched. The watcher sampled a 49.85 °C NVMe peak
with at most one D-state process and zero inflight commands at its sample
boundaries. A detached full-file hash then produced the pinned SHA-256
`ea9b7537389918d69bcf3ea1fb5693d427ae82278dd67f415fa9383858b51efb` in
141.13 seconds. SMART remained clean. This requalifies the source bytes and
read path for quantization work; it makes no model-quality claim. Details are in
[source requalification evidence](../evidence/strix-source-gguf-requalified.json).

### Strix ZeroTier upload-path diagnosis, 2026-10-13

After the storage boundary was fixed, three Mac-to-Strix bulk attempts over the
direct ZeroTier path timed out while Strix stayed on the same boot, cool, and
without D-state processes. A disk-free `/dev/zero` to `/dev/null` reproduction
failed identically. A 64 KiB/s SSH stream completed for 40 seconds, while 128
and 256 KiB/s streams timed out. Packet capture showed successful SSH
authentication, about 334 KiB of in-order stream data accepted by Strix, then a
retransmission/SACK storm. The encrypted UDP path was asymmetric: the selected
forward flow used one NAT endpoint pair and ACKs returned through another.
ZeroTier's selected IPv4 endpoints changed during the incident and Strix
subsequently probed nine endpoint combinations.

The leading diagnosis is rate-sensitive NAT/path renegotiation or packet-rate
policing on the Mac-to-Strix IPv4 path, not Strix storage or sshd. The exact
actor remains unknown. The same 579,789,056-byte standard iMatrix transferred
successfully and hash-verified in 86 seconds through the B550 jump and
B550-to-Strix LAN. Sustained main-machine uploads to Strix therefore use that
relay until the direct path is repaired. Details are in
[ZeroTier path evidence](../evidence/strix-zerotier-upload-path-instability.json).

### Deterministic standard-calibrated IQ2_XS artifact, 2026-10-13

After the source and matrix were hash-verified on Strix, a complete 1,224-entry
anchored COPY recipe quantized 96 gate/up expert tensors to deterministic IQ2_XS
and 48 down tensors to Q2_0 while preserving every other tensor type. The full
run finished in 27:48.85 with 24 threads and 29.33 GiB peak RSS. It produced
66,649,935,712 bytes with SHA-256
`f88ddbabfe3a7c307c4685ae524efddcceada6e5b427d0d5aa574a53a767cc08`; all 1,224
tensor names and shapes matched, exactly the 144 intended type conversions
occurred, and SMART stayed clean. The standard matrix adds GGUF metadata, making
this file 96 bytes larger than the old public-matrix artifact despite identical
tensor contracts.

The tracked `STRATA_EXACT_COPY` and `STRATA_DETERMINIC_IQ` gates rebuilt the
quantizer and repeated the dry-run contract exactly. `iq_pack.py --compat-bf16`
then produced the 1.38-GiB dense arena and tokenizer. The real engine loaded
32.23 GiB of experts at 6.93 GiB/s, selected 2,233 cache slots, and generated
token 157 at 26.33 tok/s decode and 6.83 tok/s prefill. A same-boot control on
the prior public-matrix artifact used the same engine and prompt, generated the
same token, and measured 24.96 tok/s decode and 5.45 tok/s prefill. This is a
one-token execution gate, not a quality or general-performance claim. Full
commands, hashes, pack identities, and non-claims are in
[deterministic standard IQ2_XS evidence](../evidence/standard-iq2xs-deterministic.json).

### Hugging Face upload preparation and storage gate, 2026-10-15

The deterministic artifact is a static per-tensor requantization, not runtime
dynamic quantization. A private Hugging Face destination was created as
`Kearm/Qwen3.8-Flash-Next-IQ2_XS-Standard-iMatrix`; its model card and manifest
were uploaded and round-trip verified at commit
`fd601ac4fe7f860980872f090f4f075299a49ce1`. The Hub rejected the single
62.08-GiB object because this account's individual-file limit is 50 GB. The
artifact was therefore split with llama.cpp into a 39.97-GiB, 344-tensor shard
and a 26.68-GiB, 880-tensor shard. Both shard hashes are recorded, and all
1,224 tensor payloads survived merge verification without mismatch.

The shard LFS upload then stopped before transferring bytes: the Hub returned
403, `Private storage limit reached for user Kearm`. The Hugging Face token
stayed on the main machine. Increasing Kearm private storage or explicitly
selecting another private destination is the remaining action; a public upload
is not permitted. Details are in
[Hugging Face upload evidence](../evidence/huggingface-standard-iq2xs-upload.json).

### Public RESMP-DEV publication, 2026-10-16

The user then made the destination policy explicit: future model uploads go to
the RESMP-DEV Hugging Face organization, and this research artifact was
explicitly authorized as public. The better-named public repository is
[RESMP-DEV/Qwen3.8-Flash-Next-IQ2_XS-Q2_0-Standard-iMatrix-GGUF](https://huggingface.co/RESMP-DEV/Qwen3.8-Flash-Next-IQ2_XS-Q2_0-Standard-iMatrix-GGUF).
Its model card follows the organization's existing GGUF format and states the
essential boundary: this is a static, compounded IQ4_XS-to-IQ2_XS/Q2_0 research
requantization, not runtime dynamic quantization or a quality-equivalent BF16
conversion, and only a one-token execution gate has run.

The main machine issued short-lived multipart LFS actions while Strix uploaded
2,000 presigned parts and persisted every ETag; no Hugging Face token was placed
on Strix. Both 1,000-part shards completed, and the final public LFS pointers
were verified with HTTP 206 logical ranges and linked SHA-256 etags. The
obsolete metadata-only personal destination was deleted after confirming it
contained no shards. Details and cleanup are in
[publication evidence](../evidence/huggingface-standard-iq2xs-publication.json).

### Public naming clarification, 2026-10-17

The word `Standard` in the first public repository name meant the internal
RESMP-DEV calibration standard, not a generic or default GGUF quantization.
That was confusing to external users. The repository and files were renamed to
make the actual meaning explicit:
[RESMP-DEV/Qwen3.8-Flash-Next-IQ2_XS-Q2_0-iMatrix-Calibrated-256x2048-GGUF](https://huggingface.co/RESMP-DEV/Qwen3.8-Flash-Next-IQ2_XS-Q2_0-iMatrix-Calibrated-256x2048-GGUF).
The old repository URL redirects. Both LFS shard OIDs and byte counts are
unchanged; this was a naming and model-card clarification only. Details are in
[naming evidence](../evidence/huggingface-imatrix-naming-clarification.json).

### Strix M.2 retainer part research, 2026-10-17

The unusable Strix bay uses ASUS's 2025 ROG tool-less rotating cam SSD latch:
official G635/G615 service documentation calls component 25 an `SSD latch`,
says to rotate the latch knobs counterclockwise and clockwise, and shows the
same left/right mechanism for G615LP and G615LW. The only separately cataloged
ASUS SSD lock latch found is `13NR0L80P10011`, listed by ASUS US for $0.08 but
temporarily out of stock with a 4 to 9 week lead time. It is not yet a
confirmed exact fit: this machine reports a G615LP board while its chassis
serial resolves as G615LW, and Asus-Parts rejects that SKU for G615LW. A
physical macro comparison, molded part number, or ASUS fitment confirmation
remains required before purchase. IPC lists the candidate at €28.90 including
VAT on an eight-week request basis; no order or seller inquiry was made. Full
manual hashes, seller snapshots, and non-claims are in
[M.2 retainer evidence](../evidence/strix-m2-retainer-part-research.json).

### DeepSeek-V4.1-Flash portability audit, 2026-10-18

DeepSeek-V4.1-Flash is `deepseek_v41`, a 40-layer, 5120-hidden MoE with 384 routed
experts per layer activating 6 plus one shared expert, FP8 dense weights with FP4
experts, CSA2 sparse attention over a 128-token sliding window plus compressed latent
KV, a 1M-token YaRN context, a DeepSeek-ViT vision tower, two Engram n-gram memory
tables of about 384M rows each, and 3 MTP layers plus a DSpark speculative block. The
published checkpoint is 510,286,023,000 bytes across 48 shards at revision
`314113f8a7dadb388215396ba5fbee2c610a04d6`.

Strata's loader refuses anything whose `general.architecture` is not `qwen4exp`, and
its `ModelGeometry` is the Qwen model itself: 48 layers, hidden 2560, 512 experts,
hc 4 with hc_lr 320, GDN plus QSA attention families, and Qwen-shaped tensor names.
The reusable surface is the runtime shell: the server and web app, the installer's
host detection and download logic, the expert tiering design, the GGUF split reader
and dequantizers, the accelerator plugin ABI, speculative acceptance policy, the
sampler, and the platform and memory tiers. The part that would have to be written is
the whole model core: geometry and layout checks, the `sqrtsoftplus` plus `noaux_tc`
top-6 router, CSA2 attention and its hierarchical indexer, FP4 and FP8 decode paths,
KV quantisation, Engram hashing and lookup, the DSpark and 3-layer MTP drafter, and the
DeepSeek-ViT encoder. This is a second model family, not a port.

On the Strix box the model does not fit: 91.4 GiB RAM and 8 GiB VRAM against roughly
475 GiB of weights, with the routed experts alone at about 253 GiB at the shipped FP4
and the Engram tables at about 183 GiB at FP8. Community GGUF ports exist but require a
patched llama.cpp, so they are not the artifact Strata reads. No weights were
downloaded, no code was changed for this audit, and no Strata backend was built. Full
compatibility matrix, measured coupling counts, scale arithmetic, and non-claims are in
[DeepSeek-V4.1-Flash portability audit](../evidence/deepseek-v41-flash-portability-audit.json).

### DeepSeek-V4.1-Flash engine landscape and B550 probe, 2026-10-18

A live probe of B550 found both RTX 3090 Ti cards idle at 0% GPU utilization with 84 GiB
RAM available and 20 TiB free on `/mnt/weights`, but no GPU-enabled llama.cpp build, no
torch, no vLLM, and no SGLang. The distro llama.cpp 8681 package carries no `deepseek_v41`
token at all, so it cannot serve this model. An unrelated transfer of
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`
(48 shards, 510,311,045,295 bytes) was already running there under another owner and was
left untouched; no model was started and no download was started, resumed, or cancelled.

Three read-only research sweeps then mapped the engine landscape. DeepSeek's own
`inference/` tree is a readable PyTorch plus TileLang reference implementation, not a
production engine: it implements CSA2-style compressed sparse attention, Engram, the
DSpark forward pass, FP8 dense weights, FP4 routed experts, and vision, but omits the
DSpark speculative loop, any expert or SSD tiering, and the 1M-token context its shipped
runner defaults to 64K interactive. The production path is FlashMLA, DeepGEMM,
TileKernels, and DeepSelect, all of which now require Blackwell-class NVIDIA or Ascend and
have dropped Hopper. Among general frameworks, vLLM ships native V4.1 support in v0.30.0
with Engram pinned-host offload on by default; SGLang has it on main and first tag
v0.5.21; TensorRT-LLM has it on main only; ik_llama.cpp has a merged `deepseek41` GGUF
arch with a `-dple` file-backed sparse-table option; FreeToken streams the two Engram
tables from NVMe; and ROCm ATOM runs it natively on MI355. Transformers has no
`deepseek_v41` at all.

Several community engines do what Strata does, which is the useful part of this survey.
JigSawPT's llama.cpp fork measures a VRAM 18 GiB, pinned RAM 72 GiB, NVMe tier on one
RTX 5090 at 5.12 tok/s cold and 21.27 tok/s resident. Lucebox places routed experts on a
first GPU, a second GPU, then SSD. antirez's DwarfStar exposes `--ssd-streaming` with
Engram always disk-mapped and targets Apple Metal. sybil-solutions mirrors experts from
VRAM to pinned RAM with disk Engram on a single RTX 3090. None of them is byte-compatible
with Strata's `qwen4exp` loader or documents the FP4 E2M1 plus UE8M0 contract Strata needs,
so the reusable value is the design, not the code. Full host state, worker receipts,
served-route evidence, engine classifications, and non-claims are in
[engine landscape evidence](../evidence/deepseek-v41-flash-engine-landscape.json).

### Lucebox V4.1 baseline fetched, 2026-10-18

The selected DeepSeek-V4.1-Flash architecture baseline is Lucebox at
`5ada2a526a88f2982e0e9a1c86b36e123ee7cada`, fetched read-only under
`build/engines/lucebox-src`. It is Apache-2.0 and has the closest public
implementation to Strata's intended design: a native `deepseek41` loader,
CSA2 shared compressed caches and indexer, Engram, DSpark speculative
decoding, paged serving, router bias and protected experts, and explicit
primary-GPU/secondary-GPU/SSD expert ownership with an `O_DIRECT` streamed
slot cache. The Block-Sparse-Attention and CUTLASS submodule commits are also
pinned and their push URLs are disabled. All three Git object stores passed
`git fsck`; the DSpark converter passed Python syntax compilation. No model
weights or LFS image payloads were downloaded.

The matching October 1017 artifacts are metadata-pinned separately. The
recommended streaming target is 382,700,617,728 bytes, the exact three-stage
DSpark MXFP4/Q8 head is 7,965,746,848 bytes with SHA-256
`02468aa72b7233d50696682817b66ea661ebf387a0b6beecf6954febd8680b1e`, and its
metadata explicitly binds that head to engine commit
`5ada2a526a88f2982e0e9a1c86b36e123ee7cada`. The existing B550
source-checkpoint transfer already contains the three safetensors shards from
which this head can be regenerated, so duplicating the 7.4-GiB sidecar before
storage is settled would be premature.

This is an architecture baseline, not a B550 runtime. Lucebox's V4.1 path is
HIP-only and its documentation refuses CUDA V4.1 builds after an sm_121
candidate-selection mismatch. B550 has two sm_86 RTX 3090 Ti cards and no HIP
backend, so running this baseline there would require a port plus parity
evidence. No B550 restart, Optane change, engine build, or model run was made.
Full licensing, submodule provenance, artifact metadata, checks, and
non-claims are in
[Lucebox baseline evidence](../evidence/deepseek-v41-lucebox-baseline.json).

### DeepSeek V4.1 own-quantization and backend order, 2026-10-18

The DeepSeek track now has an explicit artifact rule: external prequantized files,
including Lucebox ROCMFP, are references rather than runtime dependencies. A
runnable Strata artifact must come from an RESMP-owned deterministic quantizer,
start from a pinned source checkpoint, record its calibration identity and tensor
manifest, and prove exact inversion or bounded reconstruction plus CPU/CUDA
parity before it enters the engine. The preferred initial source is the official
V4.1-Flash-1017 checkpoint at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`
already transferring to B550.

The initial implementation order is CUDA and CPU extraction, not HIP-first.
B550 has two sm_86 RTX 3090 Ti cards and Strix has an NVIDIA CUDA device; neither
provides an AMD acceptance target. Lucebox's V4.1 HIP path is useful as a semantic
oracle, but its CUDA refusal is a specific candidate-selection qualification
failure, not evidence that CUDA cannot implement DeepSeek V4.1. A HIP-first path
would also pull in AMD-specific ROCMFP expert formats and conflict with the
own-quantization rule. HIP becomes appropriate only if an AMD target becomes
primary or an unmodified Lucebox oracle is temporarily required.

The portable work is the `deepseek41` tensor contract, CSA2 cache sharing and
hierarchical candidate selection, Engram hashing, DSpark capture/rollback/verify
semantics, expert ownership, protected experts, router bias, and streaming
scheduling. The graph, pack format, quantizer, CUDA kernels, placement planner,
and DSpark integration must be Strata-owned. The first correctness boundary is a
CPU candidate-selection reference that extends past 16,384 compressed rows,
followed by a CUDA implementation that passes it away from near ties. No format,
converter, kernel, or model execution was implemented by this decision. Details
and non-claims are in
[own-quantization and backend-order evidence](../evidence/deepseek-v41-own-quantization-and-backend-order.json).

### DeepSeek V4.1 implementation kickoff, 2026-10-18

The first build cycle is split into three disjoint host-safe slices so the new
model family can enter Strata without importing Lucebox's runtime or ROCMFP
artifact:

| Slice | Writer path | File ownership | First acceptance |
| --- | --- | --- | --- |
| Checkpoint manifest gate | `.worktrees/deepseek-manifest` | `scripts/check_deepseek.py`, `scripts/tests/test_check_deepseek.py`, `data/deepseek_v41_manifest.json` | Python unit tests plus `just quality`; no checkpoint download |
| Geometry contract | `.worktrees/deepseek-geometry` | `include/strata/deepseek/geometry.hpp`, `src/deepseek/geometry_test.cpp`, minimal CMake test target | Host C++ test and `just quality` |
| Candidate-selection reference | `.worktrees/deepseek-candidates` | `include/strata/deepseek/candidate_select.hpp`, `src/deepseek/candidate_select_test.cpp`, minimal CMake test target | Host C++ tests for newest-block pinning, block top-k, restricted row top-k, ties, and visibility beyond 16,384 rows |

The parent checkout remains the build owner and integration point. Writers must
not modify `scripts/strata.py` or the top-level CMake target list; the parent
will integrate their disjoint targets after reviewing diffs and receipts. The
manifest slice may encode the already-pinned 1017 shape contract but must treat
the 510-GiB checkpoint as an external input, never download it, and emit a
refusal with missing or extra tensor families. The geometry slice defines only
the model contract, not an executable engine claim. The candidate slice is the
first CUDA-blocking correctness primitive and must use deterministic tie
behavior. No worker may start a model run, alter B550, or touch the in-progress
checkpoint transfer.

### DeepSeek V4.1 first host contracts, 2026-10-18

The three worker slices are now integrated in the parent checkout. The exact
1017 manifest gate compiles 114 tensor-pattern families, 96,085 indexed tensor
names, 94,464 routed-expert tensor entries, 48 shards, unpadded layer and
expert indices, and the 510,286,023,000-byte index total. Missing or unknown
tensor families are hard failures. The same gate passed against the official
revision `1bae0e1088993b9b82de08f59de5be5a34e42170` `config.json` and
`model.safetensors.index.json` metadata already fetched under `/tmp`; it did
not open a shard or download weights.

`strata::deepseek_v41` now owns the host geometry contract (40 backbone
layers, three DSpark-tail stages, the exact compress/KV/index/Engram sets, and
DSpark/candidate-selection dimensions) plus the deterministic two-stage CPU
candidate-selection reference. Selection pins the newest block, orders block
ties by block ID, restricts row top-k to selected blocks, resolves row ties by
row ID, returns ascending rows, reports near-tie diagnostics, and exercises
more than 16,384 visible rows. It is a correctness primitive, not a CUDA
kernel or engine path.

The common gate now runs `scripts/check_deepseek_contracts.py`, which checks
the manifest and configures only the dependency-free CMake project in
`src/deepseek`. An initial attempt to place these targets behind an option in
the top-level engine CMake file was rejected after configuration reached the
unrelated llama.cpp `FetchContent`; the standalone project now configures and
runs without fetching or compiling CUDA, HIP, or SYCL. Full commands, worker
integration evidence, sanitizer results, and non-claims are in
[first host-contracts evidence](../evidence/deepseek-v41-first-host-contracts.json).

### DeepSeek V4.1 B550 host bring-up, 2026-10-18

The pinned `7f882ef` contract revision was transferred to B550 as a Git bundle
and checked out at `/home/kearm/strata-deepseek-v41`, with the push URL pointed
at a non-repository path and verified by reading the remote config back. B550 is
therefore an execution host for this work, not a source authority. The transfer
did not use remote Git credentials and did not touch the already-active DeepSeek,
Qwen, or Fable transfers, the running tmux sessions, or the GPU jobs owned by the
indexing and distillation services.

B550 answered at `192.168.1.171` with two RTX 3090 Ti cards, and reports the
hostname `b550` today where an earlier probe recorded `compute`; `AGENTS.md` now
states that hostnames drift and selection stays by alias and address. GPU 0 was
held at 99% by the indexing and distillation services, so this validation stayed
CPU-only. GCC 15.2 and CMake 4.2.3 configured the dependency-free project, built
both contract tests, and passed CTest 2/2, with a separate command asserting that
no dependency directory was created. The deterministic candidate-selection
reference passed direct optimized and AddressSanitizer/UndefinedBehaviorSanitizer
runs. All eight build inputs, including both compiled test sources and the
standalone CMake file, matched the main-machine SHA-256 digests, and all 15
official 1017 metadata checks passed after copying only `config.json` and
`model.safetensors.index.json` from the pinned revision to the host. Commands,
per-check timings, digests, and non-claims are in
[B550 bring-up evidence](../evidence/deepseek-v41-b550-bringup.json).

### B550 Qwen AVX2 ceiling and DeepSeek CPU boundary, 2026-10-18

B550's live CPU ceiling is AVX2 with FMA3 and F16C, with no AVX-512. The
regular Strata Qwen engine was therefore built there at revision `83f67c4` for
both sm_86 cards with CUDA 12.4 and the portable AVX2 CPU baseline. The build
used the pinned llama.cpp tree `3cf0325`, supplied from the main machine after
the default GitHub FetchContent clone stalled behind four active transfers. All
145 build steps completed and linked the 62 MB `strata` engine.

The existing AVX2 kernels are already wired and passed on B550: router BF16 dot
parity, the i-quant expert-row parity suite, Q8_K activation quantization, and
the opt-in Q2 bitplane check. CTest passed 3/3. The CUDA device selftest also
passed without a model: the busy GPU 0 correctly refused its default plan, while
the idle GPU 1 reported 18.9 GiB free and marked that plan as fitting. The Qwen
model itself remains unavailable as a runnable Strata artifact because the
active source transfer is only 17 of 131 safetensors shards and no GGUF pack or
PLE table exists on the host.

A DeepSeek-specific AVX2 kernel is not the next useful CPU boundary. At the real
40,000-row candidate geometry, compiling the current reference for AVX2 improved
one call by only about 9% (2.019 ms to 1.831 ms, identical checksums). Stage
two costs about 2.000 ms because it fully sorts 16,384 candidate rows to keep
512; the vectorizable 40,000-row block-maximum scan costs only 0.035 ms. The
format-independent opportunity is bounded top-k/partial ordering, not an AVX2
scan rewrite. More importantly, no RESMP-owned DeepSeek pack layout exists yet,
so there is no weight-row layout for an AVX2 expert kernel to target. Full
commands, measurements, GPU occupancy, and non-claims are in
[B550 AVX2 ceiling evidence](../evidence/deepseek-v41-b550-avx2-ceiling.json).

### Qwen B550 HDD versus NVMe scratch decision, 2026-10-18

B550's `/mnt/weights` is a 21.8-TB rotational ext4 device and the root is a
3.6-TB NVMe with 2.1 TiB free. A completed 3,200,016,032-byte Qwen shard was
used as the storage substrate for bounded fio O_DIRECT reads while the Qwen
transfer remained active. Sequential HDD throughput was healthy at 202 MiB/s;
NVMe was 2.48 GiB/s. The runtime-relevant pattern is different: a
Strata-sized 1,382,400-byte expert-slot read averaged 12.37 ms on HDD versus
1.14 ms on NVMe at queue depth 1, and 66.55 ms versus 3.64 ms at queue depth
8. NVMe was 10.8x faster for the depth-1 expert pattern and 18.3x faster at
depth 8.

The decision is therefore split by path. The HDD remains appropriate for the
long source-checkpoint transfer and could stream a mostly sequential preload
without disaster. A runnable Qwen GGUF pack should be staged to NVMe scratch
before decode: at the measured mixed copy rate, a 70-GiB pack costs about six
minutes and a 125-GiB pack about eleven. A separate 953.9-GB NVMe exists but
is unmounted and contains foreign FAT/NTFS partitions, so the safe current
scratch target is the already-mounted root NVMe; the smaller drive must not be
formatted without a separate identity and data decision. Full measurements,
conditions, and non-claims are in
[Qwen B550 storage evidence](../evidence/qwen-b550-storage-read-baseline.json).

### Qwen B550 one-versus-two-GPU optimization plan, 2026-10-18

B550's two RTX 3090 Ti cards have no NVLink and report a `PHB` topology. GPU 0
hangs off the B550 chipset with an effective Gen3 x4 maximum; GPU 1 is on the
CPU root with Gen4 x16. Both embedding replicas remain resident. This makes
Strata's native layer split, helper expert
cache, adaptive CPU/GPU expert tier, PCIe share, and pipeline windows more
relevant than a K-Transformers-style tensor-parallel port. The local
K-Transformers checkout at `c6a9ee3` contributes only Qwen/DeepSeek registry
entries on top of upstream; no K-Transformers code is imported by Strata. Its
useful concepts, hotness-aware GPU experts and a CPU/GPU MoE split, already
have native Strata counterparts.

The source checkpoint is not itself runnable, but the RESMP-owned calibrated
IQ2_XS/Q2_0 artifact already exists on Strix and in the public RESMP-DEV
repository at revision `d802333`; both local shard hashes were rechecked on
2026-10-18. It is the first benchmark candidate, with the same external
GSQ-RCO artifact available as a control. Every benchmark must use identical
hashes, prompts, context, KV mode, speculative depth, expert profile, NVMe
staging path, and engine binary. Measure each single GPU separately, then
layer-split auto in both card orders and a bounded manual split sweep, then the
helper-expert alternative.
Only after choosing the best placements should `--adapt-async`, pipeline
windows, fused prefill, short-prompt helper sharing, and batch groups be added
and compared separately. A close community 2x3090 Qwen IQ3_S report suggests a
hypothesis around 135-145 decode tok/s and 1.8K-2.3K prefill tok/s, but it is
not a B550 measurement. The complete protocol, topology, K-Transformers audit,
and non-claims are in
[Qwen B550 multi-GPU plan](../evidence/qwen-b550-multigpu-optimization-plan.json).

### Calibrated IQ2_XS staging and fixed B550 slot topology, 2026-10-19

Strix holds the verified calibrated artifact but was unreachable over LAN,
ZeroTier, mDNS, and the local relay; five Wake-on-LAN packets to its pinned
wired MAC did not restore SSH. The exact public RESMP-DEV revision
`d80233332687dd87fe3a41292b7d500c966dbb63` is therefore being staged directly
on B550 by the maintained HFD CLI, bounded to aria2 x4 and one concurrent file.
The detached job writes status under
`/home/kearm/strata-scratch/qwen-iq2xs-calibrated/` and must finish both full
verification and the two pinned SHA-256 checks before the artifact is accepted.

The ASUS TUF GAMING B550-PLUS WIFI II documentation confirms the live PCI
result is fixed wiring, not a missing BIOS setting. PCIEX16_1 is the CPU Gen4
x16 slot and offers only a Hyper M.2 `PCIe RAID Mode`, while PCIEX16_2 is a
chipset slot documented as Gen3 x4/x1. There is no x8/x8 two-GPU bifurcation
mode. Swapping cards or forcing link generations cannot turn PCIEX16_2 into
Gen3 x8 or route CPU lanes to it. Full evidence is in
[B550 staging and PCIe evidence](../evidence/b550-iq2xs-staging-and-pcie-bifurcation.json).

### Calibrated IQ2_XS artifact staged on B550, 2026-10-19

Strix returned with the expected machine ID before the fallback HFD transfer
completed. A direct Strix-to-B550 rsync measured 112-118 MiB/s, so the bounded
public-HFD job was stopped and its partial file retained as the delta basis.
The 66,649,935,904-byte GGUF shard set completed in nine minutes. Both shard
digests match the public RESMP-DEV release, and the stale `.aria2` marker was
removed only after those hashes passed. The 1.50 GB prepared pack also copied
directly, and `dense.bin`, `index.txt`, `native_experts.txt`,
`conversions.json`, and `compat-bf16.json` match their deterministic-artifact
hashes. The complete benchmark input now lives under
`/home/kearm/strata-scratch/qwen-iq2xs-calibrated/` on root NVMe with 1.9 TiB
still free. No model has been loaded or benchmarked yet. Full transfer and hash
evidence is in
[direct staging completion evidence](../evidence/b550-iq2xs-direct-staging-complete.json).

### Qwen B550 split-pack, MTP, and first one-versus-two-GPU run, 2026-10-19

The staged prepared pack was valid for the original unsharded GGUF but not for
the published split files: its expert offsets were absolute offsets in the
unsplit tensor stream, and the PLE table lives in shard 1. A new
`pack-split/` was generated without changing expert bytes. The generator
emitted native-expert table v4, including layer 13's per-role cross-shard
columns; `dense.bin` and every tokenizer file stayed byte-identical. The
original pack remains untouched as provenance.

The verbless one-shot engine then loaded 32.23 GiB of experts and executed on
both physical cards. With the same prompt, profile, 2,233 slots, and greedy
sampling, B550 produced different one-token argmaxes across runs (157, 487,
and 220), so a one-token match is no longer treated as cross-host correctness
parity. A PLE-shard check, the 2.9 GB/s chipset-link probe, and the 25.0 GB/s
CPU-root-link probe identify the physical cards despite the engine's remapped
`CUDA0` label.

The missing Qwen MTP runtime was reconstructed from the pinned BF16 checkpoint
using only its 31 range-requested tensors (5.214 GB), with all pinned
SHA-256 checks passing. The Q2_0 GGUF and runtime layout are now local NVMe
artifacts. On a 25-token prompt and 94-token completion, two-GPU auto split
with MTP held 22,207 experts and observed warm decode rates of 200.6 and
208.7 tok/s. GPU 0 alone held 11,360 experts at 82.1 and 92.9 tok/s; GPU 1
alone held 9,934 at 98.5 and 116.7 tok/s. GPU 1's Gen4 x16 link outweighs its
smaller cache. Adding pipeline windows 2 traded 234 resident experts for warm
runs of 219.1 and 214.2 tok/s, a directional gain that needs repeated
interleaving before promotion.

These are first bring-up and optimization observations, not release claims.
The one-token result does not prove generation quality, and the short warm
coding prompt does not establish long-context or multi-client performance.
All task-owned model services were stopped; the unrelated embedding replicas
remain resident and untouched. Full commands, hashes, occupancy, timings, and
non-claims are in
[Qwen B550 baseline](../evidence/qwen-b550-iq2xs-bringup-and-mtp-baseline.json).

### Speculative-head and QAT direction, 2026-10-19

Research across the pinned DeepSeek-V4.1-Flash checkpoint, the DSpark and
DeepSpec releases, current acceptance-loss papers, and the local DFlash2
campaign separates the immediate production lane from the next research
architecture. DeepSeek-V4.1-Flash does not use a DeepSeek-V3-style joint MTP
module. Its 2,401 `mtp.*` tensors are a 7.388-GiB DSpark stack: three
DeepSeek-V4 blocks, target layers 37-39, a five-token block, 128 routed
experts with three active, a rank-256 Markov head, and a scalar confidence
head. DSpark reports 16-18% longer accepted blocks than DFlash and 27-31%
longer than EAGLE-3 in matched open experiments, and 60-85% faster per-user
generation than MTP-1 in DeepSeek's production serving system.

For Strata, the native one-layer Qwen4Exp MTP remains the first practical QAT
target because it is already executable, shares the target tokenizer and LM
head, and produced 69-71 accepted drafts from 71-72 offered on the short B550
coding probe. The higher-upside research lane is not more selector-only
distillation: it is a DSpark-style rank-256 Markov head and calibrated
confidence head over the promoted DFlash2 generation-3 full head. Public
DSpark Qwen3 checkpoints are references, not Flash-Next-compatible weights;
their target feature seam and vocabulary differ.

The first training pass should freeze the calibrated target, capture
anchor-bounded pre-LM-head hidden states and exact loss mass, and compare
acceptance-directed EAL/LK objectives against KL and DSpark's weighted
0.1 CE plus 0.9 TV objective. A DSpark-style student also trains its
confidence head with weight 1.0 and position decay gamma 4. Deployment
quantization follows a BF16 candidate that already wins live acceptance. The
existing MXFP4 GPTQ path is post-training quantization, not QAT; true
fake-quant/straight-through QAT is justified only if the quantized BF16
candidate loses exact parity or paired acceptance. Promotion always requires
exact target-controlled parity, paired held-out and corpus acceptance, and a
same-day throughput/occupancy receipt. Full sources, local DFlash2 limits,
and non-claims are in
[speculative-head research](../evidence/qwen-dspark-mtp-qat-target-research.json).

### DSpark2 Qwen3.8-27B training parameters, 2026-10-19

The local Qwen3.8-27B environment is the correct place to develop DSpark2
before attempting a Flash-Next target adapter. It already has the promoted
generation-3 DFlash2 head, a 127-prompt / 11,273-cycle combined corpus, the
excluded 25-prompt Fable live gate, and a working CUDA trainer. This is a
training prototype, not a Flash-Next drop-in: the 27B target has 64 layers and
hidden size 5,120, while Flash-Next has 48 hybrid QSA/GDN/PLE layers and
hidden size 2,560.

The existing captures cannot train official DSpark unchanged. They preserve
target top-64 IDs/values and the greedy token, while DeepSpec's TV/L1 and
confidence objectives require aligned full-vocabulary target logits. A block-7
full-logit re-capture over the current combined corpus is projected at about
41 GiB, comfortably within B550 NVMe capacity. The existing corpus is highly
skewed to four- and five-position blocks, so block-7 data must be captured
rather than padded.

The staged parameters are: retain the promoted G3 five-layer head and target
layers 5/19/33/47/61; add Markov rank 256 and a scalar confidence head; use
block size 7; train new heads first for three epochs at about 3e-5 with
official loss weights CE 0.1, TV/L1 0.9, confidence BCE 1.0, and position decay
gamma 4; then jointly train rank-8/alpha-16 LoRA at about 1e-6 for two epochs.
The new heads use zero weight decay and 4 percent warmup; the joint stage uses
the measured G3 weight decay around 1e-4. A one-epoch EAL/LK or on-policy
error-replay arm follows only after a supervised winner. Official DeepSpec's
from-scratch 6e-4, batch-512, ten-epoch recipe is a reference, not a local
default: its corpus is much larger and prior local proxy metrics repeatedly
failed to predict live acceptance.

A bounded live smoke on free GPU 0 validated the current trainer: 512
stratified cycles, rank 8, LR 1e-6, 488 steps, 11.66 seconds training time,
11,041 MiB peak GPU memory, and exit 0. Full DSpark TV/full-logit training is
not included in that timing. Full corpus, prior-search, official-config,
resource, and non-claim details are in
[DSpark2 parameter evidence](../evidence/dspark2-qwen27b-training-parameters.json).

### Qwen B550 explicit-hugepage decision, 2026-10-19

	The current calibrated IQ2_XS/Q2_0 two-GPU MTP server was run with a
	runtime-only pool of 16,512 explicit 2 MiB pages and with
	`STRATA_NO_LARGEPAGES=1`. The arms alternated across four server starts. The
	default arm reported `hugetlb 2 MB pages`; the control reported 4 KiB pages.
	Each arm contributed eight warm requests after a discarded cold request.

	Warm decode was effectively tied and slightly favored 4 KiB pages: the
	medians were 217.1 tok/s for hugetlb and 217.45 tok/s for the control. Warm
	prompt medians were 136.45 and 135.8 tok/s. Every warm request accepted 71
	of 72 drafts, and all response content matched one SHA-256. Hugetlb did
	reduce the 32.23 GiB expert-load phase from a median 47.5 seconds to 40
	seconds in alternating rounds.

	The page pool was released after the test. Persisting it is not justified for
	this fully resident workload because the only measured win is startup/load
	time while a permanent pool would remove about 32 GiB from general memory.
	Revisit explicit pages for a startup-sensitive resident service or after a
	measured host-expert/cache-miss-heavy workload wins. Full protocol, cleanup,
	and non-claims are in
[Qwen B550 hugepage A/B](../evidence/qwen-b550-iq2xs-hugepage-ab.json).
	The operator procedure for temporary allocation, boot-time persistence, and
	deliberate un-persistence lives in
	[troubleshooting](TROUBLESHOOTING.md#linux-allocate-explicit-2-mi-huge-pages-only-after-a-measured-reason).
	It deliberately keeps server shutdown from releasing the pool: page ownership
	is a host decision, not a model-lifecycle side effect.

### DSpark2-Qwen full-logit distillation on the Prime cloud lane, 2026-10-19

The MLX DFlash2 prediction-head work is now the distillation template. The
draft model emits hidden states and deliberately owns no LM head; the frozen
target head is supplied externally (MLX `TargetOps.logits_from_hidden`, trainer
`--lm-head`). Full-logit capture stores `target_logits` F32 `[5, 248320]` per
cycle, which is exactly the aligned full-vocabulary signal the DSpark TV and
confidence objectives need, replacing the earlier top-64-only limitation. The
trainer branch reached `5f47153`, adding configurable selector/CE/TV weights, a
`--selection-metric auto` guard, an MXFP4 QAT fake-quant path with
straight-through gradients, and a training fingerprint (model + capture SHA-256
plus hyperparameters) bound into checkpoints and receipts. Twenty focused tests
pass locally, including the MXFP4 straight-through gradient and the aligned
full-vocab distribution loss.

Work moved to the Prime cloud lane (RTX 6000 Ada 48 GB) because Modal's free
credits are effectively exhausted. The user's own `dflash-lr-ext` session is
running the DSpark-weighted hybrid sweep there (ce 0.1 / tv 0.9, rank 8, alpha
16, seed 73) on a frozen 4,096-cycle full-logit corpus; that job was preserved
and never stopped. A Codex-armed watcher for a redundant weights-1.0 baseline
on the incoming 12,000-cycle slice was killed to avoid contending for the single
GPU; it created no training directory and left the transfer running.

Measured LR trend on the frozen corpus is monotonic through 4e-6: validation CE
7.152 at 5e-7, 7.137 at 1e-6, 7.069 at 2e-6, and 6.615 at 4e-6 (3,818 steps,
~280-300 s each), so the DSpark2-Qwen rank-8 LR floor is at or above 4e-6,
materially higher than the earlier selector-only ~1e-6 regime, and the upward
sweep is still open. A Codex MXFP4 QAT smoke (hybrid, rank 8, 128-cycle pilot)
exited 0 at 13.4 GiB peak and 164 s for 99 steps, about 11x the BF16
full-replay cost, which fixes QAT as a winner-only step rather than a sweep
default. This all trains the Qwen3.8-27B prototype (hidden 5120 / 64 layers),
not Flash-Next (hidden 2560 / 48 layers); no Flash-Next DSpark2 training, no
promotion, and no live-acceptance gate ran this turn. Full state, commands, and
non-claims are in
[DSpark2-Qwen cloud training state](../evidence/dspark2-qwen-cloud-training-state.json).

The B550 explicit-hugepage A/B is reviewed and stands: hugetlb cut the 32.23 GiB
load median from 47.5 s to 40 s but left warm decode tied at ~217 tok/s with
identical 71/72 acceptance, so `vm.nr_hugepages` stays 0 and page ownership
remains an operator action. The Optane driver work is out of scope; no B550 or
Strix GPU was probed, launched, or benchmarked this turn.

### GSQ/RCO pruned Qwen variant and prune-heal-quantize-verify order, 2026-10-19

The IST-DASLab Coder release was inspected live and pins the method this track would
reproduce: 512 routed experts per layer across 48 layers with 10 active per token, 256
of 512 retained (12,288 of 24,576), retained weights at 3.5 bpw for an effective
1.89 bpw over the original 176.9B parameters. Shard 1 is 29.6 GB and must be resident;
shard 2 is the 28.8 GB per-layer n-gram table at IQ4_NL, neither pruned nor searched,
and may be served from disk. The vendor's own framing is that the resident working set
of a 176.9B-parameter model fits a single 32 GB accelerator, which makes one V100 SXM2
the natural first runtime target and two P100 16 GB cards the layer-split control.
SWE-bench Verified 79.20 against a BF16 base of 82.80 (95.7% retained) and LiveCodeBench
v6 87.43 unchanged are vendor numbers at xhigh reasoning effort, not reproduced here.

Two Apache-2.0 tools implement the method. GSQ (arXiv 2604.18556) is post-training scalar
quantization: GPTQ second-order initialization, then Gumbel-Softmax refinement of
per-coordinate grid assignments and per-group scales under Lion with a cosine schedule,
per-layer shards offloaded to a meta device so models larger than VRAM quantize, and a
reassembled HF checkpoint. It stays in a symmetric group-wise scalar format compatible
with GGUF K-Quant deployment. RCO (arXiv 2605.00649) does the selection: it optimizes the
KL divergence of the whole pruned model jointly across layers on a Riemannian budget
manifold, with tangent projection, retraction by binary search on a monotone cost map, and
vector transport, so the budget holds to floating-point noise. `rco_search_prune.py` emits
`mask.pt`, `run_build_checkpoint.py` materializes the pruned checkpoint, and `src/unfuse.py`
splits fused 3D MoE experts. GSQ's listed wrappers cover Qwen3.5/3.6-MoE but not
Qwen3.8-Flash-Next, so wrapper support for its hybrid QSA/GDN/PLE layers is unconfirmed and
is the first gate.

On RCO versus a REAP-style heuristic: with compute available RCO is the stronger selection
method, because a top-k importance ranking is a per-layer proxy with no joint search while
RCO optimizes the full-model KL under exact per-layer budgets, and the authors report it
matching or exceeding EvoPress/EvoESAP at 3-16x lower wall-clock. Two qualifications keep
this from being unconditional. GGUF stores one expert count for the whole model, so a global
budget cannot exploit per-layer sensitivity anyway; RCO still wins on joint loss, but the
margin over a good per-layer heuristic is a measurement, not a certainty. More importantly,
method and calibration are independent axes, and calibration is our real lever: the vendor
calibrated on a generic code, agentic, vision, and spatial mixture and explicitly invited
feedback on capabilities absent from it, while we already record activated experts per layer
and position on our own agent traffic through `--dump-routing`. The decision is therefore to
run RCO as the primary arm and a REAP-style top-k-from-trace selection as a nearly free
control, and to accept RCO only if it beats that control on our held-out acceptance rather
than on vendor scores.

The stage order is prune in BF16, heal in BF16, quantize once at the end, then verify
bit-exactly. Healing a quantized artifact and re-quantizing discards most of the adaptation,
because the re-rounding removes what the heal moved. QLoRA CPT restores nothing that pruning
deleted; those parameter blocks are gone, and attempting to recover them would be
antithetical to the release's premise. What a short heal run adapts is the router
redistributing load across a 256-expert pool and the retained and shared weights compensating
for lost specialization, which is plausibly what lies behind the vendor's 2026-10-17 update
raising SWE-bench Verified from 75.60 to 79.20 with looping reduced, though they do not
attribute it to a healing run. The n-gram table stays frozen throughout, as it is a lookup
rather than a matmul weight and the vendor excludes it from both search and quantization.

Local position: we hold the RESMP-owned source at `~/models/Qwen3.8-Flash-Next-BF16`
(335 GiB, 131 shards), which is the base any re-pruning or re-quantization starts from, so
the own-artifact contract is satisfiable without depending on a borrowed quantized file. The
deployment half is already owned and is the correctness-critical half: `tools/strata_pack.py`
builds and verifies under the rule that `decode(pack) == dequant_ggml(source)` bit for bit,
canonicalized over 757,139,840 blocks; `tools/iq_pack.py`, `tools/quantscope.py`,
`tools/standard_imatrix.py` with its pinned `256x2048` calibration corpus,
`tools/make_profile.py` with `--n-expert 256`, and `tools/bench_eviction.py` complete the
pack, profile, and policy-replay machinery. The LoRA infrastructure exists but targets
Qwen3.8-27B (hidden 5120, 64 layers) rather than Flash-Next (hidden 2560, 48 hybrid layers),
and the existing MXFP4 QAT fake-quant path with straight-through gradients measured about
11x the BF16 full-replay cost, so QAT stays a winner-only step and is justified only if the
GSQ BF16 candidate loses exact parity or paired live acceptance. GSQ, RCO, peft, and unsloth
are not cloned locally.

Sequencing is Qwen first, then DeepSeek. Qwen is executable against artifacts we already
hold and validates the methodology on one 32 GB card. DeepSeek V4.1 Flash remains
contracts-only in Strata, with manifest, geometry, and candidate-selection references but no
runnable pack, quantizer, or CUDA backend, and its 763B scale with 189.13 GiB of Engram
tables in mixed FP8 and packed MXFP4 storage makes it a distinct and larger undertaking
rather than a parallel one. The same prune, heal, quantize, verify shape transfers once a
backend exists.

No component was built, no repo cloned, no model downloaded, pruned, healed, quantized,
packed, or run this turn. GSQ's Flash-Next wrapper support is unconfirmed, and the stage
costs are not estimated beyond the vendor's RCO efficiency claim. Full pinned facts, the
local inventory, the stage gates, and the non-claims are in
[GSQ/RCO prune-heal-quantize plan](../evidence/gsq-rco-prune-heal-quantize-plan.json).


### GSQ/RCO compression preflight, 2026-10-19

The first implemented slice is metadata-only: `scripts/check_compression.py` validates
the local Flash-Next BF16 source, its exact 48-layer/512-expert/10-active geometry,
the 48 gate plus 48 fused gate-up plus 48 fused-down expert tensors, the 128 PLE
n-gram shards, shard completeness, the pinned GSQ/RCO clones, and a Transformers
runtime new enough for `qwen4_exp`. Focused tests cover checkpoint mutations, the
runtime boundary, and tool revision/entrypoint checks. The check never opens a
safetensors payload, downloads a model, imports torch, or claims a compression run.

The live preflight passed against the complete local source at
`~/models/Qwen3.8-Flash-Next-BF16` (131 shards, 1,658 index tensors, no transfer
markers). It also pinned an important runtime boundary: Transformers 5.7.0, the
version declared by RCO, rejects `qwen4_exp`, while 5.20.2 loads
`Qwen4ExpConfig` and `Qwen4ExpTextConfig` and maps `qwen4_exp_text` to
`Qwen4ExpForCausalLM`. Transformers' Qwen4-Exp implementation inherits from the
Qwen3.5-MoE classes that GSQ already wraps, including the conditional
`text_config`, language-model layer prefix, hybrid linear/full layers, shared
expert, and fused 3D routed experts. A GPU environment must therefore override
RCO's declared 5.7 pin with at least the 5.16 support family (5.20.2 verified for
config loading). A later tiny-model probe then validated RCO's router hook and
physical fused-expert materializer on CPU, as recorded below. GSQ's PLE-aware
layer movement remains unvalidated.

Full commands, exact counts, the 5.7-versus-5.20 result, and non-claims are in
[compression preflight evidence](../evidence/gsq-rco-compression-preflight.json).
No model object, RCO search, GSQ run, heal, quantization, pack, or inference was
executed in this slice.


### Trace-derived REAP pruning control, 2026-10-19

The cheap control arm for the GSQ/RCO track is implemented. `tools/reap_mask_from_traces.py`
consumes the engine's exact `--dump-routing` record format (`int32 layer`, `int32 k`,
`int32 expert ids`, `float32 router weights`), ranks each layer by true router weight or
activation count, resolves ties by ascending original expert ID, and keeps an exact budget
(default 256 of 512). Sparse layers are filled deterministically and fill IDs are reported
separately. The JSON mask records every trace path and SHA-256; an optional Strata profile
is written through the existing profile format and read back in tests. Serve-mode records,
whose multi-token dispatcher emits zero weights as unit activations, are handled explicitly.

Five focused tests cover weighted tie behavior, zero-weight serve records, exact sparse fills,
CLI/profile round trip, and malformed-trace rejection. The test is wired into `just quality`
as the `compression-tools` domain. No routing trace currently exists in this checkout, so no
real mask was generated; the tool prepares for trace capture and is a heuristic control, not
RCO's joint full-model KL search. Full contract, checks, and non-claims are in
[REAP trace-control evidence](../evidence/reap-trace-control.json).


### Tiny Qwen4-Exp RCO compatibility probe, 2026-10-19

`tools/validate_qwen4exp_rco.py` instantiates a tiny random
`Qwen4ExpForCausalLM` under Transformers 5.20.2, imports the pinned RCO tree, and
exercises the two model-specific boundaries that blocked a real pruning run.
RCO reads the expected `(2,4,2)` MoE geometry; `Qwen4ExpTextTopKRouter` supplies
the three-value logits/scores/indices signature RCO consumes; the fused
`Qwen4ExpTextExperts` layout executes through RCO's STE wrapper; and RCO's
checkpoint materializer physically shrinks router plus fused gate-up/down tensors
from four experts to three before another finite forward.

This ran on CPU with a tiny random model, not the 360 GB target checkpoint. It
removes the structural Qwen4Exp/RCO adapter objection and justifies moving to a
real GPU model-load/calibration boundary, but it does not validate full-model KL,
calibration data, PLE-aware GSQ layer movement, quantization, packing, or output
quality. Evidence is in
[tiny Qwen4-Exp RCO probe](../evidence/qwen4exp-rco-tiny-probe.json).


### Modal L4 Qwen4-Exp/RCO smoke, 2026-10-19

`tools/modal_qwen4exp_rco_probe.py` runs the tiny validator on one Modal NVIDIA
L4 with Python 3.12, Torch `2.14.1+cu130`, Transformers `5.20.2`, and RCO pinned
at `9a1e09c`. The CUDA probe passed the same geometry, router triple, STE fused
expert forward, finite output, physical expert shrink, and finite shrunk output
contracts as the CPU probe. The observed Modal metered total moved from $21.38
to $21.58 while billed cost remained $0.00; the $0.20 delta includes the initial
git-less image-build failure and the successful retry. Result and log are retained
under ignored `build/compression-cloud/`, with the durable record in
[Modal L4 probe evidence](../evidence/qwen4exp-rco-modal-l4.json).

This advances runtime compatibility only. It does not load Flash-Next weights,
run RCO calibration or full-model KL, train a heal adapter, quantize, pack, or
measure performance, and it does not size the eventual multi-GPU search.

### Qwen B550 full-VRAM clean MTP control, 2026-10-19

With explicit authorization, the two resident B550 Jina embedding replicas were
stopped cleanly. The pinned Qwen IQ2_XS/Q2_0 two-GPU MTP configuration then fit
all 24,576 profiled experts in VRAM, 12,288 per card, loaded 32.23 GiB at
1.07 GiB/s, and retained 3,649 MiB free with everything loaded. On the exact
prior short coding prompt, one discarded warm-up and five measured requests
produced median prompt and decode throughput of 151.5 and 227.4 tok/s; every
request completed 94 tokens, offered 78 MTP drafts, accepted 66, and returned
one output-content SHA-256. The clean median is 4.96 percent above the prior
two-sample mean with roughly 22k experts resident, but the arms were not
interleaved, so this is an observational control rather than a causal service
benchmark.

The task-owned server was stopped afterward and both GPUs returned to idle.
The embedding units remain intentionally stopped, so their owner must restart
them before sindexer depends on ports 1236 and 1237. Strix was probed read-only
as a fallback and is online with an idle RTX 5070 and accessible NPU device
node. Full state and non-claims are in
[full-VRAM clean control](../evidence/qwen-b550-full-vram-clean-control.json).

The stopped embedding units were restored immediately after the first
post-merge `sindexer update` exposed the dependency with nginx 502 responses.
Both replicas returned the expected Jina model on ports 1236 and 1237, the Mac
tunnel route recovered, and the index-update retry completed with 337 chunks
created across 4 files. The transient initial 502s were not diagnosed beyond
that restore-and-success boundary.

### DSpark2-Qwen GPU provider selection, 2026-10-19

The full-logit lane stays on Prime Intellect, with one RTX 6000 Ada 48 GB as
the preferred card when stocked. A real Prime A100 80 GB control ran the same
DFlash2 G3 model, immutable teacher-logit capture, trainer revision, rank, and
hybrid CE/TV objective. It completed 99 steps in 11.459 s
(`0.115748 s/step`) and peaked at 9.428 GiB allocated. The authoritative
17,500-cycle Ada receipt is 16,587 steps in 1252.894 s (`0.075535 s/step`)
with 9.670 GiB peak allocation. At the observed prices, A100 is 1.53x slower
per step and 2.51x more expensive per step. Memory is not the constraint:
trainer allocations stayed below 10 GiB, while observed GPU telemetry peaked at
9.77 GiB on Ada and 10.12 GiB on A100. A 48 GB card has ample capacity, and
80 GB plus multi-GPU bandwidth does not help this sequential batch-1 workload.

Prime also remains the best provider lane after account probing: Modal has no
remaining credit headroom and higher comparable-GPU rates, RunPod has a zero
balance, GMI is inference-oriented, Vast requires elevated 2FA, and Lambda has
no token. The task-owned A100 pod was terminated after receipt harvest and
billed $0.5393. A later inventory query listed no Ada; inventory is transient,
so an unmeasured H100, H200, or RTX 5090 substitute is not authorized by this
decision. Full measurements and non-claims are in
[GPU provider selection](../evidence/dspark2-gpu-provider-selection.json).

### B550 concurrency fallback tier and CPU-embedding capability, 2026-10-19

Three questions were answered by live measurement, not documentation. First, can
Strata serve concurrent requests as an agent-tool fallback tier? The engine does
support it: `--batch N` slots with `--batch-groups auto` across a layer split,
`--batch-mtp` (one MTP proposal per slot) on a single GPU, and
`concurrency.serving` on `/v1/status`. But on this launch-bound,
expert-streaming workload concurrency buys latency fairness, not aggregate
throughput. The two-GPU full-residency config (24,576/24,576 experts, 3,491 MiB
free) measured aggregate decode 149.6 tok/s at C=1, 126.3 at C=2, and 158.1 at
C=4; per-request median fell 199.8 to 49.7 tok/s across that range, likely because each
verify window carries one token of every active slot (a hypothesis consistent with
`docs/BATCHING.md`; no profile isolated the cause). Those are two different
denominators and must not be read as one number: the aggregate figure is total
completion tokens divided by total request wall time across the whole round set
(so it includes prompt time for every request, not decode only, despite the
driver reporting it as a decode rate), while the per-request figure is the median
of each request's own engine-reported decode rate. The 227.4 tok/s figure quoted
elsewhere for B550 is a different experiment: the full-VRAM clean MTP control
from the previous entry, a two-GPU run with no batch slots. A single-GPU
`--batch-mtp` config (13,236 experts resident, rest streaming) measured 111.4 /
99.0 / 106.0 tok/s aggregate at C=1/2/4. So several concurrent agent requests can
be served without any one waiting for a whole answer, but total tokens/sec stays
flat: the two-GPU solo path (~200 tok/s) remains the fastest single stream, and
batch slots trade per-request speed for fairness. Whether MTP drafts run inside
batch slots is unconfirmed; the batch-slot timing path reports `draft_n=0` and
does not populate the per-request draft counters, which is not proof that
speculation is off.

Second, is CPU-only embedding reasonable? Two distinct embedders run on B550. The
sindexer live collection (`jina1536-mxfp4-v1`, 1536-dim) is served by the CUDA
Jina MXFP4 embedder on ports 1236/1237 behind nginx 1238; a separate always-on
CPU embedder (llama.cpp `Qwen3-Embedding-0.6B-Q8_0`, 1024-dim, 16 threads) runs
on port 1235 and is bound to a different collection, so the two are not
drop-in interchangeable without a reindex. Measured, the CPU Qwen embedder has
lower single-request latency (0.020 s versus 0.081 s through the GPU balancer)
but is about 4.7x slower on bulk throughput (GPU 183.1 versus CPU 38.8 items/s on a
batch of 32; GPU/CPU = 4.72). CPU-only embedding of the small model is therefore reasonable for
interactive or light use, not for large reindex sweeps.

Third, is the MXFP4 embedder CPU-capable, and could it move to Strix? As shipped
it is CUDA-only: `cuda_embed_server.py` raises if `torch.cuda.is_available()` is
false and runs the Triton `matmul_ogs` MXFP4 kernel on `cuda:0`. There is no CPU
execution path for the MXFP4 Jina build; the CPU embedder is the separate Qwen
model. Strix has an RTX 5070 Laptop (8 GB, CUDA) and an Intel Arrow Lake NPU
(`/dev/accel/accel0`, `render` group). The MXFP4 Jina embedder could in principle
run on the Strix RTX 5070 if VRAM and Triton MXFP4 support SM120 (Blackwell
laptop); the NPU path is a separate, unverified question. Strix was probed
read-only this turn; no embedder was deployed there and no SM120 MXFP4 execution
was tested.

During the run an unrelated fleet defect surfaced and was worked around. After a
benchmark restart the GPU embedding units crash-looped because
`kernels.get_kernel('kernels-community/gpt-oss-triton-kernels')` could not verify
publisher trust: the stored HF token at `~/.cache/huggingface/token` (dated Oct 6)
is invalid, and the org-overview trust endpoint returns 401 with the token but 200
anonymously. A reversible systemd drop-in
(`/etc/systemd/system/sindexer-embed-gpu@.service.d/hf-anon-trust.conf`,
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`) makes the trust check anonymous and loads the
cached kernel; the credential file was not modified or deleted. Both units were
verified active, the 1238 balancer round-trips a real 1536-dim embedding, and
sindexer update/search succeed end-to-end. The invalid token itself still needs
the user's `hf auth login --force`; the drop-in is a workaround to remove once a
valid token is set. Full measurements, hashes, and non-claims are in
[concurrency and embedding evidence](../evidence/qwen-b550-concurrency-fallback.json).

### Strix MXFP4 embedding tier, 2026-10-19

The question was whether the sindexer Jina MXFP4 embedder is built for
acceleration on Strix, and whether it could host there. Both are answered yes on
the CUDA path, by measurement rather than by version reading. The Triton MXFP4
`matmul_ogs` kernel from `kernels-community/gpt-oss-triton-kernels` compiles and
runs on the RTX 5070 Laptop's SM120 silicon (compute capability 12.0): a direct
probe with packed E2M1 weights and E8M0 scales returned
`MXFP4_MATMUL_OK shape (8, 64) dtype torch.bfloat16 finite True`. The unmodified
`cuda_embed_server.py` then served
`jina-code-embeddings-1.5b-block-gptq-mxfp4-32k` on Strix, returning 1536-dim
embeddings that agree with the B550 RTX 3090 Ti reference at cosine 0.99990560
and 0.99990414 on two code inputs (max absolute element difference 1.241e-3 and
2.521e-3; the residual is likely MXFP4 kernel ordering between Ampere and
Blackwell, a hypothesis no ablation isolates).

Strix's environment needed one alignment to the proven B550 combination:
`transformers` 5.20.2 there requires `kernels` >= 0.17.2, a different package
layout without `kernels.utils`, so Strix was pinned to the B550-proven
`transformers` 5.17.0 with `kernels` 0.16.0 in an isolated probe environment
(`/home/kearm/.probe-mxfp4-sm120`). Strix also carries the same invalid Hugging
Face token as B550 (`hf auth whoami` reports "Invalid user token"), so the
embedder runs with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and the publisher-trust
check runs anonymously; the credential file was not modified.

That yields three embedding tiers with different characters. The B550 RTX 3090 Ti
MXFP4 path (1536-dim) is the bulk tier at 0.081 s single-request and 183.1
items/s on a batch of 32. The Strix RTX 5070 Laptop MXFP4 path (1536-dim, the
same model) is the interactive tier at 0.036 s single-request, 2.25x lower
latency than the B550 GPU path, though far lower on bulk at 15.9 items/s. The
always-on B550 CPU embedder (llama.cpp `Qwen3-Embedding-0.6B-Q8_0`, 1024-dim,
16 threads) is fastest per request at 0.020 s and reaches 38.8 items/s on bulk,
but it is a different model and dimensionality, so it is not a drop-in for the
1536-dim collection without a reindex. Strix can therefore host the exact
production embedding model, which the CPU tier cannot.

The Strix embedder runs standalone on `127.0.0.1:1236`; it was not put into the
sindexer routing path, no nginx balancer or tunnel was created, and the Intel
NPU path was not exercised. The model (1.1 GB) and the server module were
placed on Strix, the model by a B550 to Mac to Strix relay because this Mac's
`rsync` is openrsync 2.6.9 and cannot do remote-to-remote copies. Full
measurements and non-claims are in
[Strix MXFP4 embedding tier](../evidence/qwen-strix-mxfp4-embed-tier.json).

### B550 layer-split topology optimization, 2026-10-20

B550's two cards are not symmetric on the host bus, and the difference is
larger than the card order. The PCI tree puts GPU0 (`00000000:04:00.0`, bus 04) behind the
AMD 500-series chipset at Gen3 x4, 2.9 GB/s host-to-device as the engine probes
it, while GPU1 (`00000000:08:00.0`, bus 08) is CPU-root at Gen4 x16, 25.4 GB/s. The engine's
auto placement search already scales each stage's miss cost by its probed link
(x6.96 against x1.00 here), so the question was whether an explicit placement
beats the search on this asymmetric pair.

On a GPU verified clean (a pre-flight guard aborts if any foreign process holds
VRAM) and with the embedding replicas stopped for the window, five arms of the
pinned workload ran: the auto control, the same with the fast card first, an
explicit split of 30 layers with the fast card first, and the miss-cost constant
raised to 600 and to 2000. The explicit split-30 configuration was the winner.
Decode measured 236.0 tok/s median against the control's 226.9, and 243.6
against 227.3 when the arms were re-run with the control last, with prompt
throughput 165.9 against 151.8 tok/s, a 9.29% gain. The control is stable at
226.9 and 227.3 across the two orderings, and the control's expert prefill was
still 6.1 s when it ran last, so the gain is not a warm-page-cache artifact.
Reversing the card order alone changes little (230.1 to 229.3, about 1%),
because the auto search already reaches K=24 with 100% of the profiled pairs
resident, and raising `STRATA_SPLIT_MISS_MS` leaves the placement decision
unchanged, so that constant is not a live lever on this rig.

The mechanism is a hypothesis, not an isolated proof: giving the Gen4 x16 card
30 of 48 layers moves layer work and the expert-miss stream onto the fast link,
and it also changes the per-card cache sizing (14,499 slots and 599 MiB free
against 12,288 slots and 3,649 MiB) and the draft window (67 to 68 accepted of
79 to 80, against 66 of 78). The recommendation is to run `--gpu 1,0` with an
explicit `--layer-split 30` on this rig; nothing in the deployed configuration
was changed, and the result is specific to this model, this engine revision, and
this short-prompt workload.

One measurement hazard is worth recording because it nearly produced the wrong
conclusion. An earlier pass of the same arms measured 50 to 73 tok/s and
placement logs showing only 2.99 GiB free against the control's 20.04 GiB. The
cause was another session's live Ming ASR teacher transcription, which held
about 17.5 GB on each card; the 17,459 MiB difference in free VRAM matches that
footprint exactly. That job was not stopped. It was allowed to finish on its own
(500 of 500 on both arms), and the arms were re-run on a verified-clean GPU with
a guard that aborts on any foreign VRAM holder. Full measurements, the
contamination record, and non-claims are in
[layer-split topology evidence](../evidence/qwen-b550-layer-split-topology.json).

### B550 two-request queueing decision, 2026-10-20

The C=2 penalty is real in the built-in batch path. With the topology-aware
default (`--batch-groups auto`, two groups of one slot), aggregate throughput
fell from 142.66 tok/s at C=1 to 127.02 at C=2 and per-request decode fell from
176.6 to 75.65 tok/s. Making the two slots share one pipeline group was worse
(102.17 tok/s), and the split-30 placement that wins solo regressed to 106.29
tok/s at C=2. The default grouping is therefore already the best built-in
topology-aware batch mode.

Two process-level alternatives were also measured and rejected. Serving an
independent single-GPU model on each card reached only 109.11 tok/s because each
copy held just 13.7k of 24.6k experts and the two arenas drove the host to 71
GiB RAM plus swap. Sharing the 32.23 GiB CPU expert arena through
`--shared-expert-arena` removed the duplicate backing store and passed the pack
hash check, but measured 104.79 tok/s; duplicate RAM was not the primary
bottleneck.

For the measured exact short-prompt shape, the better topology-aware policy is to
avoid batch slots and queue serially on the two-GPU split-30 solo/MTP path. An
exact repeated binary-search pair completed ten requests at 222.53 aggregate
tok/s with 252.6 tok/s median decode; one request returned in about 0.42-0.44 s
and its queued peer in about 0.83-0.88 s. The same varied-prompt driver that
showed the C=2 batch loss measured split-30 C=1 at 137.49 tok/s against
106.29 tok/s for split-30 C=2, but it did not measure two varied requests queued
serially, so it is not a serial-C=2 control. The measured recommendation is
limited to the exact short-prompt shape; a varied-prompt serial C=2 control and
longer prompts remain open. Batch slots remain the fairness mode when avoiding
queue wait across more conversations matters more than aggregate latency. A
future engine improvement would need topology-aware batch MTP across a layer
split; the current `--batch-mtp` implementation is explicitly one-GPU-only.

The refreshed Hugging Face credential was copied from the Mac to B550 and Strix
with mode 0600. Both report `user=Kearm`; B550's anonymous-token drop-in was
removed, both embedding units restarted, and both the B550 balancer and Strix
standalone embedder return 1536-dim vectors. During the shared-arena run B550's
Realtek `enp7s0` link also dropped for 52 seconds and returned. The kernel log
shows the NIC link transition and no OOM, GPU reset, or panic; the experiment
had completed and its restore trap cleaned up. That NIC flap is an operational
hazard, not a diagnosed or repaired cause. Full measurements and non-claims are
in [two-request queueing evidence](../evidence/qwen-b550-c2-queue-topology.json).

### B550 long-session serving at 256K context, 2026-10-20

The short-prompt measurements elsewhere in this log do not cover the workload
that matters most: an extremely long coding session that prefills a large
context once and then decodes against it. That case was measured directly on
B550, with the pinned Qwen IQ2_XS/Q2_0 model, the two-GPU split-30 placement,
`--max-context 262144`, `--kv int8` and `--kv-resident 32768`. The prompt was
112,626 tokens of real C++/CUDA engine source (34 files) followed by a review
question, generated 128 tokens at temperature 0.

The KV cache is handled by splitting it between VRAM and pinned host RAM. There
are 12 QSA layers (every fourth of 48) and 1,056 bytes per cell in INT8, which
is 12,672 bytes or 12.375 KiB of KV payload per token, so the whole
262,144-cell table is 3.09 GiB of payload. With 32,768 hot cells per layer kept
in VRAM, the engine reported "32768 of 262144 cells per QSA layer in VRAM, the
K/V in 1.80 GiB of pinned RAM", and 92.14 percent of 740,474 KV block reads
were served from VRAM. The engine's 1.80 GiB is its measured host allocation
for the streamed representation, not the 3.09 GiB payload arithmetic, which
does not model page granularity or the resident-cell representation; host
planning should reserve at least the measured 1.80 GiB and treat 3.09 GiB as
the upper bound. The result was a 27.743 s prefill (4059.5 tok/s) and 128
tokens at 117.1 tok/s, with a 99.5 percent decode expert-cache hit rate.

On the question of whether the host's RAM is being used: it should not be, for
this workload. B550 has 91 GiB with about 85 GiB available while idle, the KV
needs 1.80 GiB of it, and the expert arena needs about 32 GiB. RAM is not the
constraint. Using more of it actively hurt: `--ple-io ram` locked the 28.8 GiB
n-gram table in 18.2 s and then prefilled 2.52x slower (1609.2 against 4059.5
tok/s) and decoded 1.89x slower (61.9 against 117.1 tok/s) than `--ple-io
direct`. Locking that much RAM evicts the page cache that the arena and the
table are read through, so a 2.5 GiB NVMe with a warm file cache beats a
91 GiB RAM table on this host. The community 2x Titan RTX run that uses
`--ple-io ram` had 125 GiB, so its choice is not portable to B550's 91 GiB.

What the numbers say is that KV and RAM are not where the time goes at long
context. Decode at 117.1 tok/s, with a 99.5 percent expert-cache hit rate and
92 percent of KV reads hitting VRAM, is bounded by the launch-bound per-layer
pipeline and by the MTP drafter's 61 accepted drafts of 87 offered. The next
optimization target is the decode pipeline, not memory. Full measurements,
including the explicit hugepage state and what was not measured, are in
[long-session evidence](../evidence/qwen-b550-longrun-256k.json).

### Cold KV disk tier, native packs, and Optane operations, 2026-10-20

An earlier revision of this record treated Optane as a parking-only destination
and proposed a canonical Q2_0 pack for the cold tier. Both assumptions were
wrong and are corrected here.

First, the staged artifact is enough. The IQ2_XS pack's `--spec T` requirement is
an EXPERT-path requirement, not a KV-path one: a native (IQ) pack serves its
experts from the GGUF and decodes through verify windows, and those windows
resolve the MAIN QSA layers' K/V through `qsa_kv_resolve` (verify.cpp:1231,1244),
which is exactly the state the pager owns. An earlier revision refused `--spec`
with `--kv-disk` on the reasoning that a native pack "needs the MTP ring", which
conflated the verify window with the drafter. The drafter's own state is a ring
over its last window, sized by that window rather than by the context, so the
ring is still refused while `--spec` is allowed.

Second, the pager has an explicit durability position. The disk granule is 32
blocks = 135,168 B = 33 x 4 KiB exactly, because a single 4,224 B block is 128 B
past a 4 KiB boundary and could never be an aligned direct transfer. A group's
payload is written whole and its valid bit only after, and every read verifies a
CRC over the payload and refuses the group rather than returning torn bytes.
That tolerance ends at the filesystem: exFAT has no journal, so the mount's
unmount discipline, not the pager, protects the directory. Five unclean
unmounts were observed in one session on this host and the volume then surveyed
clean (`fsck.exfat -n`: 21 directories, 101 files). The operating rules are
`docs/OPTANE_KV_TIER.md`.

Two defects were found by the pager's own tests rather than by a model run, and
both are recorded because the second would have produced plausible wrong
answers: `scatter_block` did not skip the first block's byte run, so a write and
its read agreed on the same wrong offset and a torn payload passed undetected;
and the loader recomputed the file layout instead of adopting the header's own
values, so a corrupt header was reported as a configuration mismatch. The header
checksum is now verified on open, and the header's `payload_offset` is the one
source of truth for the reader and the writer.

Nothing here is a measured decode result. The engine builds on B550 and the
pager's unit tests pass there, but the cold tier has NOT yet been compared
against the RAM path on real text, and no throughput, latency, or context
figure for it is claimed.

### B550 Optane session parking and the Mooncake boundary, 2026-10-20

Strata already has two distinct KV placement mechanisms, and only one of them is
a disk tier. During live inference, `--kv-resident` keeps hot QSA cells in VRAM
but the authoritative streamed K/V is device-mapped pinned host RAM; the
resolver copies misses from that RAM pool, not from NVMe. Disk session files are
the second mechanism: an explicit save serializes the running state, deepest
checkpoint, every authoritative QSA layer's K/V, and draft K/V, and an explicit
restore reads that file into a new engine process.

The Optane path was exercised with a single RTX 3090 Ti because session files
currently reject `--layer-split`, `--peer-device`, and `--batch`. On the HPE
OEM Intel 750 GB Optane partition (`ET000750KWJTF`, SMART passed with zero
media/data-integrity errors), a 112,626-token real-source conversation at
256K context became a 1,956,182,716-byte file in 1,462.9 ms. After the engine
was stopped and restarted, restore returned all 112,626 tokens and exactly the
same byte count in 2,231.6 ms; the engine attributed 2,114.9 ms to read plus
validation. Effective end-to-end rates were 1.245 GiB/s save and 0.816 GiB/s
restore, including the exFAT path, flush/rename, hashes, and validation. This
proves native whole-session parking on Optane, not live per-token paging.

The measured file is 17,368.84 bytes per token including running state,
checkpoints, and draft K/V. With the 669 GiB available before the run, that
shape represents about 367 identical 112,626-token sessions; a linear 256K
projection is about 4.55 GB, but that projection was not measured. The current
B550 has no capacity pressure that justifies moving the live authoritative KV:
one single-GPU 256K engine reported 3.09 GiB pinned KV and the host has 91 GiB
RAM. Many simultaneously resident sessions, not one long coding session, would
be the reason to add a native pager.

Mooncake is therefore not a B550 dependency. Its Transfer Engine provides
segments and asynchronous batch transfers across DRAM, accelerator memory, and
NVMe-oF; Store adds a distributed object pool, placement Master, replication,
and local SSD/NVMe tiers. Those facilities do not understand Strata's QSA block
layout, model/config fingerprints, residency tables, or validation protocol.
Bringing in Mooncake now would add a control plane without solving the
engine-side pager. The correct sequence is native session files first, a
Strata-native asynchronous block pager if local multi-session RAM pressure is
measured, and Mooncake only when a second inference host needs a shared or
replicated KV pool. Full measurements and non-claims are in
[Optane session evidence](../evidence/qwen-b550-optane-session-store.json).

### Disaggregated KV target architecture supersedes the prototype decisions, 2026-10-20

The conclusions above about a bespoke pager and a multi-host-only Mooncake
boundary are historical observations, not the current architecture. The target
is now [KV disaggregation](KV_DISAGGREGATION.md): one coordinator owns L0 VRAM,
L1 pinned host slots, L2 local Optane/NVMe groups, and optional L3 shared
storage through a common `KvBlockStore` ABI. Mooncake is not rejected because
it lacks QSA semantics; the Strata adapter supplies the block key, geometry,
epoch, and validation, while the store treats payloads as opaque bytes.

Four corrections to the experimental iteration are binding:

1. The serving path is authoritative. `verify.cpp` appends QSA K/V directly
   with `kv_append_q8_steps`/`kv_append_q8_step` and resolves through
   `qsa_kv_resolve`; hooks in `qsa_layer` alone cannot serve the model. The
   existing prototype is therefore quarantined until its append/read boundary
   moves into verify and shares one coordinator with the token path.
2. KV format is an input, not an architecture decision. FP16, INT8, Q4_0, and
   K8V4 all have format-specific aligned group sizes; the first INT8-only
   implementation was a control-compatibility shortcut, not a constraint.
3. The MTP drafter ring is separate bounded state and can coexist with the
   main-layer tier. It must not be managed as if it were a full-context QSA
   pool, but its presence is not a reason to refuse `--mtp` or `--spec`.
4. The observed 5.20-versus-14.25 perplexity comparison is void: the cold arm
   truncated attention to the small host cache and did not exercise paging.
   It is retained only as a failure example, not as a measurement.

Phase A must establish the coordinator and actual serving-path hooks; Phase B
must pass a matched live-model NLL control with forced cold reads and
evictions; Phases C-E add asynchronous I/O, shared storage, and placement
optimization. No current implementation satisfies those gates.

### Upstream v0.1.45 performance intake, 2026-10-20

The implementation base is now isolated from both the quarantined pager and
upstream's 0.2.0 release tree. `upstream-v0.1.45-engine` imported the coherent
`v0.1.45` runtime/tests/server tree and sibling SM86 stack through
`2725c23f`; `feature/kv-disaggregation-v0145` carries the disaggregation
architecture on that base. A source comparison proves the engine tree is the
upstream SM86 tree plus the owned accelerator-plugin ABI and DeepSeek host
contracts, not a partial hand merge.

Post-`v0.1.45` performance intake is selective. The host-aware prefill planner
is the first candidate, followed by a measured CPU-pool planner and opt-in
expert-I/O mechanics. The Linux expert-tier default flip stays out. Split
overlap, prefill pipelining, and concurrent prompt reads stay off until the KV
coordinator owns per-stage/per-slot state and B550's serial queueing policy is
retested. The isolated opt-in batch/stage telemetry patch (`79d8ea06`) was
imported because those controls need per-window and per-stage evidence; it is
disabled unless `STRATA_BATCH_TELEMETRY=1`. The latest app/serve iterations are
packaging, protocol, monitor, browser, picture-reader, or release changes
rather than B550 engine performance inputs. Full commit clusters, controls, and
the B550 reachability blocker are recorded in
[upstream intake evidence](../evidence/upstream-v0145-performance-intake.json).

### Disaggregated KV coordinator, Phase A slice, 2026-10-20

The merged v0.1.45 base changes one premise of the disaggregated KV plan. The
imported engine already implements the L0 and L1 tiers for a streamed QSA
layer: the authoritative K/V is in pinned, device-mapped host memory, VRAM
holds a bounded pool of page slots, and `kv_stream_resolve` makes the selected
blocks resident with a device-side CLOCK sweep before the attention runs. The
coordinator therefore does not reimplement residency. It owns the store, the
epoch rules, the readiness contract, and the counters that prove a cold read
happened. The device map stays the L0/L1 mechanism, and the group-to-page
relationship must be documented and tested so the two granularities cannot
drift.

The first slice is now implemented on `feature/kv-coordinator-v2`: the
`KvBlockStore` ABI with an in-memory reference store, the block identity with
epoch rules, and the `KvTierCoordinator` state machine, with nine host contract
cases. The contract that is enforced today is: a whole group is published as
one store put, a partial group is never published, the open tail is pinned and
never evicted, a rewind advances the epoch so the old bytes are unreachable, a
reset invalidates the store's groups, a checksum mismatch is a refusal rather
than a zero fill, and two coordinators with different stage or slot scopes
never share state.

Two defects were found by those tests rather than by a model run. The
quarantined prototype indexed its completion bitmap by the block inside its
group, so two groups of the same layer shared a flag; the new coordinator
indexes by the block itself. And a cold-read counter that moved only on success
could not prove a foreground cold read when the read was refused, so the
counter now moves when the read is issued and a separate counter records the
refusals.

The coordinator is not wired into the engine yet. The serving append, selection,
and resolve boundaries in `verify.cpp` are unchanged, and no KV byte has moved
between tiers. The host gate now includes the coordinator's contract test.
Full commands and non-claims are in
[phase A evidence](../evidence/kv-coordinator-phase-a.json).

### Disaggregated KV coordinator: Phase A build-out, 2026-10-20

The Phase A slice is now implemented on `feature/kv-coordinator-v2`, on top of
the merged v0.1.45 base. It adds the `KvBlockStore` ABI with an in-memory
reference store, the block identity with epoch rules, the `KvTierCoordinator`
state machine, the session that owns one coordinator per layer, the L2 local
file store, and the serving-path hooks behind an opt-in switch.

The switch is `--kv-tier <path>` with `--kv-tier-l1 <groups>`; it needs
`--kv-resident`, because a cold tier publishes the layer's host copy and only
a streamed layer has one. With the tier off, the serving path costs one branch
per window and nothing else.

The hooks are at the places the serving path really moves K/V: `verify.cpp`'s
`kv_append_*` calls and its `qsa_kv_resolve` calls. The append hook reads the
window's first position from the host-mapped step buffer, because the device
`step_` buffer cannot be read from the host. The resolve hook maps the
selection's cells to their blocks, deduplicates the groups, and makes each
group ready before the attention runs.

The L2 store writes whole groups. A group's payload is durable before its
valid marker is committed, the header is the only source of truth for the
layout, a group of another geometry, format, layer, or epoch is absent rather
than misread, and a partial read is refused, because a whole-group checksum
cannot verify it.

Four defects were found by the contract tests rather than by a model run. The
quarantined prototype indexed its completion bitmap by the block inside its
group, so two groups of the same layer shared a flag. A cold-read counter that
moved only on success could not prove a foreground cold read when the read was
refused, so the counter now moves when the read is issued and a separate
counter records the refusals. A file store that mixed the layer and the group
into its file index grew the file without bound, so the store holds one layer
and refuses a key of another. And a group checksum that covers the whole group
cannot verify a partial read, so the store refuses partial reads outright.

The hooks have never run against a model. The CUDA build, the identity gate,
the matched NLL control, the forced cold reads, and the 600K and 800K gates are
all outstanding, and the cold tier's benefit is unmeasured. B550 was
unreachable on both the LAN and ZeroTier paths for the whole session, so no
hardware acceptance gate ran. Full commands and non-claims are in
[phase A evidence](../evidence/kv-coordinator-phase-a.json).

### Disaggregated KV tiers: shared store and placement policy, 2026-10-20

Two more pieces of the disaggregated KV architecture are now implemented on
`feature/kv-coordinator-v2`, alongside the coordinator, the L2 file store, the
read batch, the serving-path hooks, and the graph-split proof.

The L3 shared store keeps the semantic boundary in the right place. Strata owns
the key, the geometry, the format, the layer, the group, the epoch, the
checksum, and the quota; the transport carries opaque bytes under a key it does
not parse. The epoch is part of the object key, so a rewind's bytes are a
different object and can never be served as current data. A real Mooncake client
implements the narrow transport interface, and the in-process transport is the
reference for what that interface must provide. The adapter's contract is
testable before a second host exists, which is the point: the shared tier must
not need a second machine to be correct.

The placement policy ships with the matched control a promotion needs: the same
workload, the same cache size, and the same access pattern with only the policy
differing. CLOCK is the baseline, because the engine's own device residency map
uses it and matching it keeps the cold tier comparable with the tier the engine
already has. A policy that does not beat CLOCK in that control is not promoted.
A control that a policy wins against itself is a broken control, and the
contract test refuses that case outright.

The repository gate now runs seven contract suites for the KV tiers: the
coordinator, the session, the file store, the graph split, the read batch, the
shared store, and the placement policy.

None of this has run against a model. The CUDA build, the identity gate, the
matched NLL control, the forced cold reads, and the 600K and 800K gates are all
outstanding, and the cold tier's benefit is unmeasured. B550 stayed unreachable
on both the LAN and the ZeroTier path for the whole session: the Mac's own LAN
moved to a different subnet, so the 192.168.1.0/24 home network is unreachable
from here, and the B550's ZeroTier node appeared once as a relay peer and then
left the peer list again. A persistent watcher with both lanes is running and
will capture the host's uptime, its boot window, the previous boot's kernel
errors, the Optane mount, the SMART state, and the GPU state on the first
successful login. Full commands and non-claims are in
[phase A evidence](../evidence/kv-coordinator-phase-a.json).

### Disaggregated KV tiers merged, and the hardware gate still open, 2026-10-20

The disaggregated KV work is merged. PR #10 brought in the v0.1.45 engine base
and the upstream intake ledger, and PR #11 brought in the coordinator, the store
ABI, the L2 file store, the L3 shared store, the read batch, the placement
policy, the serving-path hooks, the graph-split proof, the failure-injection
suite, and the live-model acceptance gate. The merged main is `b6927556`.

CodeRabbit reviewed PR #11 and posted thirteen actionable findings. All
thirteen were repaired and answered with the repair evidence, and its re-review
posted no new findings. The findings were real, and the most serious were that
the tier hooks ran inside the captured graph rather than at a host boundary, and
that the resolve hook read a device-resident selection from the host. The cold
read sent the checksum of the L1 slot rather than the checksum of the group it
published, so every shared-store cold read failed; the regression test for that
one fails on the old code and passes on the new. The file store's group
directory held only 102 entries, far below what the 600K target needs.

Six OpenCodeReview profiles ran against the PR and none posted a finding, for
infrastructure reasons: two failed with agent errors and three have been
looping on a checkpoint fingerprint mismatch with empty attempt logs. Under the
review policy these are `deferred/skipped-unavailable`: not approvals, not
waivers, and not merge gates.

The hardware gate is still open. The B550 did not come back after the reboot.
The main machine's own LAN moved to a different subnet, so the home network
that holds the B550 and the Strix is unreachable from here, and the B550's
ZeroTier node appeared three times as a relay peer and never carried traffic.
A persistent watcher with both lanes is running and will capture the host's
uptime, its boot window, the previous boot's kernel errors, the Optane mount,
the SMART state, and the GPU state on the first successful login. That capture
is what will answer what happened in the previous session's IO errors.

Until the host answers, the following remain unrun and unclaimed: the CUDA
sm86 build, the SM86 identity gate, the matched NLL control, the forced cold
reads, the 600K and 800K scale gates, and any measurement of the cold tier's
benefit. Full evidence is in
[phase A evidence](../evidence/kv-coordinator-phase-a.json) and
[B550 reachability evidence](../evidence/b550-reachability-2026-10-20.json).

### Cold KV tier corrections, engine-host restore, and the B550 acceptance harness, 2026-10-20

The merged disaggregated KV tier had four defects that no host test could see,
because the hooks had never run against a model. A parent audit of the merged
code found all four, and an independent read-only worker confirmed the two
serving-boundary ones from its own trace of the same files.

The commit boundary addressed only the stage's first QSA ordinal, so a layer
split served one layer's cold tier and left the rest untouched. The session
opened exactly one file store, whose identity was layer 0, and then created a
coordinator for every layer that asked for one, so a non-primary layer's
publish was refused by the store's own identity check. The selection matrix
was one buffer shared by every layer, so each layer's top-k overwrote the
previous layer's selection, and the resolve hook read one row of a multi-row
window's selection. The store's path was used as a file prefix, which put the
per-layer files beside the tier directory instead of inside it, outside the
mode-0700 directory the operations procedure requires.

The corrections are: one fixed-layout file per QSA layer inside the tier
directory, a per-layer selection matrix indexed by layer and window row, a
commit boundary that covers every QSA ordinal the session owns, and a resolve
that reads every row of the window's selection. The layer-split and batch
modes are refused explicitly, because the coordinator's key must carry a stage
and a slot, and the singleton session has neither. The session's identity is a
random per-process value, so two engines cannot alias each other's object
keys in one store.

The largest finding was that the tier's own bytes never reached the engine.
The engine's attention reads the pinned host copy through `kv_stream_resolve`,
inside the captured graph, and the coordinator's L1 was a parallel copy that
nothing consumed: the tier was write-only. The resolve hook now scatters each
ready cold group back into the engine's exact per-run host layout, in the
same run order the append used, and counts the restores as
`engine_host_restores`. That counter is the difference between a tier that
publishes bytes and a tier the attention actually reads, and the B550 harness
now refuses a cold arm that reports zero restores.

Two more corrections came from the workers. The shared store's quota was
per-store accounting with a hard refusal, which is not a pool; it is now a
pool-wide admission with bounded LRU eviction across every layer store, one
immutable object per key, and a counter for the evictions the admission
caused, which is the Master-side role the architecture assigns to it. The
asynchronous slice is now a real implementation rather than declarations: one
serialized worker owns the store I/O, a read owns its L1 slot until its
completion is consumed, a put owns an immutable snapshot of the group rather
than the evictable slot, and a canceled operation never installs bytes.

A bounded acceptance harness is now in the repository at
`scripts/b550_acceptance.py`. It verifies the host, the two RTX 3090 Ti cards,
and the designated 750 GB device by live model and serial, builds the CUDA
engine for sm_86, runs the identity gate, runs the matched cold control with
the tier off and on, and compares prompt tokens, output tokens, argmax values,
and mean NLL. It fails closed: a cold arm without real counters, without
nonzero cold reads, without nonzero L1 evictions, without a complete
graph-split window, or without an engine host restore is a failed receipt, and
the receipt is written either way.

B550 was unreachable for the whole session. The main machine's LAN moved to
192.168.12.0/24, so the home network is unreachable, and the host's ZeroTier
node was absent from the peer table. One wake-on-LAN packet was sent to the
known NIC address and a bounded watcher polled both lanes without a login. The
repository's own watcher had a defect that made it exit before its first
probe: it stamped its log lines with GNU `date -Is`, which macOS rejects, and
it now uses a portable format.

What remains unclaimed is unchanged and now larger. The CUDA sm86 build, the
identity gate, the matched NLL control, the forced cold reads, the 600K and
800K gates, and any measurement of the cold tier's benefit are all unrun. The
engine's pinned host allocation is not reduced by this work: the tier's
restore path repopulates the copy the attention reads, and evicting a group
from the tier's L1 does not yet free the engine's pinned host bytes. The
Optane-class naming of the 750 GB device is also unverified: the recorded
identity is `HPE OEM Intel ET000750KWJTF`, and no live SMART read in this
session confirmed whether the device is 3D XPoint. Full commands and
non-claims are in
[implementation evidence](../evidence/kv-b550-implementation-2026-10-20.json).

### CodeRabbit review repairs for the B550 cold tier, 2026-10-20

CodeRabbit reviewed PR #12 at `8a30e3d2` and posted nine actionable
findings. All nine were repaired together at `7db98a56`.

The critical finding was a compile defect: the per-process session identity
constructed `std::random_device` temporaries instead of calling the device, so
`generate.cpp` could not compile. The expression now calls one device twice and
reserves zero. The review also found that the commit boundary copied the
per-layer selection matrix with `last_t_` as its stride while the device layout
uses `max_t_`; `cudaMemcpy2D` now copies each layer's active rows with the
correct source and destination strides. Cold restoration stops at
`geometry.blocks`, so an unaligned final group cannot restore 32 blocks into a
shorter pinned allocation.

The asynchronous operations now reserve distinct L1 slots for distinct batch
groups, release a reservation on failure or cancellation, preserve an explicit
read completion until it is waited on (or recognize a flush-installed group as
ready), and flush a publication before the host boundary reports success. The
B550 harness asks `lsblk` for `TYPE` and parses the `sha256sum` path marker;
both defects had been hidden by hand-built unit-test JSON and single-space hash
fixtures. The durable identity receipt now states the device model and serial
in the same order as the recorded receipt and code.

Validation at `7db98a56`: `just quality` passed, all eight KV contract suites
passed, all thirteen harness unit tests passed, Ruff format and lint passed,
and every touched host KV translation unit compiled with warnings enabled. No
CUDA backend or model ran. The separate one-hour async worker timed out with a
hanging partial implementation and was not integrated. Full evidence is in
[review repair evidence](../evidence/kv-b550-review-repairs-2026-10-20.json).

### B550 and Strix identity rediscovery, 2026-10-20

The fixed-IP rule is now operational rather than aspirational. A read-only
network worker rediscovered both hosts from current ARP/NDP, router DNS, SSH
banners, and exact host-key fingerprints; it rejected alternate overlay
addresses whose keys did not match. Parent logins then succeeded over both Mac
links, Wi-Fi `en0` and USB 2.5GbE `en13`, and verified B550 machine-id
`27e232eefb5743a48455b58e0fb997a0` and Strix machine-id
`b71b3a116ec74b7a8797f38a0af8168a`. Both targets currently route through their
wired interfaces; their Wi-Fi interfaces are down. The current IPv4 answers are
observations in the receipt, never identities for a later run. Full evidence is
in [network rediscovery](../evidence/b550-strix-network-rediscovery-2026-10-20.json).

The B550 acceptance harness also stopped carrying historical addresses. Its
`--address` is now required and documented as live-discovered, ZeroTier is
optional, lane probes that answer on another lane are evidence rather than an
automatic aggregate failure, and remote arguments are shell-quoted so CMake
values with spaces cannot be split. The Optane check now requires the requested
mount endpoint rather than accepting the root filesystem that `findmnt -T`
falls back to when the tier is unmounted. A local pinned llama.cpp checkout can
be verified and supplied to avoid a redundant fetch. These repairs, plus the
missing `bench/gdn_chunk_bench.cpp` source restored from the already-fetched
commit that introduced it, made current main configurable on B550.

Live re-identification found the designated 750 GB device at `/dev/nvme0n1`
this boot, not the prior `/dev/nvme1n1`. SMART passed with 0% used and zero
media/data-integrity errors, the exFAT survey was clean, and the data partition
was mounted at `/mnt/optane` with `noatime` and `errors=continue`. At merged
revision `f286891c`, B550 configured and built the sm_86 engine, `strata-device`
passed its selftest and enumerated both RTX 3090 Ti cards, and the new
confidence test passed in the same backend tree. The aggregate build receipt
still fails closed because two unrelated `rust-indexer` Python processes hold
1,446 MiB and 1,400 MiB CUDA contexts; they were not signalled or restarted.
Therefore no model, NLL identity, cold-read, 600K, or 800K gate ran. Full
commands and the deliberate failure are in
[sm86 build evidence](../evidence/b550-sm86-build-2026-10-20.json).

### DeepInfer portable scheduling intake, 2026-10-20

The fetch-only DeepInfer clone is pinned at
`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`, with its push URL disabled and no
submodules initialized. The portable mechanisms are confidence-scheduled
variable-length MTP, one-step CPU/GPU overlap, incremental token/metadata
assembly, launch-phase sync guards, budgeted prefill/decode scheduling, and
copy-on-write state. DeepEP/RDMA buffers, SM100/Hopper kernels, Ascend kernels,
cluster load balancing, and the model-specific Engram tables are not Strata
ports for the current Qwen path. The complete ranked map and non-claims are in
[DeepInfer audit evidence](../evidence/deepinfer-portable-optimizations-2026-10-20.json).

The first safe slice is implemented at merge `f286891c`:
`plan_confidence_varlen` is a host-only policy that gives every slot its golden
row, ranks cumulative conditional-confidence prefixes under a global row
budget, honors per-slot maxima and zero-confidence cutoffs, preserves stable
slot/position ties, and emits contiguous row counts and offsets. It has no CLI,
session, drafter, verifier, or GPU-kernel caller. The focused test passed in the
parent macOS build and the B550 sm_86 tree, and `just quality` passed after
integration. This establishes a planning contract only; it is not exercised
runtime MTP and makes no acceptance or performance claim. Full evidence is in
[confidence varlen evidence](../evidence/deepinfer-confidence-varlen-2026-10-20.json).

### Optane cold-KV control on B550, and the indexing move to Strix, 2026-10-20

The cold tier has now run against a model. The blocker before this session was
not the tier: it was that the S-indexer's Jina MXFP4 embedding service held
both of B550's RTX 3090 Ti cards, and the acceptance harness refuses to measure
a model on a contended card. The service moved to the Strix laptop's RTX 5070
instead, which freed B550's GPUs and made the hardware gate possible.

The migration is a straight move of the same service, not a reimplementation.
Strix already had the byte-identical model weights and server source, and a
previously validated SM120 Python environment. The new systemd unit
`/etc/systemd/system/sindexer-embed-gpu.service` runs the same
`cuda_embed_server:app` on Strix's loopback port 1236, and the local consumer
endpoint `127.0.0.1:1236` is unchanged. The two B550 units
`sindexer-embed-gpu@0.service` and `sindexer-embed-gpu@1.service` are disabled
and stopped, and the now-dead nginx balancer on B550's port 1238 is retired.
B550's unrelated CPU embedding service was left alone. The measured embedding
parity is a cosine similarity of 0.999916 over 1,536 dimensions.

The purpose of the move is not indexing speed. Indexing speed is not a problem
at this concurrency and is not an optimization target: the service answers a
small number of embedding requests for the code index, and a single request is
about 32 ms on Strix against about 86 ms on B550's balancer, with neither a
bottleneck. A single-concurrency comparison of the two hosts is expected to be
slower on the split arrangement, because the pieces now live in different
places for the use case. The move exists to free B550's two GPUs for model
work, and the use case it serves is many agents accessing one served model with
a mix of cache hits and cache misses that the KV tier can be optimized against.

The Strix unit was not proven to survive a reboot of that host, and no indexer
throughput was measured. The full receipt is
[the indexing migration evidence](../evidence/indexing-migration-to-strix-2026-10-20.json).

Getting there needed the identity rule to be repaired first. The host's
addresses are not stable, and macOS's resolver could not resolve the
router-served `.lan` names at all, even though the router's DNS server answered
them. The SSH aliases now resolve through the live gateway at connection time
and keep their host-key checks, and the acceptance harness compares the
address the alias actually reaches rather than the string it is configured
with. A fixed LAN address is never an identity.

The matched cold control passed on the hardware. The same engine, the same
model, the same 8,192-token real-source prompt, the same fixed seed, and the
same Optane device, with the tier off and then on, produced the same 16 output
token ids, the same 8,191 scored positions, a mean NLL delta of exactly 0.0 and
a maximum absolute log-probability delta of exactly 0.0. The cold arm's
counters are the proof that the tier was in the serving path: 24,660
graph-split windows with 24,660 selection, readiness and resolve phases and
zero split violations, 794,873 engine host restores, 756,560 L2 reads covering
102,262,702,080 bytes, 756,560 L1 evictions, and zero read failures and zero
checksum errors.

The result is not a speed claim, and the numbers point the other way. The
prompt read took 1,086,512.9 ms with the tier against 73,222.8 ms without it,
about 14.8x slower, and the decode took 7,049.2 ms against 243.3 ms, about 29x
slower. The cause is visible in the counters: with 8 L1 groups per layer and 768
published groups, 756,560 of 794,873 group reads missed the cache and came from
the device, and every one of them waited for the readiness phase. The tier
traded a device read for host memory on every selection. The engine's pinned
host allocation is still not reduced, because the restore path repopulates the
copy the attention reads. The 600K and 800K scale gates did not run, and the
8,192-token prompt is far shorter than the 262,144-cell context, so the tier's
behaviour at a long context is not covered.

The gate also found a real defect, which is the kind only a hardware run
exposes. The coordinator published each group with an FNV-1a checksum while the
file store verified with its own CRC-32, so every group read back from a store
was refused with a checksum mismatch: the cold read could never succeed. The
tiers now share one checksum function, and the file-store test pins that the
file store, the shared store and the coordinator agree.

Two harness defects were repaired along the way, both found by running it. A
512-token prompt took the batched prefill path, which neither writes the
teacher-forced log-probabilities the control compares nor runs the tier's
hooks, so the harness now owns `--short-read` and a longer real-source prompt
keeps the measurement on the verify-window path. And exFAT cannot express a
0700 directory: the mode is synthesized from the mount's `dmask`, so the gate
now checks the achievable property, that other users cannot write the tier,
instead of a mode the filesystem cannot hold.

The sm86 identity gate is still open. With the sm86 performance stack on and
off, the teacher-forced log-probabilities are bit-identical over 511 scored
positions, but one generated token differs at position 11, because the two arms
take different expert-arithmetic paths. That is a separate claim from the cold
control, and it did not pass. The full receipt, including the counters and the
non-claims, is [the cold-tier evidence](../evidence/b550-optane-cold-kv-2026-10-20.json).

### Closing the LAN exposures found by the indexing move, 2026-10-20

Two exposures turned up while the embedding service moved, and both are now
closed.

The first was a stray read-only SSH forward on the main machine's
`127.0.0.1:12399`, created by an audit worker that could not kill it from its
own sandbox. It is gone.

The second mattered more: B550's deployed Milvus compose file published four
ports (19530, 9091, 9000, 9001) on `0.0.0.0` and on IPv6, and set the
object store's credentials to the image's default `minioadmin`. Any host on
the LAN could reach the vector store and the object store with published
default credentials. Nothing depended on it: the stack had no established
connections, the parent's indexer uses its own local OrbStack Milvus with 40
collections and 1,281,865 rows, and the stack's own data is 19 MB in total.

The compose file now binds all four ports to loopback, moves the standalone
health port to 19091, and takes the object-store credentials from an untracked
`.env` with mode 0600. The three containers were recreated and are healthy, the
volumes are unchanged, and a request to `192.168.1.171:19530` from the main
machine no longer connects. The old file is kept beside it as
`milvus-compose.yml.pre-loopback-20261020`. The parent's own Milvus and its
data were not touched, and no B550 service other than the two retired
embedding units and this stack's containers was stopped. The receipt is
[the exposure-closure evidence](../evidence/b550-lan-exposure-closure-2026-10-20.json).

### Flash-Next prefiller direction corrected, 2026-10-20

The Qwen3.8-27B NVFP4 release and disaggregated-quantization paper are now
canonical templates, not runtime model families. The target is an auxiliary
Qwen3.8-Flash-Next prompt-state model trained against the frozen low-bit
decoder: prompt positions construct GDN, QSA and PLE state with auxiliary
weights, then response positions continue unchanged on the deployed decoder.
The first host-only planner selects 684 dense/shared state tensors and a
per-layer hot-expert set, estimates MXFP4 bytes from deployed metadata, binds
inputs by hash, and refuses the 27B checkpoint as a decoder.

Focused planner tests and lint passed. Against read-only metadata from the
real B550 split pack, the 64-expert profile-rank plan is 9.238 GiB MXFP4:
1.767 GiB dense plus 7.471 GiB routed experts. Its coverage is null because
the shipped profile records ranking only. No trace was captured, no model or
trainer ran, and no runtime or quality effect is claimed. Full evidence is in
[prefiller planning](../evidence/flash-next-prefiller-plan-2026-10-20.json).

### README reference split, 2026-10-20

The English README is now a compact 124-line product entry page. Its long dated
status matrix remains in this architecture authority under `Project status and
governance`, and a new `docs/README.md` index groups the existing specialized
documents by start-up, serving, hardware, memory/KV, quantization and research
use. This does not create a second roadmap or status authority: README routes to
the architecture matrix, while the index only routes to operational detail.

The repository now checks that the README stays at or below 140 lines, its
status table remains compact, every nested documentation page is represented in
the index exactly once, and local targets plus Markdown heading anchors in the
README and index resolve. Each localized README received an index route without
rewriting its translated body. Full commands and results
are in [README split evidence](../evidence/readme-reference-split-2026-10-20.json).

### Full-model runtime baseline for pruning, 2026-10-20

The full calibrated Flash-Next decoder ran a real-source 13.96K-token agent-like
sweep on B550 with four concurrent clients serialized by the split-30 FIFO.
Unique cold prompts read at a median 3,355.9 tok/s; a held 13,943-token prefix
reduced the median fresh read from 4,159.5 ms to 192 ms. Decode aggregate
expert-cache hits were 99.9211%, leaving 402 misses whose stall time was not
attributed separately. The sweep therefore cannot identify decode's bottleneck;
pruning must prove footprint, cold-agent, startup, concurrency, miss-latency, or
quality-adjusted value.

A separate single-GPU serve trace captured all 48 layers with 25 verify-window
records per layer. In that decode-only activation trace, 64 retained experts per
layer covered 92.08%, 96 covered 98.27%, and 128 covered 99.68%. The planner now
distinguishes routed-activation coverage from router-weight coverage and records
per-layer trace counts.

The current vendor Coder revision then ran as a separate hashed scratch
artifact with a remapped current profile. It was 9.7% faster on cold prompt
reads and loaded an 8.81 GiB smaller expert arena, but was 13.6% slower on the
primed concurrent-4 shared-prefix burst. No literal batched prompt-prefill
routing capture existed at that point. Full evidence and non-claims are in [runtime statistics](../evidence/flash-next-agent-runtime-stats-2026-10-20.json)
and [Coder statistics](../evidence/coder-current-runtime-stats-2026-10-20.json);
the two-lane plan is [pruning](PRUNING.md).

### Forced-window prompt routing diagnostic, 2026-10-20

A fresh single-request serve run forced a 14,020-token real-source chat prompt
through verify windows with `--short-read 16384` and `--dump-routing`. The
trace has exactly 14,020 records per layer, all 48 layers, top-10 routing, and
zero weight payloads interpreted as unit activations. It touched 23,813 of
24,576 layer/expert pairs; 64 experts per layer covered only 55.61% of prompt
activations and 256 covered 92.31%. The exact top-64 mask selected from the
earlier decode trace covered only 14.80% when applied to this prompt.

A direct one-shot negative control then ran 13,963 prompt tokens through the
normal batched path. Its routing file contained only the 48 final-generation
window records, proving that the existing host callback does not observe
batched prompt positions. The forced-window result is therefore a prompt
activation diagnostic, not literal `Prefill::run` coverage and not a timing
claim. Full evidence and the compressed raw trace are in
[prompt routing](../evidence/flash-next-prompt-routing-2026-10-20.json).

### Prefiller physical expert IDs made explicit, 2026-10-21

The host-only prefiller planner now separates physical pack slots from the
512-expert source universe. Native layouts may declare 256, 288, 384, or 512
physical experts per layer. Pruned packs require a
`strata-expert-id-map-v1` JSON sidecar with one fully validated per-layer
physical-to-source row; omitting the sidecar is valid only for 512 slots, where
the mapping is exactly identity. Profile IDs are treated as physical IDs and
translated through that map. Trace IDs declare either the physical domain or
the 512-wide source domain; source-domain evidence may contain pruned experts,
which cannot be selected but remain in the coverage denominator.

Plans keep source IDs in `selected_ids` and add parallel
`selected_physical_ids` and physical fill lists. The manifest binds both the
sidecar file and canonical mapping contents by SHA-256, and its size accounting
separates selected physical experts, the complete physical pack, and the full
source universe. Focused tests cover all supported counts, sidecar corruption,
missing-map failure, profile translation, both trace domains, and size
accounting. This remains metadata-only: no pruned model payload was built or
opened, no routing trace from a real pruned runtime existed in this checkout,
and no trainer, runtime, quality, or speed effect is claimed.

### Batched-prefill STRP trace intake, 2026-10-21

The planner now auto-detects the legacy one-position routing format and the new
batched-prefill STRP v1 wire format. STRP requires the 40-byte little-endian
header with magic `0x50525453`, version 1, prefill phase 1, no flags, 48
layers, top-k 10, 24-byte record headers, and zero reserved fields. Each
24-byte record carries layer, k, starting position, token count, and a reserved
field; token-major IDs and true float32 weights follow. The reader validates
geometry, positions, counts, payload lengths, finite nonnegative weights,
per-position expert uniqueness, all 48 layers, equal per-layer position counts,
and absence of overlap within each layer. Legacy and STRP files cannot be mixed
in one invocation.

Every STRP chunk is expanded to its individual routed positions before exact
rational scoring. Consequently `trace_records` and
`trace_records_per_layer` count positions rather than serialized chunks, and
the manifest records the detected format. The tests use synthetic binary
fixtures; this checkout still contains no real batched-prefill STRP capture,
model run, trainer, runtime integration, or measured routing result.

### Repaired STRP v2 request namespaces, 2026-10-21

Batched-prefill STRP intake moved from the transient v1 layout to the repaired
v2 contract. The header remains 40 bytes, but version is 2 and
`record_header_bytes` is 32. Each record adds a nonzero little-endian
`uint64 request_id` after the reserved field; token-major expert IDs and true
float32 router weights follow. Request ID zero is reserved and rejected.

Position overlap is grouped by `(request_id, layer)`: two requests may route
the same positions, but overlapping ranges for one request and layer are
rejected. Each request must cover all 48 layers, while aggregate per-layer
position counts remain the manifest's `trace_records` semantics. Synthetic
tests cover two requests sharing positions, reserved and partial request IDs,
and same-request overlap rejection. No real STRP file was present; no model,
trainer, runtime integration, or routing-quality result is claimed.

### Literal STRP v2 runtime acceptance, 2026-10-21

The preceding v1 and v2 intake entries describe intermediate synthetic states
and are superseded by the final contract and runtime files. Request ID 0 is
valid and means a one-shot capture; only serve capture ordinals start at 1.

The patched SM86 engine and exact base control were rebuilt on B550 from
revision 82dd8231 with the combined routing patch. CUDA builds passed
`routing_trace_test`, `strata_prefill`, and `strata`; the acceptance build used
the host-toolchain workaround `-L/usr/lib/gcc/x86_64-linux-gnu/15` because
nvcc otherwise placed GCC 13's older libstdc++ ahead of GCC 15. The ROCm
container compiled the same three targets for gfx1100 and its routing test
passed, but no AMD GPU model run is claimed. Combined-diff review then closed
two runtime-boundary issues before final rebuild: `VRAM` is refused while
routing capture is active, and HIP refuses `--kv-tier` because its verify
commit/resolve hooks are intentionally excluded.

A real one-shot CUDA capture recorded 13,963 literal `Prefill::run` positions
per layer, 96 chunk records, 670,224 routed layer-positions, request ID 0, and
true router weights. The final planner reproduced coverage of 44.4605%,
60.8184%, 71.1568%, 78.4784%, 88.2075%, and 94.0132% at 32, 64, 96, 128,
192, and 256 retained experts per layer. A two-request serve capture assigned
request IDs 1 and 2 and recorded 14,009 positions per layer for each request;
a split-30 capture recorded one request while its log placed layers 0-29 on
CUDA0 and 30-47 on CUDA1. All three traces had zero invalid IDs, non-finite
weights, or nonpositive weights.

With the capture flag absent, patched and exact-base runs produced the same
output tokens and identical window-hash and final-residual SHA-256 digests for
this one workload. This exact control does not establish speed, overhead,
model quality, pruning quality, or a trained prefiller. Full machine, GPU,
engine, model, command/log-path, artifact, and hash bindings are in
[runtime acceptance](../evidence/prefill-routing-v2-runtime-2026-10-21.json);
the durable compressed one-shot raw trace is retained beside it.

### Clean-main pre-pruning baseline and model-card contract, 2026-10-21

The full calibrated decoder remains a well-running pre-pruning control. A clean
Git-bundle checkout of merged `main` at `66067e30` built CUDA engine SHA-256
`51b5ab09ffc75044532ff4bee83a555ac8ff10b02f98fc7b3dc3a8c34a3e7f73` and passed
`routing_trace_test`. The split-30 server then ran the real 40 KB source
workload with no routing capture and no KV tier. After first-request warm-up,
four unique-cold requests read 13,983 prompt tokens each at a median 3,491.6
tok/s and decoded at 140.2 tok/s. Three shared-prefix hits reused 13,943 tokens
and read 47 new tokens each in about 216 ms, decoding at a median 153.7 tok/s.
The first post-load request is retained separately because low free VRAM and
verify-window capture warm-up made it unrepresentative.

`docs/MODEL_CARDS.md` now defines the private artifact structure and all
promotion gates for a future pruned/quantized release. The template is not an
artifact claim: no heal, custom quantization, quality gate, or trained
prefiller has run. Full machine, model, source, workload, and raw-evidence
hashes are in [current-main runtime evidence](../evidence/flash-next-current-main-runtime-2026-10-21.json);
raw request rows are retained in the [durable sweep](../evidence/flash-next-current-main-runtime-sweep-2026-10-21.json).
