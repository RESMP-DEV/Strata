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
| Accelerator plugin seam | `include/strata/runtime/`, `src/runtime/`, `plugins/` | Backend-neutral ABI v1, reference CPU plugin, probe, and external OpenVINO/NPU router plugin; no default engine call site | An accelerator becomes a routing candidate only after real-operation parity, explicit opt-in configuration, failure tests, and same-day transfer/latency accounting |
| Quantization calibration | `tools/standard_imatrix.py`, opt-in llama.cpp token patch | Pins the RESMP-DEV Qwen3.8 `[256,2048]` token view and preserves its selected windows; the public mradermacher matrix remains a separate dated input | A standard-calibrated artifact needs deterministic quantization plus quality and same-day throughput/memory controls |
| Installer | `setup.py`, shell and Windows launchers | Existing installer tests cover many choices without downloads | Run affected setup tests and a platform-appropriate smoke test |
| Local API and web app | `serve/` | OpenAI-, Anthropic-, and Responses-compatible surfaces are implemented in the inherited tree | Exercise the changed protocol and lifecycle boundary |
| Verification state | `build/reports/`, `evidence/` | Generated receipts are transient; durable evidence is tracked | Receipts must bind revision, dirty state, scope, commands, and non-claims |
| Remote hardware | SSH aliases `B550`/`compute` and `strix` | B550 owns bounded heavy-CUDA execution; Strix owns bounded heterogeneous NVIDIA/NPU/CPU validation | The main machine remains authoritative; select remotes by alias and verify identity live |
| Strix host watcher | deployed `strata-watch.service`, source `tools/strix_watch.py` | Persistent PSI/disk/SSH/journald sampling is enabled on Strix even when interactive SSH is unavailable | A hardware failure claim requires watcher evidence plus live SMART/controller/journal state |

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
| Common quality mode | `just quality` checks host-safe structure and selected regressions | Add a check only with a test and an explicit non-claim |
| Full setup matrix | Existing full discovery is not accepted as a gate on macOS Python 3.14 because native build and upstream-release cases fail | A dedicated setup change runs its named modules and the affected platform smoke test |
| Backend acceptance | No local backend build is claimed on this Mac | Require named compiler/driver/backend evidence on the target host |
| Eight-GB NVIDIA runtime | Documentation and setup accept an 8 GB NVIDIA card as slow; an sm_120 engine built and launched on the Strix laptop | Require a real model run and same-day performance measurement before claiming usable throughput |
| Strix Intel NPU integration | llama.cpp's whole qwen4exp IQ graph fails on unsupported IQ matmuls plus an NPU layout mismatch, but the explicit Strata plugin ABI now runs a representative one-token `2560x512` router on `Intel(R) AI Boost` through OpenVINO, translating BF16 ABI tensors to F16 because this driver rejects BF16 inference precision; the model engine does not load the plugin | Integration requires explicit opt-in CLI/config, real-model parity and failure tests, plus same-day CPU/CUDA/NPU latency and transfer accounting |
| Strix heterogeneous routing | NVIDIA dGPU, Intel NPU, and CPU are independently visible; Strata currently routes only its existing CUDA/CPU paths | A routing proposal needs per-part correctness, scheduling, and same-day throughput/latency controls before it can replace a default |
| iMatrix low-bit routed experts | The cross-host IQ2_XS difference is isolated to the derived-weight square-root expression before codebook selection; `STRATA_DETERMINISTIC_IQ=1` applies the validated double-precision expression to downloaded llama.cpp source while the default pinned path stays unchanged; an exact-token path now consumes the RESMP-DEV standard `[256,2048]` Qwen view | A full standard-calibrated patched rerun, perplexity/output quality, and same-day throughput/memory controls remain required before replacing a normal quantization choice |
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
