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
| DeepSeek V4.1 quantization and port | planned RESMP-owned converter/quantizer; Lucebox commit `5ada2a5` is a read-only semantic reference | External prequantized artifacts, including Lucebox ROCMFP files, are research inputs only; runnable Strata artifacts must be produced and hashed by an RESMP-owned pipeline from a pinned source checkpoint | Require exact pack inversion, deterministic conversion, backend parity, DSpark parity, quality controls, and same-day memory/throughput evidence before an executable path is claimed |
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
| DeepSeek V4.1 backend order | CUDA/CPU extraction on B550 is the initial implementation path; Lucebox HIP is reference-only | A HIP-first path is justified only if an AMD target becomes primary or an unmodified Lucebox oracle is required; either way, named AMD hardware and parity evidence are mandatory |
| DeepSeek V4.1 artifact source | Start from the pinned official 1017 checkpoint already transferring to B550, then quantize through an RESMP-owned tool | Do not promote the Lucebox ROCMFP artifact as a Strata runtime dependency; a different source or borrowed quantized artifact needs an explicit dated architecture decision |
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
