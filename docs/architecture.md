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
| DeepSeek V4.1 quantization and port | exact 1017 tensor-taxonomy gate, geometry contract, and deterministic CPU candidate-selection reference are implemented host-side; RESMP-owned converter/quantizer remains to be built, with Lucebox commit `5ada2a5` as a read-only semantic reference | External prequantized artifacts, including Lucebox ROCMFP files, are research inputs only; runnable Strata artifacts must be produced and hashed by an RESMP-owned pipeline from a pinned source checkpoint | Require exact pack inversion, deterministic conversion, backend parity, DSpark parity, quality controls, and same-day memory/throughput evidence before an executable path is claimed |
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
| Common quality mode | `just quality` checks host-safe structure, selected regressions, the plugin ABI, and dependency-free DeepSeek host contracts | Add a check only with a test and an explicit non-claim |
| Full setup matrix | Existing full discovery is not accepted as a gate on macOS Python 3.14 because native build and upstream-release cases fail | A dedicated setup change runs its named modules and the affected platform smoke test |
| Backend acceptance | No local backend build is claimed on this Mac | Require named compiler/driver/backend evidence on the target host |
| Eight-GB NVIDIA runtime | Documentation and setup accept an 8 GB NVIDIA card as slow; an sm_120 engine built and launched on the Strix laptop | Require a real model run and same-day performance measurement before claiming usable throughput |
| Strix Intel NPU integration | llama.cpp's whole qwen4exp IQ graph fails on unsupported IQ matmuls plus an NPU layout mismatch, but the explicit Strata plugin ABI now runs a representative one-token `2560x512` router on `Intel(R) AI Boost` through OpenVINO, translating BF16 ABI tensors to F16 because this driver rejects BF16 inference precision; the model engine does not load the plugin | Integration requires explicit opt-in CLI/config, real-model parity and failure tests, plus same-day CPU/CUDA/NPU latency and transfer accounting |
| Strix heterogeneous routing | NVIDIA dGPU, Intel NPU, and CPU are independently visible; Strata currently routes only its existing CUDA/CPU paths | A routing proposal needs per-part correctness, scheduling, and same-day throughput/latency controls before it can replace a default |
| iMatrix low-bit routed experts | The cross-host IQ2_XS difference is isolated to the derived-weight square-root expression before codebook selection; `STRATA_DETERMINISTIC_IQ=1` applies the validated double-precision expression to downloaded llama.cpp source while the default pinned path stays unchanged; an exact-token path now consumes the RESMP-DEV standard `[256,2048]` Qwen view | A full standard-calibrated patched rerun, perplexity/output quality, and same-day throughput/memory controls remain required before replacing a normal quantization choice |
| DeepSeek V4.1 backend order | CUDA/CPU extraction on B550 is the initial implementation path; Lucebox HIP is reference-only | A HIP-first path is justified only if an AMD target becomes primary or an unmodified Lucebox oracle is required; either way, named AMD hardware and parity evidence are mandatory |
| DeepSeek V4.1 artifact source | Start from the pinned official 1017 checkpoint already transferring to B550, then quantize through an RESMP-owned tool | Do not promote the Lucebox ROCMFP artifact as a Strata runtime dependency; a different source or borrowed quantized artifact needs an explicit dated architecture decision |
| Speculative-head/QAT direction | The native one-layer Qwen MTP is the immediate executable QAT lane; a DSpark-style rank-256 Markov plus confidence head over the promoted DFlash2 G3 full head is the next research architecture | A candidate needs exact target-controlled parity and improved paired live acceptance before integration; ordinary KL, offline replay, compression, or throughput alone is not acceptance |
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
