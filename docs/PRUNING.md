# Flash-Next expert pruning and prefiller integration

This page owns the prototype plan for combining expert pruning with the
Flash-Next phase-specialized prefiller. Runtime claims and open decisions remain
bound to [architecture](architecture.md); the prefiller state contract remains in
[PREFILLER.md](PREFILLER.md).

## Current runtime observation

On B550, the full calibrated IQ2_XS/Q2_0 Flash-Next decoder was measured with
four concurrent client requests serialized by the split-30 server FIFO. Each
request carried about 13.96K tokens of real Strata C++/CUDA source.

| Arm | Requests | Median fresh read | Median prompt rate | Median decode | MTP acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unique cold prompts | 12 | 4,159.5 ms | 3,355.9 tok/s | 132.1 tok/s | 68.27% |
| Shared-prefix prime | 3 | 4,300.0 ms | 3,249.9 tok/s | 148.6 tok/s | 70.37% |
| Shared-prefix hit | 9 | 192 ms | 161.1 tok/s over 30-32 fresh tokens | 143.5 tok/s | 75.15% |

The shared-prefix arm reused 13,943 tokens. Aggregate decode expert-cache hits
were 509,121 of 509,523, or 99.9211%. Therefore this warm workload is already
not decode-expert-miss bound. Prefix reuse is the immediate latency lever;
pruning must prove its value through footprint, cold-agent behavior, concurrency
headroom, startup, or a different routing mixture.

The durable raw sweep and decode trace are in
[runtime evidence](../evidence/flash-next-agent-runtime-stats-2026-10-20.json).
The trace has 1,200 records, exactly 25 per layer, and all 48 layers. It is a
serve verify-window trace with unit weights. Its activation-coverage ladder is:

| Retained experts/layer | Activation coverage | Dense prefiller + hot experts, MXFP4 |
| ---: | ---: | ---: |
| 32 | 74.07% | 5.502 GiB |
| 64 | 92.08% | 9.238 GiB |
| 96 | 98.27% | 12.973 GiB |
| 128 | 99.68% | 16.708 GiB |
| 192 | 100% | 24.179 GiB |
| 256 | 100% | 31.650 GiB |

This ladder is in-sample decode coverage. It does not describe prompt-prefill
routing and does not predict quality. The engine currently records routed ids
only in verify-window pool callbacks; a prompt-prefill routing capture seam is
required before training a prefiller-specific expert mask.

## Prototype A: vendor Coder as-is

[ISTA-DASLab's Qwen3.8-Flash-Next GSQ-RCO Coder GGUF][coder] is the fastest way
to measure pruning on real hardware. It keeps 256 of 512 routed experts per
layer, stores retained experts at about 3.5 bpw, and reports an effective 1.89
bpw over the original transformer. The current card reports 58.4 GB total: a
29.6 GB resident transformer shard and a 28.8 GB n-gram shard that remains
disk-served.

Use this artifact only as a measured prototype:

1. Download both current shards and verify their published SHA-256 values.
2. Prepare the normal Strata pack and Coder profile without modifying the
   vendor expert bytes.
3. Run the same 13.96K source-prompt cold/shared sweep used above.
4. Repeat at 32K context and with a colder expert/page cache.
5. Exercise two- and four-client FIFO plus explicit batch-slot controls only if
   the server reports the configuration is compatible.
6. Record prompt rate, decode rate, TTFT, cache hits, expert tier counts, GPU
   and host memory, startup time, and answer identity/quality controls.

Acceptance for continuing this lane requires a same-day full-model control and
a concrete footprint or throughput gain without an unacceptable quality loss.
The vendor artifact remains a reference prototype, not the Strata-owned runtime
dependency of record.

### Measured current-Coder result

The current Coder revision was downloaded at pinned repository
revision `bba88856f8c3c0c9e40e45f82d05e228b814ac43`, both shard hashes were
verified, and its profile was remapped through the current allocation file. The
old-profile reconstruction matched the shipped pinned profile byte-for-byte;
the current allocation changes 12,283 of 12,288 profile-order entries.

Against the full-model baseline on the same 13.96K-token source workload and
split-30 configuration:

| Measure | Full model | Current Coder | Result |
| --- | ---: | ---: | --- |
| Unique-cold fresh prompt read | 3,355.9 tok/s | 3,681.1 tok/s | Coder 9.7% faster |
| Unique-cold decode | 132.1 tok/s | 151.7 tok/s | Coder 14.8% faster |
| Four-request cold client aggregate | 10.39 tok/s | 10.95 tok/s | Coder about 5.4% faster |
| Shared-prefix fresh read | 192 ms | 210 ms | Full model 9.4% faster |
| Shared-prefix decode | 143.5 tok/s | 121.7 tok/s | Full model 15.2% faster |
| Primed concurrent-4 shared burst | 86.1 tok/s | 74.4 tok/s | Full model 13.6% faster |
| Resident expert arena | 32.23 GiB | 23.42 GiB | Coder saves 8.81 GiB |
| Resident expert coverage | 23,627 / 24,576 | all 12,288 retained | Coder has full retained-set residency |

This exact warm code-agent workload therefore does not support Coder as a
throughput upgrade. Its clear benefit is footprint: all retained experts fit in
the two-GPU cache with 1.397 GiB VRAM still free. Pruning remains attractive
for a higher-capacity custom artifact only if it converts that residency
headroom into quality or a colder/mixed-workload gain without losing the full
model's shared-prefix speed.

This run used a separately hashed research scratch artifact and remapped
current profile. It is not the older Coder revision and matching profile
registered by the installer.

## Prototype B: Strata-owned pruned decoder

The Strata-owned artifact should be generated from the pinned Flash-Next BF16
master and retain the GGUF's uniform per-layer expert count. The first matrix to
evaluate is:

| Candidate | Routed experts/layer | Expert bytes | Retained expert precision | Why evaluate |
| --- | ---: | ---: | ---: | --- |
| Coder control | 256 | 25.146 GB / 23.419 GiB | 3.3307 bpw exact (3.5 release-level) | Establishes the measured external baseline |
| Primary custom | 256 | about 29.75 GB / 27.7 GiB | about 3.9-4.0 bpw | Existing UD-IQ4_XS-style paths with more precision per retained weight |
| Coverage control | 288 | 28.289 GB / 26.347 GiB | 3.3307 bpw | More retained experts without increasing per-expert precision |
| Later capacity lane | 384 | 37.719 GB / 35.129 GiB | 3.3307 bpw | Evaluate only after the first two gates pass |

The first custom decoder should use an existing executable native recipe
(IQ3_S-class gate/up with IQ4_NL or Q8_0-class down paths), not MXFP4 or NVFP4.
MXFP4 is a planner format here, and the current NVFP4 kernels do not by
themselves provide a converter, packer, verifier, and setup path. Exact
feasibility is a byte-accounting decision, not a name decision; resident RAM,
startup, page-cache, and context budgets must be measured on B550.

The mask must come from held-out agent traces, not only the current 25-record
decode trace. At minimum, capture:

- cold and shared-prefix code prompts;
- tool-call and result messages;
- long-document prompts;
- failed and retried agent trajectories;
- response decode windows;
- multiple concurrent agents.

The first mask control is the published Coder allocation. Additional controls
are REAP-style trace masks and a uniform geometric mask. A candidate must win
held-out KL/perplexity and task-quality gates before a throughput claim matters.

## Prefiller coupling

Pruning and phase specialization are complementary but must not be fused into
one unmeasured artifact:

1. Run the vendor Coder decoder as-is and collect its routing/state statistics.
2. Generate a Strata-owned pruned decoder and pass exact pack inversion,
   routing, quality, and same-day runtime controls.
3. Train a dense/shared prompt-state prefiller against that exact frozen
   pruned decoder in shadow mode.
4. Add a separately owned hot-expert prefiller source only after dense state
   handoff passes. Re-run the state gates; a result from the dense rung does
   not transfer to the expert rung.
5. Re-train or revalidate the prefiller whenever the decoder mask, quantization,
   PLE input, tokenizer, or routing changes.

The 27B NVFP4 release remains the format/training template. It is not a
Flash-Next decoder, mask, or prefiller dependency.

## Decision gates

- **Vendor Coder promotion:** same-day matched full-model control, exact shard
  hashes, quality controls, and a measured footprint/throughput reason to keep
  using it.
- **Strata-owned mask:** representative prompt and decode traces, held-out
  quality, Coder and REAP controls, exact pack inversion, and deterministic
  generation from the pinned master.
- **Higher-bpw pruning:** resident-size accounting plus a quality gain over the
  3.5 bpw Coder-class control at equal or acceptable throughput.
- **Prefiller on a pruned decoder:** exact GDN/QSA/PLE state capture/restore,
  shadow-mode parity, default-off byte identity, and many-agent mixed-hit
  serving evidence.

[coder]: https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF
