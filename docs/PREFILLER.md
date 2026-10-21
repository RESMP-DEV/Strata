# Flash-Next phase-specialized prefiller

This is the architecture record for a future opt-in prefiller that is specific
to Qwen3.8-Flash-Next and its deployed low-bit decoder. The current implemented
piece is the host-only planner in `tools/prefiller_plan.py`; no training,
GGUF artifact, engine path, or runtime integration is claimed.

## Decision

The [ISTA-DASLab Qwen3.8-27B NVFP4 prefiller][odp-model] and
[*Disaggregated Quantization: Specializing LLM Prefill and Decode*][dq-paper]
are templates. They are not Strata model families or runtime dependencies.
The target is an auxiliary prompt-phase set of Flash-Next weights, trained
against the currently served frozen low-bit Flash-Next decoder.

The distinction matters because Flash-Next is a hybrid MoE: 48 trunk layers,
36 GDN layers, 12 QSA layers, 512 routed experts per layer, top-10 routing,
one shared expert, PLE at layer 2, and a native 262,144-token context. The 27B
reference is dense and has ordinary transformer KV state. A direct format or
block port would miss the central handoff.

MXFP4 is the Strata-side size format: 16 payload bytes plus one scale byte for
every 32 weights, or 4.25 bits per element. On B550's RTX 3090 Ti (sm_86), that
would be a W4A16/dequant path. It can recover capacity for quality work, but it
cannot inherit Blackwell native-FP4 speedup claims.

## Phase handoff

Prompt positions use the auxiliary weights to construct:

- GDN recurrent state;
- QSA K/V and indexer state;
- PLE history where the PLE layer participates;
- the ordinary layer outputs that feed later prompt positions.

At the prompt/decode boundary, those states are handed to the frozen decoder
unchanged. Response positions use the deployed low-bit decoder weights. Thus a
response-only loss still has gradients through prompt-generated state into the
prompt weights. This is the MoE-aware version of QADD's phase mask; it is not a
second complete model that merely produces conventional KV.

The teacher signal comes from the pinned Flash-Next BF16 master. The student's
response path is the exact frozen low-bit decoder that production serves. A
BF16 decoder used as the student response path, a generic NVFP4 decoder, or the
dense 27B decoder changes the target and is not accepted.

Initially share the token embedding, output head, norms and scalar parameters,
routers, and PLE key/value/n-gram table. Train or replace:

1. dense GDN projections;
2. dense QSA projections and indexer projections;
3. hyper-connection projections;
4. the shared expert;
5. selected hot routed experts.

Routing stays fixed at first. A selected routed expert uses its auxiliary
prompt weights; a cold expert falls back to the frozen decoder expert. This
keeps prompt and decode routing aligned and makes the hot set an explicit
manifest decision.

## Size boundary

The deployed B550 split-pack index and native-expert layout establish the
following candidate sizes. MXFP4 counts 17 bytes per 32-element group.

| Component | Elements | MXFP4 size |
| --- | ---: | ---: |
| 684 dense/shared state tensors | 3,571,384,320 | 1.767 GiB |
| 64 routed experts per layer | 15,099,494,400 | 7.471 GiB |
| 128 routed experts per layer | 30,198,988,800 | 14.941 GiB |
| 256 routed experts per layer | 60,397,977,600 | 29.883 GiB |
| all 512 routed experts per layer | 120,795,955,200 | 59.766 GiB |

A dense-only first artifact is small, but its quality reach is unknown. A full
expert artifact fits the B550's HPE P4800X 750 GB storage but should not be the
first training target. The 27B paper's SSD streaming result is dense-model
evidence, not evidence that streaming all 512 Flash-Next experts per layer is
efficient: only 10 of 512 experts are active per token, which changes the
compute-to-load ratio. Hot-expert coverage must come from real agent routing
traces.

The B550 device is an HPE P4800X TLC SSD. `/mnt/optane` is a legacy operational
mount name, not a media claim.

## Planner

`tools/prefiller_plan.py` reads only pack metadata:

```bash
python3 tools/prefiller_plan.py \
  --index /path/to/pack/index.txt \
  --native-experts /path/to/pack/native_experts.txt \
  --expert-id-map /path/to/pack/expert-id-map.json \
  --trace-domain physical \
  routing-trace.bin... \
  --hot-experts 64 \
  --out prefiller-plan.json
```

`--expert-id-map` is required for a pack with 256, 288, or 384 physical expert
slots. A complete 512-slot pack may omit it, in which case the only implicit
mapping is physical ID equal to source ID. The sidecar schema is
`strata-expert-id-map-v1`: exactly `schema`, `layers` (48),
`source_expert_count` (512), `physical_expert_count` (256, 288, 384, or 512),
and a per-layer `physical_to_source` array where
`physical_to_source[layer][physical_id]` is the source expert ID. Every layer
row has one slot per physical expert, every source ID is in `[0, 512)`, and
source IDs are unique within each layer. Duplicate JSON keys, unknown fields,
count mismatches against `native_experts.txt`, and non-integer IDs are
rejected.

`--profile data/expert-profile.bin` may be used without traces for a
deterministic rank-based plan, but coverage is emitted as null: the profile
stores order, not frequencies. Weighted traces provide an aggregate
router-weight share and hashes that bind the manifest to its evidence. Router
weights are summed as exact binary rationals and also emitted as a reduced
numerator/denominator pair. A planning trace must cover all 48 layers with ten
unique in-range experts per record; malformed, non-finite, negative, duplicate,
or partial traces are rejected. The planner refuses a source that identifies as
the 27B NVFP4 prefiller.

`--trace-domain physical` is the default because native pack traces carry
physical slot IDs; those IDs are checked against `[0, physical_expert_count)`
and translated through the sidecar. `--trace-domain source` accepts source IDs
in `[0, 512)`. Source-domain evidence may include experts pruned from the
physical pack: selection considers only mapped source IDs, while aggregate
coverage still includes the unmapped trace evidence in its denominator.

Trace files are auto-detected independently. The legacy engine format remains
`int32 layer`, `int32 k`, `k int32 expert IDs`, and `k float32 weights`, one
record per routed position. Batched prefill can instead use the repaired
`strp-batched-prefill-v2` format: a 40-byte little-endian header containing
`uint32` magic `0x50525453`, `uint16 version 2`, `uint16 phase 1`, `uint32
flags 0`, `uint64 layer_count 48`, `uint64 top_k 10`, `uint64
record_header_bytes 32`, and `uint32 reserved 0`; then repeated 32-byte
records containing `int32 layer`, `int32 k`, `int64 position`, `int32
token_count`, `int32 reserved`, and `uint64 request_id`. Each record is
followed token-major by `token_count*k int32` expert IDs and then
`token_count*k float32` true router weights. The planner expands
every chunk to per-position contributions; `trace_records` and
`trace_records_per_layer` therefore count routed positions, not stored chunks,
and remain comparable with legacy traces. The manifest names the detected wire
format in `expert_selection.trace_format`.

The batched reader rejects wrong magic, version, phase, flags, geometry,
header size, reserved fields, layer, top-k, position
range, token count, truncated records or payloads, non-finite or negative
weights, duplicate experts at one position, incomplete layers for any request,
unequal per-layer position counts, differing per-request layer position sets,
and overlapping positions within one
`(request_id, layer)` group. Distinct request IDs may validly reuse the same
layer and position ranges. The reader also refuses to combine legacy and STRP
files in one invocation. `--unweighted` discards the STRP weights and ranks
unit activations, while the default weighted mode consumes their true router
weights.

Legacy serve-mode `--dump-routing` traces use unit weights because the serve
verify path does not surface router weights to its host pool callback; the
opt-in literal STRP v2 serve capture below records true weights. The planner
names their
result `aggregate_routed_activation_coverage`, never a router-weight share, and
records how many trace records each layer contributed. The current durable
serve traces are decode/verify-window scoped and a forced-short-read
prompt-position diagnostic; the literal batched format below is now planner
input. Manifests now use
`strata-prefiller-plan-v2` with explicit `coverage_measure` and
`coverage_value` fields; version-1 manifests used
`aggregate_positive_weight_coverage`. Pruning prototypes and measured coverage
are in [PRUNING.md](PRUNING.md).

## Batched-prefill routing capture

`--dump-prefill-routing PATH` opt-in captures the literal multi-token
`Prefill::run` routing selected by the CUDA/HIP prompt path. The default flag
value is empty: `Prefill::on_routing` stays null and the prompt path performs
no routing download or stream synchronization for this feature.

When enabled, the engine writes one chunk-level record per MoE layer and
prompt chunk after routing (and after any configured research override), before
expert execution. IDs and F32 weights are the actual token-major `[T, k)`
tables copied from device memory; weights are not replaced by units. This is
routing capture, not timing instrumentation.

The file is little-endian and self-describing. Its 40-byte header is:

| Offset | Type | Value |
| ---: | --- | --- |
| 0 | `uint32` | magic `0x50525453`, bytes `STRP` |
| 4 | `uint16` | format version `2` |
| 6 | `uint16` | phase `1`, batched prefill |
| 8 | `uint32` | flags, currently `0` |
| 12 | `uint64` | layer count |
| 20 | `uint64` | top-k |
| 28 | `uint64` | record-header byte count, `32` |
| 36 | `uint32` | reserved, `0` |

Each repeated record starts with the following 32-byte header:

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `int32` | zero-based model layer |
| 4 | `int32` | top-k |
| 8 | `int64` | absolute position of the chunk's first token |
| 16 | `int32` | token count in this chunk |
| 20 | `int32` | reserved, always `0` |
| 24 | `uint64` | capture request ID |

The record header is followed by `token_count * top_k` little-endian `int32`
expert IDs and then the same number of little-endian F32 router weights, both
in token-major order. A one-shot trace uses request ID `0`. In serve mode the
first callback of each incoming request that actually invokes batched prefill
lazily allocates the next nonzero ordinal, so a fully reused request consumes
no ID and all pieces and layers of one request share that ID. These are
capture-request ordinals, not stable client or conversation identifiers. They
disambiguate repeated `(position, layer)` pairs across fresh turns of the same
conversation. Records can be serialized in nondeterministic order when
layer-split stages overlap; consumers must group by `(request_id, position,
layer)` rather than assume monotonic file order.

Strict refusals preserve interpretation. A one-shot run without a multi-token
batched prefill refuses the flag rather than emitting an empty file. A serve
run whose final `--short-read` threshold is greater than or equal to the final
auto-resolved and fit-adjusted prefill chunk refuses the flag because it can
bypass `Prefill::run`. While capture is active, a `STRATA_PIPELINE_SWITCH`
`short_read=` override is ignored and logged rather than allowed to change that
boundary; its other pipeline-testing fields remain operative. The serve `VRAM`
command is also refused while capture is active because re-lending can shrink
the final chunk after startup. Other serve
admission decisions can still leave a particular request uncaptured; zero body
records mean no prefill callback ran, never that routing selected no experts.
The existing `--dump-routing` path remains decode/verify-scoped and does not
consume or mix with this file.

Real B550 validation exercised the v2 seam on CUDA with the calibrated
Flash-Next pack. A one-shot 13,963-position run emitted 96 chunk records and
670,224 routed layer-positions with request ID 0; a two-request serve run
emitted request IDs 1 and 2 and 14,009 positions per layer per request; and a
split-30 run emitted one request across both cards while its log assigned
layers 0-29 to CUDA0 and 30-47 to CUDA1. All three traces had zero invalid
expert IDs, non-finite weights, or nonpositive weights. The compressed
one-shot raw trace and complete build/model hashes are retained in
[runtime evidence](../evidence/prefill-routing-v2-runtime-2026-10-21.json); the larger
serve and split raw files remain remote scratch evidence represented by their
SHA-256 digests. No timing, model-quality, trainer, or HIP-model claim follows
from these captures.

The manifest binds the decoder index and native-expert metadata by SHA-256.
It validates the declared index tensor count, exact dense role shapes,
48-layer native-expert layout, expert-ID sidecar, and profile length, but does
not validate model payload bytes and is not an executable artifact. The
manifest records both sidecar-file SHA-256 and a SHA-256 of the canonical
per-layer mapping array. Expert selections remain source IDs in `selected_ids`;
parallel `selected_physical_ids` and `*_physical_ids` fill fields identify the
deployed slots. Size accounting distinguishes selected physical experts, every
physical expert in the deployed pack, and the full 512-expert source universe.

A future executable manifest must additionally pin the native GGUF shards, PLE
input, MTP setting, tokenizer identity, control-vector setting, KV format,
context limit, chunk policy, and engine revision. A hash mismatch must fail
before any auxiliary state can enter a target session.

## Training ladder

1. **Data and routing first.** Capture representative decode/verify
   `--dump-routing` traces and literal `--dump-prefill-routing` STRP v2 traces
   from the planned many-agent workload. Separate shared-prefix/hit-heavy,
   cold-tail, code, document, and mixed sessions; reserve agents and prompts
   for held-out controls.
2. **One-block surrogate.** Implement phase-masked QADD on one GDN layer, one
   QSA layer, the PLE layer, and one MoE layer before scaling. Verify state
   checkpoint/restore, finite differences, and frozen-decoder parameter counts.
   Use the existing `conversation_state`, `conversation_kv_copy`, checkpoint,
   and snapshot contracts rather than flattening `SessionState`. Compare GDN
   recurrence and convolution layout, QSA page tables and indexer state, and
   row-fastest PLE history per layer; aggregate L2 distance alone is not a
   handoff gate.
3. **Exact state oracle before runtime.** Capture target residuals, routing,
   GDN, QSA K/V/indexer/page state, and PLE history; restore it into a scratch
   target; and reproduce teacher-forced continuation before an auxiliary is
   allowed near authoritative state.
4. **Adapters first.** Train low-rank or scale/code adapters for the dense and
   shared projections. Merge and quantize to MXFP4 only after the float/adapter
   control improves the target metrics and matches the deployment block layout.
5. **Hot experts only after dense shadow handoff passes.** Add the smallest
   selected expert set that covers a useful measured route-weight share. Train
   selected expert codes or adapters while cold experts remain byte-identical
   decoder reads, and repeat the state/handoff gates rather than transferring
   the dense rung's result.
6. **Full experts only by measurement.** Expanding to all 512 experts per layer
   requires held-out quality, failure-path, storage, and concurrent-serving
   evidence that the smaller sets cannot provide.

Quality gates must include teacher-forced response NLL/KL, exact-ID smoke on
fixed seeds, long-context behavior, and a frozen-decoder control. A candidate
that only improves tensor reconstruction or short-prompt perplexity is not
accepted.

## Runtime ladder

1. Add an opt-in `--prefiller <manifest>` path; absence of the flag must remain
   byte-identical to today.
2. Run the first dense/shared auxiliary in shadow mode, preferably in a
   separate process, and compare its state against the target prompt state. It
   must not mutate the authoritative `SessionState` until per-layer layout,
   failure-path, and fallback controls pass.
3. Promote the fixed-layout auxiliary into prompt execution while preserving
   decode weights and every graph captured for the decoder. A handoff or
   restore failure is fatal; inference must not continue from partially copied
   state.
4. Add a separately owned hot-expert source wrapper: selected prompt reads use
   the prefiller; all other reads use the current expert source. The auxiliary
   must not share mutable cache loans, residency tables, transient pointers, or
   the process-wide expert layout without an explicit ownership contract.
5. Stream auxiliary blocks from the P4800X through pinned host rings with
   explicit prefetch and failure accounting. Reuse existing `ple_reader`,
   `direct_file`, `PinnedRing`, and expert-source seams rather than adding a
   second storage subsystem.
6. Accept only under the target workload: many agents, mixed cache hits and
   misses, shared prefixes, and cold tails. A single-concurrency slowdown is
   expected while components are intentionally placed on different machines and
   is not by itself a rejection.

[odp-model]: https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-NVFP4-prefiller
[dq-paper]: https://arxiv.org/abs/2609.26333
