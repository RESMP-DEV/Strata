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
| 64 routed experts per layer | 15,728,640,000 | 7.471 GiB |
| 128 routed experts per layer | 31,457,280,000 | 14.941 GiB |
| 256 routed experts per layer | 62,914,560,000 | 29.883 GiB |
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
  routing-trace.bin... \
  --hot-experts 64 \
  --out prefiller-plan.json
```

`--profile data/expert-profile.bin` may be used without traces for a
deterministic rank-based plan, but coverage is emitted as null: the profile
stores order, not frequencies. Weighted traces provide an aggregate
router-weight share and hashes that bind the manifest to its evidence. Router
weights are summed as exact binary rationals and also emitted as a reduced
numerator/denominator pair. A planning trace must cover all 48 layers with ten
unique in-range experts per record; malformed, non-finite, negative, duplicate,
or partial traces are rejected. The planner refuses a source that identifies as
the 27B NVFP4 prefiller.

The manifest binds the decoder index and native-expert metadata by SHA-256.
It validates the declared index tensor count, exact dense role shapes,
48-layer native-expert layout, and profile length, but does not validate model
payload bytes and is not an executable artifact.

A future executable manifest must additionally pin the native GGUF shards, PLE
input, MTP setting, tokenizer identity, control-vector setting, KV format,
context limit, chunk policy, and engine revision. A hash mismatch must fail
before any auxiliary state can enter a target session.

## Training ladder

1. **Data and routing first.** Capture representative `--dump-routing` traces
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
