# Disaggregated KV-cache architecture

Status: target design. This document is the architecture contract for the
multi-tier KV work; it is not an implementation claim. The current
`--kv-disk` prototype remains experimental, is not wired into the serving
verify path, and must not be treated as accepted.

## Goal

Serve extremely long agentic coding sessions with a hot KV window in VRAM, a
bounded working set in pinned host RAM, and a cold tail on local Optane/NVMe
or a remote KV store. The target context envelope is 600K cells initially and
800K cells next. The MTP drafter ring and model experts remain outside this
tier.

The important property is not merely "KV can be placed on disk." The tier must
own placement, I/O, metadata, validation, and observability as one system.

## Non-goals

- Do not replace the hot VRAM window.
- Do not make a whole-session checkpoint act like a live cache.
- Do not require a particular KV numeric format; fp16, INT8, Q4_0, and K8V4
  are layout inputs.
- Do not hide a partial or unavailable cold tier behind zero-filled KV.
- Do not promote any implementation to a default before a matched live-model
  perplexity control passes.

## Layer model

```text
             Selection / append boundary
                        |
                KvTierCoordinator
        (one per session, stage, and batch slot)
                        |
      +-----------------+-----------------+------------------+
      |                 |                 |                  |
  L0 VRAM hot       L1 host cache     L2 local store     L3 shared store
  existing CLOCK    pinned, bounded   Optane/NVMe        optional Mooncake
  resident pages    working set       append-only groups replicated pool
      |                 |                 |                  |
      +-------- KvBlockStore ABI ------ KvBlockStore ------ KvBlockStore
``+

### L0: VRAM hot window

The existing `--kv-resident` page slots and CLOCK resolver remain the hot tier.
The disaggregated manager may ask the existing resolver to load a block, but it
must not duplicate the VRAM page table.

### L1: pinned host cache

The host tier is a bounded set of page slots, not a full-context authoritative
pool. Each slot has:

- logical block key;
- format and geometry epoch;
- resident/pending/invalid state;
- pin count for in-flight reads and the open tail block;
- host address valid for captured CUDA graphs;
- sequence number used by CLOCK or another explicit policy.

The host cache is deliberately sized by bytes and working-set behavior. It is
not assumed to be "realistic" at any particular fraction of the context.

### L2: local Optane/NVMe store

Local storage uses immutable, fixed-size groups. A group is the smallest
write, checksum, read, and validity unit. The payload is a pure byte object;
it does not need to understand QSA arithmetic.

The group size is derived from the selected KV format:

```text
blocks_per_group = 4096 / gcd(block_bytes, 4096)
```

For the current Qwen geometry:

| KV format | Bytes per 4-cell block | Blocks/group | Group bytes | 4 KiB pages |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 8,192 | 1 | 8,192 | 2 |
| INT8 | 4,224 | 32 | 135,168 | 33 |
| Q4_0 | 2,304 | 16 | 36,864 | 9 |
| K8V4 | 4,416 | 64 | 282,624 | 69 |

The format is therefore an input, not a reason to redesign the tier. A format
comparison must measure quality, cold bytes moved, and decode latency together.

### L3: shared/remote store

An optional Mooncake-compatible store is part of the same ABI, not a separate
engine path. Mooncake's useful roles are:

- a quota-bounded, pinned restore/client buffer;
- asynchronous batch transfer;
- object placement and replication across hosts;
- local SSD/NVMe or NVMe-oF backing;
- a shared pool once more than one inference host exists.

Strata still owns the QSA block key, page geometry, format fingerprint,
residency ordering, and validation. Mooncake treats KV bytes as opaque values,
which is the correct boundary for an adapter.

The dependency-free `KvMooncakePoolTransport` is the reference for the Master
side Strata can test before deploying a second host: one quota and one LRU
eviction order are shared across every store identity, replacement of an
existing object is refused, and an object larger than the pool cannot evict the
entire pool to admit itself. It does not provide Mooncake's replication,
placement, RDMA transfer, or standalone Master service; those remain the real
transport's responsibilities.

### What the imported engine already provides

The v0.1.45 engine already implements the L0 and L1 tiers for a streamed QSA
layer: the authoritative K/V lives in pinned, device-mapped host memory, VRAM
holds a bounded pool of page slots, and `kv_stream_resolve` makes the blocks
named by a selection resident with a device-side CLOCK sweep before the
attention runs. The coordinator does not replace that machinery. It owns:

- the L1 admission and eviction decision for the group granularity the store
  uses, above the page granularity the device map uses;
- the L2 and L3 store, the epoch rules, and the readiness contract;
- the counters that prove a cold read happened.

The device-side residency map remains the L0/L1 mechanism. A group is composed
of the pages the map already resolves, so the coordinator's group and the
engine's page must have one documented relationship, and the group's payload
is exactly the pages it contains in layout order.

## Block identity

A logical key names one immutable group:

```text
{session_id, geometry_id, format_id, layer_ordinal, group_index, epoch}
```

For local storage the key can map positionally to a fixed slot. For a shared
store it becomes the object key. `epoch` changes on rewind or conversation
reset; a new epoch never reinterprets old bytes.

The open tail block is not immutable. It remains pinned in L1 and is published
only when its final cell has been appended.

## Core objects

### `KvTierCoordinator`

One coordinator owns the KV residency graph for one session, stage, and batch
slot. It is the only component allowed to change a logical block's tier.

Responsibilities:

- size and own L1 slots;
- issue L2/L3 reads and writes;
- maintain readiness and pin state;
- update device-side host-slot tables;
- publish completed groups;
- invalidate on reset, rewind, or checkpoint restore;
- export counters and traces.

Layer-split and batch modes require one coordinator per stage/slot. Sharing a
coordinator across those modes is forbidden because the same layer ordinal can
refer to different states.

### `KvBlockStore`

```cpp
struct KvBlockKey {
    uint64_t session_id;
    uint64_t geometry_id;
    uint64_t format_id;
    int32_t layer;
    int32_t group;
    uint32_t epoch;
};

class KvBlockStore {
public:
    virtual bool contains(const KvBlockKey& key) const = 0;
    virtual bool get(const KvBlockRequest& request, std::string& error) = 0;
    virtual bool put(const KvBlockPut& request, std::string& error) = 0;
    virtual bool invalidate(const KvBlockKey& key, std::string& error) = 0;
};
```

`get` and `put` are asynchronous request objects, not synchronous file calls.
The initial implementation may complete synchronously behind the same ABI, but
the ABI must not expose that shortcut as architecture.

### `KvReadBatch`

A read batch is produced from the actual attention selection. It contains:

- logical blocks, deduplicated;
- target L1 slots;
- CUDA event/stream dependencies;
- priority and deadline;
- whether the request is foreground, prefetch, or migration.

The batch is the unit of I/O scheduling. It prevents one token from issuing
hundreds of independent tiny reads.

## Engine integration

The current prototype's central defect is that its hooks are in `qsa_layer`,
while the serving path writes and resolves QSA KV directly in `verify.cpp`.
The target integration is:

```text
verify.cpp append:
  kv_append_q8_steps / kv_append_q8_step
      -> append into the pinned host slot named by the coordinator
      -> coordinator publishes an immutable group

verify.cpp selection:
  qsa_block_scores + qsa_block_topk
      -> GPU produces a deduplicated block-read plan
      -> coordinator waits only for unavailable L1 blocks
      -> device host-slot table is ready

verify.cpp resolve:
  qsa_kv_resolve
      -> copies selected blocks from the named L1 slot into L0 VRAM
      -> attention proceeds only after every plan entry is ready
```

The same coordinator must be called by the token path (`qsa_layer`) and prompt
path. There cannot be one pager for generation and a second implicit policy
for serving.

### Graph-boundary rule

A CUDA graph cannot perform host-side file I/O or update a host cache from a
CPU thread. The captured sequence must therefore split at the selection
boundary:

1. selection graph/kernel produces the read plan;
2. CPU/I/O coordinator makes L1 slots ready;
3. resolve/attention graph consumes only ready slots.

The first correctness implementation may make this a synchronous split. The
target is asynchronous read batches with prefetch overlap.

## Write and eviction policy

1. Append writes the pinned open tail block in L1 and the existing VRAM slot
   when resident.
2. When a block completes, the coordinator marks it publishable.
3. A complete group is serialized, checksummed, and submitted as one store put.
4. The store's valid marker is committed only after the entire payload.
5. An L1 group becomes evictable only after its L2/L3 put is complete.
6. L0 remains governed by the existing CLOCK residency map.
7. A read failure or checksum mismatch is a miss/refusal, never a zero fill.

The tail is pinned. No incomplete block may be evicted or exposed to the
resolver.

## Read policy

- Foreground: blocks named by the current selection and absent from L1.
- Tail prefetch: the next expected append group.
- Selection prefetch: likely blocks inferred from the previous layer/token,
  but speculation can never replace the foreground plan.
- Migration: groups moving between local and shared stores.

Admission should begin with CLOCK, matching the existing VRAM policy. A more
placement-aware policy must beat CLOCK in a matched control before replacing
it.

## Integrity and metadata

Local file header:

- magic and format version;
- geometry and KV-format identity;
- block and group bytes;
- block count;
- payload offset;
- clean-shutdown marker;
- header checksum.

Group entry:

- valid bit;
- payload checksum;
- write epoch;
- completion identity.

The payload checksum covers all block runs in layout order. The header is the
single source of truth for its layout; readers must not independently
recompute a second layout. A group with an invalid marker or bad checksum is
absent, not corrupt garbage to return.

## Observability

Every accepted run must report:

- foreground reads, prefetch reads, and bytes;
- group writes and bytes;
- L1 evictions and pins;
- read latency p50/p95/p99;
- time the GPU spent waiting for cold reads;
- checksum refusals and store failures;
- distribution of distinct cold blocks per attention call;
- L0 hit rate separately from L1 hit rate.

A run with zero foreground cold reads cannot be presented as a cold-tier
measurement.

## Engineering quality gates

The first implementation iteration failed because it was built against the
wrong execution path. Every subsequent PR must include a path inventory before
code:

| Concern | Required live call sites |
| --- | --- |
| Serving append | `verify.cpp` `kv_append_q8_steps` / `kv_append_q8_step` (and format equivalents) |
| Serving selection | `verify.cpp` `qsa_block_scores` / `qsa_block_topk` |
| Serving resolve | `verify.cpp` `qsa_kv_resolve`, both row and batch forms |
| Token path | `layer.cpp` `qsa_layer` append and resolve |
| Prompt path | prefill append and staging entry points |
| Capture boundary | every graph that contains selection or resolve |

Additional PR rules:

- one coordinator implementation; no private second policy in verify or prefill;
- every public constructor/state transition has a test or an explicit non-claim;
- all supported formats run the same page movement test;
- no synchronous I/O call may be hidden inside a graph-capturable function;
- no new configuration default changes until live-model acceptance passes;
- telemetry is enabled in test builds and asserted by integration tests;
- failure injection covers I/O error, short read, checksum mismatch, invalid
  epoch, pinned allocation failure, and coordinator shutdown;
- code that changes paging behavior must cite the measured call path and the
  counters proving that path ran.

Review must reject a diff solely because its unit tests pass: the serving-path
integration and live counters are part of the definition of done.

## Acceptance gates

### Unit and kernel parity

1. Every supported KV format round-trips byte-exactly after forced eviction.
2. Partial final groups remain readable.
3. Torn headers and payloads are refused.
4. Random host-slot mappings produce identical resolver output to identity
   mappings.
5. Rewind/reset invalidates prior epochs.
6. One coordinator per stage/slot is exercised by a multi-state test.

### Live-model correctness

Matched RAM control and cold tier on the same model, artifact, context, prompt,
seed, expert cache state, MTP state, and numeric KV format:

- identical token IDs and targets;
- zero argmax mismatches over the scored trace;
- mean NLL delta within the pre-registered tolerance, initially `1e-4`;
- the cold arm must report nonzero foreground cold reads and evictions;
- report max absolute log-probability delta, not only perplexity.

Perplexity is a sanity check for "not garbage"; byte-exact page movement is
the stronger internal invariant.

### Scale and performance

- 600K context correctness run, then 800K;
- decode and prefill throughput against same-day full-host-RAM control;
- cold-read p50/p95/p99 and GPU wait time;
- host-cache byte budget sweep;
- KV format comparison only after correctness passes.

No speed or context-support claim is valid without those controls.

## Implementation phases

### Phase 0: establish the performance integration base

The coordinator must be built against the newer verify/prefill shape, not the
October 8 `fb58e0d` engine from the quarantined prototype.

1. Keep `upstream-v0.1.45-engine` as the imported runtime base and build the
   coordinator on `feature/kv-disaggregation-v0145`.
2. Import the coherent upstream `v0.1.45` runtime, tests, and stable server
   adapter, plus the sibling SM86 commit `2725c23f746e`; do not merge the
   October 19 0.2.0 app/package/release tree.
3. Preserve the local accelerator-plugin ABI and DeepSeek host contracts on
   top of that tree. The engine source must remain exactly the upstream SM86
   tree plus those owned additions until a ticket says otherwise.
4. Build CUDA sm86 on B550 and run the existing calibrated Qwen control before
   porting any coordinator hook.
5. Apply post-`v0.1.45` engine work only as ordered clusters with matched B550
   controls. The latest serving/app iterations are not performance inputs and
   do not justify a wholesale merge.

No coordinator integration is based on the old engine after this point.

### Upstream intake ledger, 2026-10-20

`v0.1.45` is `477ac13cbe8e`; the audited October 19 tip is
`e88ee39d1d6b`, dominated by 0.2.0 packaging and app work. `2725c23f746e` and
the I/O/CPU/prefill branches are siblings. A tree diff between siblings can
delete `nv_defaults`, so only ordered clusters may be imported.

| Cluster | Commits | Decision | Required B550 control |
| --- | --- | --- | --- |
| Batch/stage telemetry | `79d8ea066a6f` | Imported on the owned base as an opt-in measurement dependency; disabled unless `STRATA_BATCH_TELEMETRY=1` | Sampler unit test plus B550 batch and split-stage telemetry during the existing controls |
| Imported SM86 exact-kernel stack | `2725c23f746e` | Present in the base; upstream's 3060/A4000 numbers are provenance only | Same-day `STRATA_SM86_PERF=1/0`, token/logit identity, decode and prefill throughput |
| Host-aware prefill planner | `e77f1dfc3fa1`, `d8471050bde2`, `8253477bf9be` | First post-baseline candidate; portable and directly relevant to long context | Host-plan on/off, token identity, prefill tok/s, page-cache and drive counters |
| CPU pool planner | `5ece068f7e45`, `036c8f6ec3b5`, `bec4c838ab18` | Apply only after a B550 worker sweep proves benefit | automatic versus explicit worker counts, decode, RSS, worker warnings |
| Expert I/O mechanics | `c85cc6dd19d2`, `2044864dba32`, `8b1139d026c9` | Import only the opt-in mechanics after the planner; keep all defaults off | `STRATA_IO_STATS=1`, identity gate, cold/read-ahead and device-I/O counters |
| Linux expert-tier defaults | `82c80ac4d76a`, `c3136797` | Do not import yet; extract only cgroup clean-page accounting if needed | Forced expert-cold control plus KV foreground latency before any default change |
| Multi-GPU/concurrency | split overlap `56146c4c4906`, prefill pipe `125025a770c4`, concurrent reads `bc90e1acffde`/`34b726e09051`/`073d079d27d2` | Keep off until per-stage/per-slot coordinator keys exist and B550's serial queue policy is retested | Greedy-output identity, hang/watchdog proof, separate GPU0/GPU1 timing, and why `concurrent_read_off()` reports active or inactive |
| Gyro/Trellis SwiGLU guard | `8837cb6f` | Defer; current target is the calibrated Qwen IQ2_XS/Q2_0 artifact | Needed only when a Trellis format becomes executable |
| HIP/gfx12, SM120, packaging, release wiring | architecture-specific branches after `v0.1.45` | Reject for this B550 CUDA lane | No control applies until a different target host/model is selected |

Post-`v0.1.45` serving commits through the audited tip are protocol, packaging,
monitor, browser, picture-reader, and release-wiring changes. They contain no
B550 engine-performance claim. A selected correctness/UX fix may be ported
after the coordinator if its behavior is needed, but it is not part of Phase 0
performance intake.

The stable base is not merely the tag snapshot. Through `2725c23f` it already
contains the audited prefill MoE scatter, SM86 MMVQ/QSA kernel clusters,
conversation-cache prefix/park work, admission/slot-copy machinery, split
overlap, prefill pipe, and concurrent-read code. Presence is not promotion:
their runtime switches remain explicit or retain upstream defaults until the
named B550 controls run.

The planner's host model must eventually include L1 KV bytes and L2 foreground
read bandwidth. Expert prefetch and KV reads must share one deadline/priority
view; two independent bandwidth policies are a defect.

### Phase A: stabilize the boundary

- Quarantine the current one-file pager behind the `KvBlockStore` ABI.
- Move all verify append and resolve hooks to the coordinator.
- Keep synchronous reads, but make readiness explicit.
- Add resolver parity tests for all four KV formats.

### Phase B: correct serving path

- Split verify capture at selection/read-plan/resolve.
- Publish complete groups from the actual append path.
- Force cold misses in a small controlled test.
- Run the live-model NLL control.

### Phase C: asynchronous I/O

- Add read batches, double-buffered pinned slots, and background writes.
- Prefetch the open tail.
- Overlap store reads with independent GPU work.
- Add latency and GPU-wait metrics.

### Phase D: shared tier

- Add the Mooncake-compatible `KvBlockStore`.
- Keep a native local store for single-host operation.
- Use one key and byte-object mapping across both stores.
- Introduce a second host only after local correctness and scale gates pass.

### Phase E: placement optimization

- Sweep host-cache size and admission policy.
- Study per-layer and selection-aware placement.
- Consider local/shared replication for shared agent prefixes.
- Promote a policy only from matched measurements.

## Execution backlog

A ticket is not ready unless it names its path inventory, changed paths, and
counters. Upstream commit-message measurements are provenance, not acceptance.

| ID | Phase | Deliverable | Primary files | Acceptance |
| --- | --- | --- | --- | --- |
| U0 | 0 | Owned integration branch from the imported base | branch only | dirty prototype preserved elsewhere and ancestry receipt |
| U1 | 0 | `v0.1.45` runtime/tests/server import without 0.2.0 packaging | imported source tree | host quality plus exact-tree comparison receipt |
| U2 | 0 | Preserve accelerator-plugin ABI and DeepSeek host contracts | owned runtime additions | plugin/DeepSeek checks and CUDA link |
| U3 | 0 | Validate SM86 exact-kernel stack on real B550 | `nv_defaults.*`, `generate.cpp`, kernels | `STRATA_SM86_PERF=1/0` identity and same-day throughput control |
| U4 | 0 | Import and validate host-aware prefill planner | prefill plan/generate files | host-plan on/off long-prompt control with page-cache counters |
| U5 | 0 | Sweep and import CPU-pool planner only if beneficial | CPU pool/plan files | automatic versus explicit workers, decode, RSS, warnings |
| U6 | 0 | Import opt-in expert I/O mechanics; keep defaults unchanged | expert source/prefill/pool files | identity, I/O counters, forced expert-cold control |
| U7 | 0 | Build and run calibrated Qwen baseline before coordinator work | B550 runtime | exact configured control, source/model hashes, receipt |
| K1 | A | `KvBlockStore` ABI and local store adapter | new core KV files | all-format group-size and checksum tests |
| K2 | A | `KvTierCoordinator` state machine and epoch keys | new coordinator files | reset/rewind/failure and cross-stage collision tests |
| K3 | A | Move serving append boundary into coordinator | `verify.cpp` append sites | counters prove every append and no orphan producer |
| K4 | A | Move serving read-plan/resolve boundary into coordinator | `verify.cpp` selection/resolve sites | readiness assertion and explicit sync-read proof |
| K5 | A | Share token/prompt-path coordinator hooks | `layer.cpp`, prefill append sites | one policy and no duplicate residency state |
| K6 | A | Graph split proof | verify capture code | capture log contains selection/readiness and resolve phases |
| K7 | B | Real-source matched NLL control | B550 runtime | zero argmax mismatch, mean NLL delta at most `1e-4`, forced cold reads and L1 evictions nonzero |
| K8 | B | 600K scale gate | B550 runtime | correctness first, then throughput and cold-latency receipt |
| K9 | B | 800K scale gate | B550 runtime | correctness first, then throughput and cold-latency receipt |
| K10 | C | Async read batches and background group writes | coordinator/store | GPU-wait metric falls and no foreground write dependency |
| K11 | D | Mooncake-compatible opaque-object adapter | `KvBlockStore` implementation | round trip, quota, failure injection, and replication contract |
| K12 | E | Placement and bandwidth policy sweep | coordinator policy | beats CLOCK in matched control without starving expert reads |

## Known risks

| Risk | Guard |
| --- | --- |
| Selection touches many scattered cold blocks | Log distinct blocks/call; batch reads; measure p99 |
| Graph capture hides a host dependency | Selection/resolve split and readiness assertion |
| Append and resolve use different mappings | Shared coordinator and resolver parity test |
| Layer split or batch aliases state | Coordinator key includes stage/slot |
| NVMe latency dominates decode | GPU-wait metric and foreground/prefetch separation |
| Filesystem dirty state loses metadata | Clean unmount and `fsck.exfat -n` procedure |
| Smaller KV format hides a correctness bug | Format parameter included in parity and live controls |

## Implementation status

| Ticket | State | Evidence |
| --- | --- | --- |
| K1 `KvBlockStore` ABI and reference store | implemented, host tested | [phase A evidence](../evidence/kv-coordinator-phase-a.json) |
| K2 `KvTierCoordinator` state machine and epoch keys | implemented, host tested | [phase A evidence](../evidence/kv-coordinator-phase-a.json) |
| K3 serving append boundary | implemented and run against a model on B550 | [cold-tier evidence](../evidence/b550-optane-cold-kv-2026-10-20.json) |
| K4 serving read-plan and engine-host restore boundary | implemented and observed: 794,873 engine host restores | [cold-tier evidence](../evidence/b550-optane-cold-kv-2026-10-20.json) |
| L2 per-layer local file store | implemented, host tested | [implementation evidence](../evidence/kv-b550-implementation-2026-10-20.json) |
| K5 shared token and prompt path hooks | implemented, not yet run against a model | [phase A evidence](../evidence/kv-coordinator-phase-a.json) |
| K6 graph split proof | implemented and observed: 24,660 windows, zero split violations | [cold-tier evidence](../evidence/b550-optane-cold-kv-2026-10-20.json) |
| K9 read batch | implemented, host tested | [phase A evidence](../evidence/kv-coordinator-phase-a.json) |
| K10 Mooncake-compatible store and shared pool reference | implemented, host tested; real Mooncake transport not deployed | [implementation evidence](../evidence/kv-b550-implementation-2026-10-20.json) |
| K11 placement policy sweep | policy and matched control implemented, host tested; not swept on hardware | [phase A evidence](../evidence/kv-coordinator-phase-a.json) |
| Phase C explicit background operations | opt-in coordinator API and background publication implemented, host tested | [implementation evidence](../evidence/kv-b550-implementation-2026-10-20.json) |
| B550 acceptance harness | implemented and run on hardware: the matched cold control passes | [cold-tier evidence](../evidence/b550-optane-cold-kv-2026-10-20.json) |

The coordinator is now wired into the serving path behind an opt-in switch:
`--kv-tier <path>` turns the cold tier on and `--kv-tier-l1 <groups>` sizes
the L1 group cache per layer. It needs `--kv-resident`, because a cold tier
publishes the layer's host copy and only a streamed layer has one. With the
tier off, the serving path costs one branch per window and nothing else.

The hooks sit where the serial serving path really moves K/V: `verify.cpp`'s
commit boundary and its captured QSA selection/resolve phases. A per-layer
selection matrix preserves each QSA layer's plan, every owned layer gets its own
coordinator and fixed-layout file, and a serial generate/server request resets
the process-scoped epoch before rewriting its session. Layer split and batch
modes are refused until their stage/slot coordinators are wired.

The L2 store is a directory of one fixed-layout file per QSA layer. A group's
payload is durable before its valid marker is committed, the header is the only
source of truth for the layout, a group of another geometry, format, layer, or
epoch is absent rather than misread, and a partial read is refused because a
whole-group checksum cannot verify it.

The token path's publish happens at the commit boundary, not inside the
layer. The token path runs inside a captured graph, and a captured graph cannot
do host I/O or update a host cache from a CPU thread. The commit boundary is
where the window's writes have become host-visible, and the same coordinator
that serves the window publishes the tokens, so there is one policy and not a
second implicit path.

The selection's cells are collected into one read batch and deduplicated by
group, so a token that names hundreds of blocks issues one request per GROUP
rather than one per block. After a group is ready, its bytes are scattered back
into `QsaState::host`, the pinned host layout the captured `kv_stream_resolve`
actually reads; `engine_host_restores` counts that consumption boundary.

The graph-split proof counts the phases each captured window went through: a
captured selection, a host readiness phase, and a captured resolve, in that
order. A resolve with no preceding host readiness is a split violation, because
that means the graph tried to do host work. The counts, the batch statistics, and
the cold-read counters all appear in the tier's report line, so a run that
claims the cold tier is in the serving path can show it.

The L3 shared store keeps the semantic boundary in the right place. Strata owns
the key, the geometry, the format, the layer, the group, the epoch, the
checksum, and the quota; the transport carries opaque bytes under a key it does
not parse. The epoch is part of the object key, so a rewind's bytes are a
different object and cannot be served as current data. A real Mooncake client
implements the narrow transport interface; the in-process transport is the
reference for what that interface must provide, and the adapter's contract is
testable before a second host exists.

The placement policy ships with the matched control a promotion needs: the same
workload, the same cache size, and the same access pattern with only the policy
differing. CLOCK is the baseline, because the engine's own device residency map
uses it. A policy that does not beat CLOCK in that control is not promoted, and
a control that a policy wins against itself is a broken control, which the
contract test refuses.

`--kv-tier-async` opts one coordinator into a single serialized store worker.
Background publication snapshots immutable bytes, exposes explicit completion,
and drains before append, read, rewind, reset, or shutdown; cancellation and
failure never expose bytes as ready. The explicit foreground read API is host
tested, but the serving resolver still uses the synchronous `read` fallback, so
no overlap or latency improvement is claimed.

What is not yet true: the hooks have never run against a model. The CUDA
build, the identity gate, the matched NLL control, the forced cold reads, the
engine-host restore counter, and the 600K and 800K gates are all outstanding.
The engine still allocates its full pinned host copy, so this is a correctness
and consumption boundary, not a RAM saving; throughput and capacity benefits
are unmeasured.

## Current prototype status

The existing experimental pager has useful primitives:

- checksummed immutable groups;
- bounded L1 slot cache;
- partial-group handling;
- torn-payload refusal;
- forced-eviction round-trip tests.

It is not the target architecture and is not serving-path accepted. Its group
format and tests may be absorbed into Phase A; its integration points must be
replaced.

### The first hardware control

The matched cold control has run on B550 against the designated 750 GB device.
The tier off and the tier on, with the same engine, model, 8,192-token
real-source prompt and fixed seed, produced the same 16 output token ids, the
same 8,191 scored positions, a mean NLL delta of exactly 0.0 and a maximum
absolute log-probability delta of exactly 0.0. The cold arm reported 24,660
graph-split windows with zero split violations, 794,873 engine host restores,
756,560 L2 reads covering 102,262,702,080 bytes, 756,560 L1 evictions, and zero
read failures and zero checksum errors.

The tier was slower for that workload, not faster: about 14.8x on the prompt
read and about 29x on the decode. With 8 L1 groups per layer and 768 published
groups, nearly every group read missed the cache and came from the device, and
every one waited for the readiness phase. The L1 size is the knob that trades
device reads for host memory, and that trade has not been swept. The engine's
pinned host allocation is still not reduced, because the restore path
repopulates the copy the attention reads.

The gate also found the defect that only a hardware run exposes: the
coordinator published each group with an FNV-1a checksum while the file store
verified with its own CRC-32, so every group read back from a store was
refused. The tiers now share one checksum function.

Two operational facts belong here. The device is exFAT, so it cannot hold a
0700 directory: the mode is synthesized from the mount's `dmask`, and the
property that does hold is that other users cannot write the tier. And a cold
tier needs a context larger than the resident window, or the layer is fully in
VRAM and there is no host copy to publish. The receipt, the counters and the
non-claims are in
[the cold-tier evidence](../evidence/b550-optane-cold-kv-2026-10-20.json).
