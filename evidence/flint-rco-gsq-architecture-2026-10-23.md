# Flint RCO/GSQ architecture and first implementation

Date: 2026-10-23
Agent: Codex

## Scope

Architect expert pruning and alternative quantization for the owned Flint
40B-A2.5B checkpoint, with a primary 25% routed-expert prune lane, a separate
GSQ lane, and an assessment of the vLLM Metal serving route. Three read-only
audits covered RCO/GSQ reuse, MLX recalibration/packing, and vLLM Metal
integration. The parent implemented the first host-safe slice.

## Baseline identity

- Strata base: `f384039268190508f161e3191f0ac8e935469318`
- RCO: `9a1e09c07d468109cbe60a1b87d5036034a79d10`
- GSQ: `03fc16484c369e3127225615d5e03e8d3a6043e3`
- MLX stack at audit start: `d4ad5c0e8fb69e3bdd5b177fb0f3bbc2e1bbbbd4`
- MLX-LM: `b384a406ad5cac168877c32b048f31aa9c8db244`
- Flint BF16 source: `RecursiveSI/flint` revision
  `2a247be0c0aec895ccb0ac7ca92229ec6df37288`

## Geometry and exact primary budget

The pinned BF16 index has 36,734 tensors:

- 47 routed-expert layers and two dense MLP layers;
- 47 router tensors;
- 47 times 141 shared-expert tensors;
- 36,096 routed-expert tensors (`47 × 256 × 3`);
- 450 other language-model tensors.

The primary 25% lane retains 192 and prunes 64 experts in every routed layer:

- retained expert slots: `47 × 192 = 9,024`;
- pruned expert slots: `47 × 64 = 3,008`;
- retained expert tensors: `27,072`;
- omitted expert tensors: `9,024`;
- BF16 routed-expert saving: 18,924,699,648 bytes;
- MXFP4 routed-expert saving: 5,026,873,344 bytes.

This does not change the six active routed experts per token or the shared
expert. It changes the physical expert universe, footprint, routing candidates,
and likely batched expert locality; it does not automatically improve quality
or arithmetic throughput.

## Implemented first slice

### Exact metadata planner

`tools/flint_compression_plan.py` validates the real config and index against
the exact 36,734-tensor inventory. It emits one primary 192-expert lane plus
224/192/160 sensitivity lanes and records separate RCO and GSQ method order.

Real-source result:

- Config SHA-256:
  `a57c33d11f725099105b657043c113149763066d204fd472d0b2ee399abc2ba7`
- Index SHA-256:
  `e5e9c492685f84c8d2d8369f36c74674b03410715f23f6179313f6f1feb90311`
- Ignored report: `build/reports/flint/compression-plan.json`

### Stock-RCO topology adapter

`tools/flint_rco_adapter.py` maps logical RCO MoE layers to physical Flint
layers 2..48, skipping the two dense layers. It preserves sigmoid router
scores, top-6 selection, STE non-renormalization, deployed hard-mask survivor
renormalization, and the additive shared expert.

`tools/validate_flint_rco.py` drives a tiny model through stock RCO's reference
cache, Gumbel budget assignment, STE mask, full-model KL, and backward pass.
The live CPU probe produced:

- target and actual pruned experts: 3;
- nonzero alpha gradient: true;
- hard-mask KL: 0.0547661845;
- physical keep rows for every logical layer.

This is a tiny topology/gradient integration, not a full stock optimizer run.
Stock RCO's optimizer hardcodes CUDA for search state, so the full source-model
search must run on the B550 or another owned CUDA host.

### Streaming materializer

`tools/materialize_pruned_flint.py` validates a
`strata-flint-expert-mask-v1` document, slices router rows in keep order,
renumbers retained experts to target IDs 0..191, omits dropped expert tensors,
copies shared/dense tensors, writes a fresh index/config, and emits a
physical-to-source expert map and receipt.

A synthetic 192-per-layer dry run against the real BF16 metadata selected
27,710 tensors and omitted 9,024. It did not open a tensor payload or create
the approximately 56.8 GB routed-expert output.

### B550 live router-control pass

The branch was staged on B550 via a local owned Git remote. The real 75 GB
BF16 source has config SHA-256
`a57c33d11f725099105b657043c113149763066d204fd472d0b2ee399abc2ba7` and index
SHA-256
`e5e9c492685f84c8d2d8369f36c74674b03410715f23f6179313f6f1feb90311`.
The machine exposes only GPU0; GPU1 is `vfio-pci` and remains untouched.

The adapter and tiny stock-RCO integration passed on B550 in the Torch
2.14 environment. A new streaming collector then performed the first
full-source 16×256 real-token pass in 193.787 seconds using one resident
layer at a time on GPU0. Its deterministic 192-expert mask has:

- mean retained router mass: 0.843538642;
- mean retained top-6 frequency: 0.803393245;
- mask SHA-256: `52e0d859c19b739186364658e157c1ff70eca8a63a65f01f22f700c4bb381ea2`.

A 128×1024 pass was then launched as a stronger control before any physical
materialization or RCO STE search.

## Architecture

### RCO lane

1. Use the full 256-expert BF16 source and the corrected NeoX half-split
   calibration skeleton committed in MLX stack at
   `3f4b972068d3b7eb591a19ab02ba5e8d9627f110`.
2. Build RCO-compatible 256×2048 calibration tokens from the pinned corpus.
3. Run stock RCO with the Flint adapter, `--sparsity 0.25`, and
   `--per-layer-budget`, yielding 64 pruned experts per routed layer.
4. Compare the selected mask against router-score/frequency controls and retain
   the full-model KL receipt.
5. Convert the prune mask to the self-describing Flint keep-ID document.
6. Materialize the 192-expert BF16 checkpoint.
7. Load-check it with the eager Flint skeleton.
8. Recalibrate all 49 layers with GPTQ-MXFP4 from pinned token windows.
9. Validate every artifact, pack to MLX MXFP4, and carry the mask/expert-map
   sidecars into the final artifact.
10. Run quality and performance gates before any publication claim.

The ordering is deliberate: physically materialize 192 experts before Stage A.
Do not pair a 256-expert converted MLX model with artifacts for only 192
experts. The existing pack failure for partial expert artifacts is correct.

### GSQ lane

GSQ is a separate replacement optimizer, not a stage after RCO. It would
replace GPTQ-MXFP4 Stage A on an already materialized mask. Current GSQ lacks a
Flint wrapper, sigmoid routing support, separate expert projections, an
E2M1/E8M0 group-32 MXFP4 quantizer, and an MLX exporter. The proven MLX stack
GPTQ-MXFP4 path is the primary lane.

### vLLM Metal lane

Stable Homebrew 0.32 is not the right base: it has MLX 0.32.2, Transformers
5.19, no Flint MLX-LM module, and fails Flint config recognition. Current
source/dev builds are closer and pin MLX 0.32.4. Existing source supports paged
full attention, per-layer sliding windows, GQA, and learned sinks.

The audited current source revision is
`f0812a6b679ce6afbe4cb5120d53be35ad625cc9`, with dev release
`0.32.0.dev20261023120028`. The exact minimum compatibility surface is:
`FlintCompatConfig`, a metadata-only `FlintForCausalLM` architecture alias
restricted to the MLX backend, a dynamic attention contract that treats
`module.rope is None` as position-free, and a runtime-only
`softmax_offset` fallback where the paged wrapper currently reads `sinks`.
Flint is absent from the compiled-MLP policy, so MoE compilation is a separate
post-correctness optimization rather than part of minimum integration.

The durable adapter needs guarded config registration, a metadata-only vLLM
architecture alias, the owned MLX-LM fork dependency, per-layer no-RoPE
contracts, and a `softmax_offset` to `sinks` alias, followed by hybrid
cache-plan, MXFP4 MoE parity, load, and concurrent benchmark gates. No patch
was applied to the fetch-only upstream checkout.

## Required controls and acceptance

Quality arms:

1. full 256-expert BF16;
2. masked 192-expert BF16;
3. recalibrated full 256-expert MXFP4 after the RoPE repair;
4. recalibrated masked 192-expert MXFP4.

The existing published MXFP4 artifact was calibrated before the RoPE repair and
must not be the only control for the mask effect.

Acceptance requires an exact `(47, 256)` mask, deterministic hashes, all-ones
adapter parity, zero-mask shared-only behavior, reconciled materialized tensor
counts, `[192, 2048]` routers, complete artifacts with zero validation
failures, full MLX load, held-out PPL/top-1 and paired bootstrap,
workload-specific quality, and same-day throughput controls.

## Validation already run

```text
tools/test_flint_compression_plan.py: 4 passed
tools/test_flint_rco_adapter.py: 7 passed
tools/test_materialize_pruned_flint.py: 2 passed
tools/test_validate_flint_rco.py: 1 passed
```

Pinned Torch environment:

```text
tools/test_flint_rco_adapter.py: 7 passed
tools/test_validate_flint_rco.py: 1 passed
tools/validate_flint_rco.py: PASS
```

Focused Ruff and format checks passed for every new tool and test.

## Non-claims

- No RCO mask was selected on the released 40B source.
- No approximately 56.8 GB pruned BF16 payload was written.
- No Stage A calibration, GSQ optimization, MLX pack, quality evaluation, or
  vLLM Metal server run was performed.
- The synthetic mask used for the materializer dry run is not a quality mask.
- The expected footprint saving is not a throughput prediction.
- vLLM Metal kernel capability does not establish Flint integration readiness.
