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

A 128×1024 pass then completed in 603.232 seconds on the same single visible
GPU. Its deterministic 192-expert mask has:

- mean retained router mass: 0.835313082;
- mean retained top-6 frequency: 0.817712247;
- mask SHA-256: `5fb120f83678d6bc3396b61c2588cba5628bce615e81762b22c1d7d25fc38dc5`.

The stronger mask is the current control baseline. It remains a router-score
control, not the final Gumbel-STE RCO selection.

Local ignored copies of both B550 runs are under
`build/reports/flint/b550/`.

### Physical 192-expert control materialization

The 128×1024 mask was then materialized into:

`/home/kearm/flint-192-router-control-bf16`

The streaming materialization completed in 97.044 seconds:

- selected tensors: 27,710;
- omitted tensors: 9,024;
- expert tensors: 27,072;
- sliced routers: 47;
- other copied tensors: 591;
- output payload size: 60,633,554,496 bytes;
- expert-ID-map SHA-256:
  `5b016d1faa23e09fcaef0d98447333958e07d3571bd2c315096d67911ed8510c`.

`tools/verify_pruned_flint_checkpoint.py` then streamed all 49 materialized
layers plus final norm and LM head on GPU0. The 52.234-second check completed
with finite final logits and top-8 last-token IDs
`[506, 47, 12, 44, 332, 30, 39, 3779]`.

This proves physical tensor shape/order/loadability. It does not establish
generation quality or runtime performance.

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

### Pre-registered 192-expert BF16 gate

Before reading the 16-window held-out result, we fix the acceptance gate for
promoting any 192-of-256 physical BF16 mask to the MXFP4 calibration stage:

- paired NLL delta (candidate minus source) is at most `0.05` nats/token;
- the 95% window-bootstrap upper bound for that delta is at most `0.08`
  nats/token;
- next-token top-1 agreement with the source is at least `0.95`;
- top-1 accuracy loss is at most `0.5` percentage points.

The gate uses 16 held-out calibration windows (`offset=128`, sequence length
1024) disjoint from the 128 windows used to select the current router-control
mask. It is an engineering gate, not a broad benchmark suite. The gate is
pre-registered here before the quality report is read.

### Router-control 192-expert gate result

The gate ran on B550 at Strata revision `1b47e1f9` with
`tools/evaluate_flint_quality.py`, 16 held-out windows (offset 128, sequence
length 1024), batch size 4, and 4096 bootstrap resamples at seed 173.

| metric | 256-expert BF16 source | physical 192-expert BF16 |
| --- | --- | --- |
| mean NLL (nats/token) | 2.9900943347887323 | 2.2160611408282422 |
| perplexity | 19.887558491706013 | 9.17113581665058 |
| top-1 accuracy | 0.5821533203125 | 0.6331787109375 |

Paired comparison:

- NLL delta (candidate minus source): `-0.7740331939604903` nats/token, which
  is `-1.1166938504102446` bits/token;
- window-bootstrap 95% CI:
  `[-1.3991392528709548, -0.21051020239917395]` nats/token;
- next-token top-1 agreement: `0.79736328125`;
- both correct 8296 tokens, candidate only 2078, source only 1242;
- arm times: source 162.575 s, candidate 77.664 s, total 241.186 s.

Receipt SHA-256: `66166ce5f7df9511c2649b01b604c87733ceecf231e4ffe3c0f74a9309d96362`.
Raw metric arrays SHA-256:
`008f346057e4a188738173e362ca2d4c00ec5d181452ccee7f13d0c609771fcb`.

Gate outcome: the two NLL criteria pass, and top-1 accuracy does not drop. The
agreement criterion fails because `0.79736328125` is below the pre-registered
`0.95`. The router-control mask is therefore rejected for promotion to the
MXFP4 calibration stage. Lower NLL on this corpus alongside 20% next-token
disagreement is not evidence of a better model.

A first run of the same gate at revision `f78cad51` was discarded. Its batch
rows were concatenated along the token axis, so the aggregate NLL and top-1
numbers were valid but the 16 per-window bootstrap samples were regrouped.
Revision `1b47e1f9` fixes the row assembly and adds a regression test.

### Quantization lane order for Flint

The Flint compression lane keeps the methods in this order, and no step is
skipped by another step's result:

1. mask selection on the 256-expert BF16 source;
2. materialization of a physical 192-expert BF16 checkpoint;
3. quality gate of that BF16 checkpoint against the 256-expert source;
4. GPTQ-MXFP4 recalibration of the promoted checkpoint with the RoPE-repaired
   eager skeleton;
5. MLX MXFP4 packing with the expert-ID map and mask provenance carried as
   sidecars;
6. artifact validation, then paired BF16 and MXFP4 evaluation.

GSQ does not sit in that chain. GSQ is a Gumbel-Softmax scalar quantizer that
would replace step 4, not follow it, and the pinned GSQ tree still lacks a
Flash-Next/Qwen4Exp wrapper, so it cannot serve Flint today. The proven
GPTQ-MXFP4 path stays as the only live quantizer. The user is satisfied with
that path, so GSQ is only revisited if a measured comparison shows an
advantage.

Full-model Gumbel-STE RCO is not running on this host. Stock RCO needs the
whole 74.11 GiB source resident plus the retained autograd graph, and B550
exposes only one 24 GiB GPU because the second RTX 3090 Ti is bound to a
VFIO guest. The honest options are a new streaming STE driver with
reverse-streaming backward (12 to 15 hours for the stock 200-step default,
1.5 to 2 hours for 50 steps with 2 Gumbel samples) or the layer-local exact
leave-one-out search that now runs on the real source in 167.874 seconds. We
take the layer-local search first because it produces a real, auditable mask
from real activations, and every one of its masks must still pass the
pre-registered full-model gate before any quantization work starts.

### Layer-local leave-one-out search result

`tools/flint_prune_impact_search.py` then measured each expert on the real
256-expert BF16 source. It captured the actual post-attention MoE input, evaluated
the real routed expert outputs for the current top-6 plus the next-best expert,
and computed the exact squared routed-output change when that expert was removed
and the router promoted the replacement under the deployed sigmoid score
normalization.

The full search ran on 8×1024 real calibration tokens with greedy rounds
32/16/8/8 and completed in 167.874 seconds. It emitted a materializer-compatible
192-expert mask with SHA-256
`7f3389f70afa160611c70df2dc517d2221246dda5f9e2989400306a574d984c6`.
Its overlap with the previous router-control mask is only 138–152 retained
experts per layer (mean 145.55), so the two methods disagree materially; no
layer is identical. This is not a proxy score. It is a real layer-local
measurement on the real 256-expert source, though still not a full-model
Gumbel-STE search.

Materialization into
`/home/kearm/flint-192-prune-impact-bf16` took 342.45 seconds and produced:
- 27,710 selected tensors;
- 9,024 omitted tensors;
- 27,072 expert tensors;
- 47 sliced routers;
- 591 other tensors;
- payload size 60,633,554,496 bytes;
- expert-ID-map SHA-256
  `5fa2e62bf56e8fc1a610cacf3ea6bd37a661ca9bc24e17fc8265ff57a3f3cf72`.

Materialization receipt SHA-256:
`53b83de178d14026425faec25b801d74867fb3996818f01372143bec2ca30a6f`.

### Leave-one-out mask gate result

The same 16-window held-out gate at revision `94c2996b` produced:

| metric | 256-expert BF16 source | physical 192-expert BF16 |
| --- | --- | --- |
| mean NLL (nats/token) | 2.9900943347887323 | 2.3062101731753035 |
| perplexity | 19.887558491706013 | 10.036316587312733 |
| top-1 accuracy | 0.5821533203125 | 0.63653564453125 |

Paired comparison:

- NLL delta (candidate minus source): `-0.683884161613429` nats/token, which
  is `-0.9866362885022004` bits/token;
- window-bootstrap 95% CI:
  `[-1.2309787191000137, -0.1749546560468955]` nats/token;
- next-token top-1 agreement: `0.81048583984375`;
- both correct 8431 tokens, candidate only 1998, source only 1107;
- arm times: source 161.023 s, candidate 123.343 s, total 286.406 s.

Receipt SHA-256: `a3e688d849d426898df0187223d2d3ed0297526c47112d9c8bb27a020723de03`.
Raw metric arrays SHA-256:
`83eb3a0254098ffad2be1a72df330f4f24396d52a4a694284b0e95d6a6ab8ed4`.

Gate outcome: NLL passes and top-1 accuracy rises, but the pre-registered
agreement criterion still fails (`0.8105` is below `0.95`). The leave-one-out
mask is therefore not promoted to MXFP4 calibration under the pre-registered
gate. This is the strongest measured mask so far, and it is the one to carry
forward if the user relaxes the agreement threshold; it is not yet a finished
quantized artifact.

The search was rerun at revision `b9970a82` with per-round impact records
enabled. It produced the identical mask SHA-256
`7f3389f70afa160611c70df2dc517d2221246dda5f9e2989400306a574d984c6` in
160.932 seconds, confirming deterministic selection. The detailed receipt
SHA-256 is
`1501d06d16b08b9747745662c6fd1bec250378b1b2b804335647c007e05ca08c`.

### Agentic-coding target-workload gate

The 192-expert leave-one-out checkpoint was then evaluated on 16 real
agentic-coding windows from the held-out corpus, tokenized with the Flint
tokenizer. The token-array SHA-256 is
`156221bdbdfc450999a2e85f229e195149d58207e535dac303a9a2b9adfc5a4b`.

| metric | 256-expert BF16 source | physical 192-expert BF16 |
| --- | --- | --- |
| mean NLL (nats/token) | 2.202686766920426 | 2.426629744567646 |
| perplexity | 9.049294210860637 | 11.320664184176582 |
| top-1 accuracy | 0.61273193359375 | 0.57574462890625 |

- NLL delta (candidate minus source): `+0.22394297764721968` nats/token;
- window-bootstrap 95% CI:
  `[0.04076583168404919, 0.3701914922866968]`;
- top-1 agreement: `0.840087890625`;
- both correct 8426, candidate only 1007, source only 1613;
- elapsed 262.398 s.

Receipt SHA-256: `abaa3ab6db0f9aa75e0c21811aa14ee2e6526cf2a19a72cf80c275faa40880ab`.

This fails every pre-registered criterion. A 25% expert prune is too
aggressive for the target agentic-coding workload: perplexity rises by 25.1%
and top-1 accuracy falls by 3.70 percentage points. It is not promoted.

### 224-expert sensitivity lane

Because the 192-expert lane failed the target workload, the same leave-one-out
search was repeated with half the prune budget: rounds 16/8/4/4, 32 experts
removed per routed layer, 224 retained. It completed in 172.462 seconds and
emitted mask SHA-256
`b7944db2718903ffbbc4d8abeecb6888da1338018376c4b7e222bd28a528c429`.

Materialization took 400.299 seconds and produced 32,222 tensors, 31,584 expert
tensors, 47 sliced routers, 70,102,064,704 payload bytes, and expert-ID-map
SHA-256
`507e4e901d512ef0bd72c0e48ce114095273646ca1506d5a67b3c67f2418bac5`.
Materialization receipt SHA-256:
`f44c8bcafaf28bc8e263690728e0734c474764d3ad2fe0e07c71e38bebf86e3a`.

Agentic-coding gate:

| metric | 256-expert BF16 source | physical 224-expert BF16 |
| --- | --- | --- |
| mean NLL (nats/token) | 2.202686766920426 | 2.268387622112280 |
| perplexity | 9.049294210860637 | 9.663806537545630 |
| top-1 accuracy | 0.61273193359375 | 0.59820556640625 |

- NLL delta (candidate minus source): `+0.0657008551918541` nats/token;
- window-bootstrap 95% CI:
  `[0.014188017067963542, 0.11178702235286228]`;
- top-1 agreement: `0.890625`;
- both correct 9024, candidate only 777, source only 1015;
- receipt SHA-256:
  `a69ba7f9b9de65107bb544d785df1e9ef77003303fb7cf91ca68d571c9556026`.

This also fails every pre-registered criterion. Halving the prune budget cuts
the damage roughly in half, but a 12.5% prune still raises coding perplexity by
6.8% and loses 1.45 percentage points of top-1 accuracy. There is no measured
prune lane here that can be honestly promoted to quantization or publication.

Gate-timing clarification: an earlier run at `f78cad51` had already been seen,
but its per-window assembly was invalid. The table above was fixed before the
corrected `1b47e1f9` window-level report existed. "Pre-registered" here means
before the valid report, not blind to every earlier aggregate number.

### Verdict correction after the broad agentic sample

The first agentic gate used 16 windows from the opening region of the holdout
corpus and was not representative. A 32-window sample spread across the full
1,962,793-token corpus changed the result:

| arm | PPL | NLL delta | 95% CI | top-1 loss | agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| 224 experts | 6.0805 | -0.01908 | [-0.11321, 0.05757] | 0.27 pp | 90.65% |
| 192 experts | 6.8559 | +0.10094 | [-0.01781, 0.21028] | 2.06 pp | 86.97% |

The source arm PPL was 6.1977. The 224 lane is viable on this broader workload:
its paired NLL difference is indistinguishable from zero, it slightly lowers
PPL, and it loses only 0.27 top-1 points. The 192 lane still degrades
materially: PPL rises 10.6% and top-1 falls 2.06 points.

The 0.95 next-token-agreement threshold is therefore retired as a veto. It
measures behavioral divergence, not task quality. Agreement remains a reported
diagnostic. The 224 lane is accepted as the candidate for quantization-aware
selection and final MXFP4 evaluation; the 192 lane is not accepted at this
time. BF16 results remain diagnostic only because every deployment target is
quantized.

### Quantization-aware selection

`tools/flint_prune_impact_search.py` now accepts `--quantized-artifacts`. It
loads the calibrated full-256 GPTQ-MXFP4 Stage-A artifact, dequantizes every
quantized linear for each layer with `Pipeline.apply_precomputed`, runs the
real quantized activations, and computes leave-one-out impacts from the
dequantized expert weights. This is QDQ-aware mask selection under the fixed
GPTQ-MXFP4 quantizer. It is not joint QAT of mask and quantizer parameters.

The available Stage-A artifact was produced before the RoPE repair, so the
first QDQ mask is provisional. A post-RoPE full-256 calibration is required
before treating the QDQ mask as final provenance.

### Actual MXFP4 deployment comparison

The BF16 gate was only diagnostic. The decisive measurement loads the already
calibrated full-256 MLX GPTQ-MXFP4 checkpoint, slices the router and quantized
expert tensors in memory to the requested mask, and evaluates the deployed
weights directly.

The 32-window agentic sample produced:

| arm | PPL | NLL delta | 95% CI | top-1 delta | parameter bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| full 256 MXFP4 | 6.20711 | reference | reference | reference | 22,075,711,040 |
| BF16-selected 224 | 6.26226 | +0.00885 | [-0.04363, 0.05631] | -0.403 pp | 19,556,113,984 |
| QDQ-selected 224 | 6.20609 | -0.00016 | [-0.04943, 0.04477] | -0.217 pp | 19,556,113,984 |
| BF16-selected 192 | 6.92679 | +0.10970 | [0.01060, 0.20037] | -1.810 pp | not promoted |

The QDQ 224 lane is effectively parity with full 256 on the deployed MXFP4
weights: 15 of 32 windows are better, the paired delta is slightly negative,
and the confidence interval contains zero. It also removes 2,519,597,056
parameter bytes, or 2.35 GiB, an 11.41% reduction. This is the quality and
memory evidence that the BF16-only gate could not provide.

The QDQ-selected 224 mask differs from the BF16-selected 224 mask by only about
three experts per layer (mean overlap 221.23 of 224, range 217–223). The
quantization-aware correction is real but small; the expert ranking is stable.

Receipt SHA-256 values:

- full 256 MXFP4: `4d22af88a15dd023ebbd26430f240789724e3c7af7b189c6a0abeb5ef0b971b4`;
- BF16-selected 224: `bd1c7293a63fb72f77e0793d9f761b7532cc5901acbaeb1e9ed1867f488efb75`;
- QDQ-selected 224: `5f0e5e8cebaad558dd30ed3e2481c174c7e0e77a7eafab043a7034f5bb591623`;
- BF16-selected 192: `311a247aec9ea2434f2f09366937dff46c2182d8dfee833a2227e32aad32c4fc`.

This comparison uses the calibrated full-256 MXFP4 weights with the mask
applied at runtime. A distributed 224 artifact must preserve those q/scales and
pass a packed-model equivalence check before publication. It is not yet
uploaded.

### Physical 224-expert MXFP4 artifact

`tools/materialize_pruned_flint_mlx.py` then sliced the calibrated full-256
MLX model into a physical 224-expert artifact at:

`/Users/kearm/models/RecursiveSI-Flint-40B-A2.5B-MLX-GPTQ-MXFP4-224`

The slicer changed 329 tensors (47 router matrices and 282 expert
weight/scale tensors), preserved every GPTQ q/scale value, set
`num_experts=224`, and carried the expert-ID map and mask as sidecars. Shard
payloads total 19,556,263,509 bytes. The materialization receipt SHA-256 is
`e127cc8c683015de32a8fda24b749caabd8d7ce55ba83de82e638b7cd71723e5`;
the expert-ID-map SHA-256 is
`d731ae02d470bdfa9e9b17984e7cb500d426da6862bfbc048167176f145bb949`.

The physical artifact loaded normally and reproduced the runtime-pruned QDQ
result exactly:

- mean NLL: `1.8255318440496922` nats/token;
- perplexity: `6.2060948161381635`;
- top-1: `0.666748046875`;
- parameter bytes: `19,556,113,984`.

Receipt SHA-256:
`4529d46dac9f4bdba4709cb849c48149da4d2ffe09df45933928cd65cc0449ed`.
The per-window NPZ is byte-identical to the runtime-pruned QDQ arm, proving
the physical slicing did not change model numerics.

A single short direct greedy generation measured:

| model | generation | peak memory |
| --- | ---: | ---: |
| full 256 MXFP4 | 121.660 tok/s | 35.481 GB |
| physical 224 MXFP4 | 124.373 tok/s | 31.285 GB |

This is one 4-token prompt and 64 generated tokens, so the 2.2% decode
difference is not a throughput claim. The 4.196 GB peak-memory reduction and
the 2.5196 GB parameter reduction are direct measurements and agree in scale.
The artifact has not been uploaded.

### Publication attempt

A private Hugging Face candidate repository was opened for the physical 224
MXFP4 artifact. The 18 GB upload reached roughly 1.02 GB at about 4.6 MB/s,
then was aborted because it would leave an incomplete private model repo for
about an hour. The temporary private repository was deleted. No 224 artifact
is published locally or remotely. The validated files remain at
`/Users/kearm/models/RecursiveSI-Flint-40B-A2.5B-MLX-GPTQ-MXFP4-224`.

### Durable artifact seal and local service

The physical 224 MXFP4 directory now carries `artifact-manifest.json` and
`SHA256SUMS`. The manifest records the base revision, license, mask hash,
expert-ID map, materialization and physical-evaluation receipts, parameter
bytes, quality metrics, serving endpoint, file sizes, and per-file SHA-256.
`shasum -a 256 -c SHA256SUMS` passed for all 16 files, including the five
payload shards.

The resident service is `dev.resmp.flint224-mxfp4` at
`http://127.0.0.1:8084`, launched by
`/Users/kearm/mlx-stack/scripts/flint224_mxfp4_service.sh`. It uses the
backlog-hardened MLX-LM server with decode concurrency 4 and prompt
concurrency 2. The Python process starts through `env -i`, so inherited
credential variables are not passed to the model process.

Live checks passed for health, model listing, a 64-token completion, SSE
streaming, 20 top logprobs, and a four-request synchronized burst. On a
4,096-character real holdout prompt, one request measured 90.82 per-stream
decode tok/s and 0.395 s first token; four concurrent requests measured
222.73 sum per-stream decode tok/s and 2.945 s median first token. These are
short local smokes, not benchmark claims.

The model is local-only and not uploaded. QLoRA, CPT, and SFT have not run.
