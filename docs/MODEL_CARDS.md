# Pruning and quantization model-card preparation

This page owns the publication checklist for a future Strata-owned
expert-pruned and quantized Flash-Next artifact. It is a template and gate
list, not a claim that the artifact exists. The instantiated card belongs in
the private Hugging Face artifact repository; this repository records the
contract, evidence, and review history.

## Publication boundary

The target is a private `RESMP-DEV/<descriptive-name>` repository by default.
Public visibility requires a separate current authorization. The artifact must
be derived from a pinned Flash-Next source snapshot, not from the vendor Coder
control artifact. The ISTA-DASLab Coder and 27B prefiller releases are
references and controls only.

Every card must distinguish:

- source weights, retained weights, and generated Strata pack metadata;
- total payload bytes, bytes resident at runtime, and bytes streamed from disk;
- exact router pruning coverage, task quality, and measured throughput;
- one workload's measurement and a general performance claim;
- a clean source revision and an artifact built from that revision.

## Required artifact files

| File | Required contents |
| --- | --- |
| `README.md` | Human-readable card, status, scope, usage, measurements, limits, credits, and licenses |
| `provenance.json` | Source repositories and revisions, source file manifest, tool revisions, corpus/rights boundary, and clean/dirty state |
| `SHA256SUMS` | SHA-256 for every distributed payload and metadata file except the checksum manifest itself |
| `LICENSE` | Exact license text for the artifact and derived weights |
| `NOTICE` | Base-model, tool, and dataset attributions and constraints |
| `pruning-map.json` | `strata-expert-id-map-v1`: 48 unique physical-to-source rows and canonical hash |
| `quantization-allocation.json` | Exact tensor-role-to-format assignment, group/scale semantics, and deterministic build settings |
| `evaluation.json` | Held-out quality, exact controls, runtime hardware, workload, and non-claims |

If calibration data or traces cannot be distributed, publish their schema,
selection rules, rights boundary, and SHA-256 manifests, and state that the
artifact is not fully input-reproducible.

## Card front matter

Use the artifact's exact values after verification; never publish an unverified
template value:

```yaml
---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
license: <verified-license-tag-or-other>
license_name: <verified-license-name>
license_link: <verified-license-url>
tags:
  - gguf
  - strata
  - expert-pruning
  - quantization
  - moe
  - qwen3.8
---
```

Live publication must verify the current base repository metadata and license
rather than copying this template or another artifact's card.

## Required card sections

1. **Status and scope.** Research or promoted state; default/non-default status;
   included and excluded paths; no inherited vision or auxiliary-prefiller
   claim unless separately measured.
2. **At a glance.** Architecture, 48 trunk layers, 512 source experts per layer,
   retained physical experts, 10 active experts per token, PLE shard count,
   heal state, quantization recipe, retained and effective bits per weight,
   payload/resident bytes, source revision, engine binary hash, and HF revision.
3. **Files and integrity.** Per-file bytes, SHA-256, mandatory split semantics,
   resident/streamed role, and whole-file versus tensor-payload verification.
4. **Source identity.** Pinned base model revision, source manifest, exact
   license URL and immutable license blob/hash used for the derivative, tool
   revisions, calibration/heal corpus identity and rights, and explicit
   statement that Coder is only a control.
5. **Pruning map.** Physical/source mapping schema, canonical mapping hash,
   selection method, calibration traces, train/held-out split, controls,
   seeds, and frozen components.
6. **Heal.** If used, method, parameters, data, unhealed control, changed
   tensors, source checkpoint hash, and proof that heal happened before
   quantization.
7. **Quantization.** Every changed tensor role and target format, unchanged
   roles, scales, groups, deterministic settings, exact conversion counts, and
   inverse verification.
8. **Strata pack.** Exact pack-tool revision, generated metadata, pack
   verification, expert-profile binding, and default setup compatibility.
9. **Quality gates.** Held-out symmetrized full-vocabulary KL, perplexity,
   task-quality suites, failed-generation checks, and every control arm.
10. **Runtime gates.** Same-day full-model and vendor-Coder controls, clean
    engine, named hardware, cold and shared-prefix workloads, concurrent
    behavior, memory footprint, startup, and routing/quality bounds.
11. **Reproducibility.** Commands, environment, source dirty state, model and
    tool hashes, raw evidence location, and what cannot be reproduced from the
    public artifact.
12. **Non-claims.** Explicitly state unmeasured quality, hardware classes,
    vision, safety, auxiliary-prefiller behavior, and general throughput.
13. **Credits and license.** Base model, calibration data, GSQ/RCO or
    replacement tools, Strata, and any inherited usage restrictions.

## Promotion gate

A model-card draft may be created only after the candidate artifact has:

1. a deterministic build from a pinned source snapshot;
2. a fully validated physical/source pruning map;
3. quantization allocation and exact pack inversion;
4. held-out quality and task gates against full-model, unhealed-pruned, vendor
   Coder, and trace/uniform mask controls;
5. clean-main same-day runtime controls for cold and shared-prefix agent
   traffic;
6. memory, startup, and occupancy accounting;
7. all distributed files hashed and round-tripped from Hugging Face.

The current full-model control is recorded in
[current-main runtime evidence](../evidence/flash-next-current-main-runtime-2026-10-21.json).
