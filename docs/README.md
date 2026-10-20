# Documentation index

This index routes readers to the operational detail. The README remains the
product entry point, architecture remains the single implementation plan and
decision record, and each page below owns only its specialized procedure or
evidence.

## Authorities

| Reference | Authority |
| --- | --- |
| [README](../README.md) | Product entry point and current product status decisions |
| [Architecture](architecture.md) | Single implementation plan, ownership, choices and work log |
| [Contributing](../CONTRIBUTING.md) | Procedure, code standards and required checks |
| [Agent contract](../AGENTS.md) | Contributor and agent boundaries |

## Start here

| Document | Use it for |
| --- | --- |
| [AI setup](AI_SETUP.md) | Let a coding agent inspect the PC and complete installation |
| [Install](INSTALL.md) | Requirements, Windows/Linux setup, updates, Docker, storage and options |
| [Models](MODELS.md) | Model sizes, variants, RAM fit, downloads and measured speed |
| [How it works](HOW_IT_WORKS.md) | The plain-language engine, memory-tier and speculative-decoding model |
| [Details](DETAILS.md) | The long-form user and API reference |
| [Troubleshooting](TROUBLESHOOTING.md) | Startup, memory, GPU, port and update failures |

## Serving and applications

| Document | Use it for |
| --- | --- |
| [Batching](BATCHING.md) | Parallel requests, slots, fairness and concurrency limits |
| [MCP server](MCP_SERVER.md) | Agent-safe install, status and control operations |
| [Test requests](TEST_REQUESTS.md) | Representative API and model smoke requests |
| [llama-swap](LLAMA_SWAP.md) | Swapping Strata in a multi-model llama-swap deployment |

## NVIDIA, AMD and Intel hardware

| Document | Use it for |
| --- | --- |
| [AMD HIP](AMD_HIP.md) | Radeon setup, kernels, tuning and measured behavior |
| [AMD HIP performance](AMD_HIP_PERFORMANCE.md) | Radeon-specific performance procedures and controls |
| [Intel Arc](INTEL_ARC.md) | Community Arc setup and validation |
| [Intel oneAPI/SYCL](INTEL.md) | Experimental Intel backend build and qualification |
| [Multi-GPU](MULTI_GPU.md) | Layer splits, multiple NVIDIA/AMD cards and topology |
| [Second GPU](SECOND_GPU.md) | Adding a helper card and PCIe work sharing |
| [NVIDIA V100](NVIDIA_V100.md) | Volta kernels and validation |
| [Older GPUs](OLDER_GPUS.md) | Turing, Pascal, Volta and older Radeon community paths |
| [Strix Halo](STRIX_HALO.md) | AMD Ryzen AI Max build and tuning |
| [Accelerator plugins](ACCELERATOR_PLUGINS.md) | Backend-neutral ABI and external NPU/OpenVINO routing |
| [Strix watcher](STRIX_WATCH.md) | Persistent Strix host telemetry and failure diagnosis |

## Memory, KV and expert placement

| Document | Use it for |
| --- | --- |
| [Batched DMA](BATCHED_DMA.md) | Batched host/device transfer design and controls |
| [Disjoint expert cache](DISJOINT_EXPERT_CACHE.md) | Avoiding shared expert-cache corruption |
| [Exchange rotation](EXCHANGE_ROTATION.md) | Rotating resident expert complements safely |
| [KV disaggregation](KV_DISAGGREGATION.md) | Local/shared KV pool architecture and ABI |
| [KV prefetch](KV_PREFETCH.md) | Prefetch controls and measured effects |
| [Message-boundary cache](MESSAGE_BOUNDARY_CACHE.md) | Reusing state at chat message boundaries |
| [P4800X KV tier](OPTANE_KV_TIER.md) | B550 fast-NVMe cold-tier operations and identity |
| [Prompt-cache tail](PROMPT_CACHE_TAIL.md) | Long-prompt tail cache behavior |
| [VRAM elastic](VRAM_ELASTIC.md) | Dynamic VRAM allocation and resize behavior |

## Quantization and model variants

| Document | Use it for |
| --- | --- |
| [Calibration](CALIBRATION.md) | iMatrix provenance, token views and deterministic reruns |
| [Orca manual setup](ORCA.md) | OrcaRouter IQ3_XXS compatibility and launch |
| [Orca Q4_K_S](ORCA_Q4_K_S.md) | Four-bit OrcaRouter pack procedure |
| [Unsloth Q4](UNSLOTH_Q4.md) | UD-Q4_K_XL and UD-IQ4_XS setup and verification |
| [Unsloth Q6](UNSLOTH_Q6.md) | UD-Q6_K_XL artifact verification and use |
| [Flash-Next prefiller](PREFILLER.md) | Phase-specialized auxiliary prefiller architecture and planner |

## Measurements and research

| Document | Use it for |
| --- | --- |
| [Community benchmarks](COMMUNITY_BENCHMARKS.md) | External hardware reports and submission format |
| [Research runs](RESEARCH_RUNS.md) | Local research controls and dated measurement series |
| [MMVQ IL table](MMVQ_IL_TABLE.md) | Instruction-level table generation and use |
| [Windows Zen2 gfx1030 workers](benchmarks/windows-zen2-gfx1030-workers.md) | Dated community worker benchmark detail |
| [Strata paper](paper/Strata-Paper.pdf) | The engine's detailed design paper |
