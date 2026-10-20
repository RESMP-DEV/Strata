<h1 align="center">Strata</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

## Project status and governance

| Area | Current status |
| --- | --- |
| Engine, installer, local API, web app | Implemented and documented by the inherited upstream code at `fb58e0d`; not rebuilt or measured locally by the governance change on 2026-10-09 |
| Local project governance | Lapis-style authority structure in place: [AGENTS.md](AGENTS.md), [architecture](docs/architecture.md), [CONTRIBUTING.md](CONTRIBUTING.md), and [stand-up evidence](evidence/project-standup.json) |
| Common verification | `just quality` runs on this macOS host; it does not claim a CUDA, HIP, or SYCL build |
| Strix hardware evidence | An 8 GB RTX 5070 engine build launched; the Intel NPU also passed an external, opt-in one-token router probe through the new plugin ABI. Neither is a Strata model-run or speed claim. See [accelerator evidence](evidence/accelerator-plugin-router-probe.json) |
| DeepSeek V4.1 port preparation | Exact 1017 checkpoint metadata taxonomy, model geometry, and deterministic CPU candidate selection are implemented and exercised host-side on both this machine and the B550 compute host. B550 also built the regular Qwen engine and passed its AVX2 parity targets; its DeepSeek-specific AVX2 ceiling is measured but no DeepSeek pack or backend exists. See [first host contracts](evidence/deepseek-v41-first-host-contracts.json), [B550 bring-up](evidence/deepseek-v41-b550-bringup.json), and [AVX2 ceiling](evidence/deepseek-v41-b550-avx2-ceiling.json) |
| Qwen B550 storage placement | The HDD source store sustains 202 MiB/s sequential reads but only 106 MiB/s for Strata-sized random expert-slot misses; the root NVMe is 10.8x faster at queue depth 1 and 18.3x at depth 8. Stage runnable Qwen packs on NVMe scratch. See [storage baseline](evidence/qwen-b550-storage-read-baseline.json) |
| Qwen B550 calibrated IQ2_XS execution | The staged split GGUF now has a shard-aware pack, the model loaded and executed on both cards and through the two-GPU server, and the missing Q2_0 MTP drafter was fetched, verified, packed, and exercised. First observations put two GPUs around 1.9x GPU 1 alone; with the embedding replicas stopped, all 24,576 experts fit resident and a five-request short-prompt control reached 227.4 tok/s median decode. Broader performance and quality remain unclaimed. See [B550 baseline](evidence/qwen-b550-iq2xs-bringup-and-mtp-baseline.json) and [clean control](evidence/qwen-b550-full-vram-clean-control.json) |
| Speculative-head/QAT target research | The next high-upside research head is DSpark-style DFlash2 (rank-256 Markov plus calibrated confidence), while the already-running native Qwen MTP is the first practical QAT lane. No QAT backend or trained artifact is claimed yet. See [head research](evidence/qwen-dspark-mtp-qat-target-research.json) |
| DSpark2 Qwen 27B parameter plan | The local 11,273-cycle DFlash2 corpus, prior G2/G3 searches, official DeepSpec defaults, and a live B550 smoke define a staged block-7 Markov/confidence training plan. Existing top-64 captures cannot train the official TV/confidence loss unchanged. See [parameter evidence](evidence/dspark2-qwen27b-training-parameters.json) |
| DSpark2-Qwen full-logit distillation | The MLX DFlash2 prediction-head template now captures full-vocab F32 logits and trains DSpark-weighted hybrid loss (CE 0.1 / TV 0.9) with an MXFP4 QAT path. Prime RTX 6000 Ada is the selected lane: a real A100 80GB control was 1.53x slower and 2.51x costlier per step. This trains the Qwen3.8-27B prototype, not Flash-Next, and no candidate is promoted or live-accepted. See [cloud training state](evidence/dspark2-qwen-cloud-training-state.json) and [GPU selection](evidence/dspark2-gpu-provider-selection.json) |
| Qwen B550 concurrency and embedding tiers | Strata serves concurrent agent requests via `--batch` slots; measured on one short coding-prompt family over 3 rounds, MTP activity inside concurrent batch slots is unconfirmed. On this launch-bound workload concurrency buys latency fairness, not aggregate throughput (two-GPU full-residency: 149.6/126.3/158.1 tok/s aggregate at C=1/2/4; per-request 199.8 to 49.7). CPU-only embedding of the small Qwen3-0.6B is reasonable for light use (0.020 s single-request, faster than the 0.081 s GPU round-trip) but 4.7x slower on bulk (38.8 vs 183.1 items/s); the sindexer MXFP4 Jina embedder is CUDA-only, not a CPU path. See [concurrency evidence](evidence/qwen-b550-concurrency-fallback.json) |
| Strix MXFP4 embedding tier | The sindexer Jina MXFP4 embedder runs on Strix's RTX 5070 Laptop (SM120): the Triton MXFP4 kernel executes there, and the 1536-dim output matches the B550 reference at cosine 0.9999. It is the low-latency 1536-dimensional MXFP4 tier (0.036 s single-request versus 0.081 s on B550) but far weaker on bulk (15.9 versus 183.1 items/s). It runs standalone on :1236 and is not in the sindexer routing path; the NPU path is untested. See [Strix tier evidence](evidence/qwen-strix-mxfp4-embed-tier.json) |
| B550 layer-split topology | B550's GPU0 is chipset-attached at Gen3 x4 (2.9 GB/s) while GPU1 is CPU-root Gen4 x16 (25.4 GB/s). On a verified-clean GPU, an explicit `--layer-split 30` with the fast card first measured 236.0 to 243.6 tok/s decode against the auto control's 226.9 to 227.3 (about 4 to 7%), and 165.9 versus 151.8 tok/s prompt. Reversing card order alone gains about 1%, and `STRATA_SPLIT_MISS_MS` does not move the placement. No deployed configuration was changed. See [topology evidence](evidence/qwen-b550-layer-split-topology.json) |
| B550 two-request serving policy | C=2 batch slots lose aggregate throughput even with topology-aware groups: 127.0 versus 142.7 tok/s at C=1, and split-30 falls to 106.3. Independent per-GPU servers and a shared CPU arena are worse. For the measured exact short-prompt shape, queue serially on split-30: ten requests reached 222.5 tok/s aggregate and about 252.6 tok/s per request; varied-prompt serial C=2 and longer prompts remain unmeasured. No deployed routing tier was changed. See [queueing evidence](evidence/qwen-b550-c2-queue-topology.json) |
| B550 long-session serving | At 256K context with KV streaming (32,768 hot cells per QSA layer in VRAM, the engine's measured 1.80 GiB pinned host allocation, against a 3.09 GiB full-context payload upper bound), a 112,626-token real-source prompt prefilled in 27.743 s (4059.5 tok/s) and decoded at 117.1 tok/s with a 99.5% expert-cache hit rate and 92% of KV reads from VRAM. RAM is not the constraint: `--ple-io ram` was 2.5x slower to prefill and 1.9x slower to decode than direct SSD reads on this 91 GiB host. See [long-session evidence](evidence/qwen-b550-longrun-256k.json) |
| B550 Optane KV sessions | Native session files can park and restore whole long conversations on Optane: a real 112,626-token state became a 1.956 GB file in 1.463 s and restored into a restarted single-GPU process in 2.232 s. This is explicit session parking, not live per-token NVMe paging; layer-split, peer-device, and batch modes remain incompatible, and Mooncake is reserved for a future multi-host shared pool. See [Optane session evidence](evidence/qwen-b550-optane-session-store.json) |
| Disaggregated KV tier | The target is a coordinator over L0 VRAM, L1 pinned host slots, L2 local Optane/NVMe groups, and optional L3 shared storage, described in [KV disaggregation](docs/KV_DISAGGREGATION.md). The current `--kv-disk` prototype is quarantined experimental code: its hooks are not yet on the serving verify path, and it has no accepted live-model result. Operational rules for the device are in [Optane operations](docs/OPTANE_KV_TIER.md). |

<p align="center"><b>Run a 125-billion-parameter AI model on your own gaming PC</b><br>
NVIDIA or AMD graphics card (12 GB recommended) · Windows or Linux · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on a normal PC. This is a
large, smart AI model that usually needs a server. It chats, writes code, reads pictures and works with your apps
and coding agents. Nothing leaves your PC.

## How fast is it?

We measured it on two ordinary gaming PCs. A token is about ¾ of a word.

- **Writes answers:** how fast the reply appears in a short chat. 60 tokens per second is faster than you can read.
- **Reads your prompt:** how fast it takes in what you send (here a 32K-token document, code or chat history).

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 with engine 0.1.36, the other rows with 0.1.26 (4K answers, 32K prompts). The full tables are in
[DETAILS.md](docs/DETAILS.md#speed-measured). A card with more VRAM is faster: an RTX 3090 (24 GB) should write
about 100-140 tokens per second. Long chats and other cards: [speed of each model](docs/MODELS.md#how-fast-is-each-size),
[community results](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata is free. If it runs well on your PC, a coffee keeps the work on it going.</sub></p>

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series. **12 GB is recommended**; an NVIDIA card with 8 GB runs, but slowly. |
| **RAM** | 32 GB or more. Your RAM decides [which model](#which-model-should-i-pick) fits. 64 GB runs every size. |
| **Disk** | About 80 GB free. Use an SSD if you can: the first start is much faster. |
| **System** | Windows 10 / 11 or Linux, and a current graphics driver from NVIDIA or AMD. |

The installer sets up everything else. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).

Experimental, written and tested by community members on their own machines:

- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, built from source on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, built from source on Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk and picks the model that fits. Then it installs and starts it and tells
you how to connect your apps. AI tools can also install, start and stop Strata through its
[MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the Strata folder.

The steps are the same for NVIDIA and AMD. The installer finds your card and sets up the right engine for it. It
asks you a few questions:

- which model and which size,
- how much context (how much text the model keeps in mind),
- whether it should read pictures.

Press Enter each time for the recommended answer. Then it downloads the model (about 70 GB) and starts it. If the
download stops, run it again: it continues where it left off. Your browser opens the Strata app at
`http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time).
> Strata loads 35-55 GB into your RAM and locks part of it for the graphics card. This is normal. Wait, and don't
> close the window. The window shows what Strata is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker,
several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

## Which model should I pick?

The installer recommends one for your RAM. The same model comes in several sizes, compressed more or less. Smaller
sizes are faster. Larger sizes are a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's UD-IQ4_XS (~4-bit) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder):** a coding version with half of the experts removed. It reaches 91% of the full
  model's SWE-bench Verified score (measured by its authors) and fits 32 GB of RAM. It is weaker outside code,
  including Chinese and other CJK text (#438). For those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15):** a fine-tune that thinks for a much shorter time before it answers. You
  get the answer sooner, at about the same quality.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** Unsloth's ~4-bit version, between IQ3_S and
  UD-Q4_K_XL in quality. A 94 GB download. With less than ~80 GB of RAM, Strata reads part of it from the SSD
  while it answers, so it is slower there (an NVMe SSD helps).
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (experimental): the closest to the full
  model. But Strata reads most of it from the SSD while it answers, so it writes only 7-8.5 tokens/s on a 64 GB PC.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** you set it up by hand. It is
  not in the installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). To add another model later, run
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** open `http://127.0.0.1:8080`. It has **Chat**, a live **Monitor** of the model and your
  GPU/CPU/RAM, and **About** with the settings and addresses.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the
  fastest. High is best for hard questions.
- **Pictures:** say yes to "Images?" in setup. Then click **Picture** in the chat, or attach pictures in your app.
  AMD cards read pictures on Linux through the processor; on Windows they can't yet.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **One request at a time:** by default Strata answers one request, and the others wait. To answer several at once,
  set `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card this makes each answer slower.
- **Long prompts:** Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up
  messages start in seconds.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

## Something went wrong?

- **My PC froze the first time Strata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again. It continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** Strata is already running. Look for its window.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder. Found a
security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was
compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
