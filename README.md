<h1 align="center">Strata</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run a 125-billion-parameter AI model on your own gaming PC</b><br>
NVIDIA or AMD graphics card (12 GB recommended) · Windows or Linux · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on a normal PC. This is a
large, smart AI model that usually needs a server. It chats, writes code, reads pictures and works with your apps
and coding agents. Nothing leaves your PC.

The complete documentation map is [docs/README.md](docs/README.md). The detailed
implementation and exercise matrix is [project status](docs/architecture.md#project-status-and-governance);
[architecture](docs/architecture.md) owns decisions and the append-only work log.

## Project status and governance

| Area | Current position |
| --- | --- |
| Product surface | The inherited upstream engine, installer, local API and web app are implemented; local changes remain opt-in unless a measured architecture decision changes them |
| Verification | `just quality` passes on this macOS host and covers host-safe contracts; it does not claim a CUDA, HIP or SYCL build |
| Hardware and serving research | B550 and Strix evidence covers calibrated Qwen execution, topology, long sessions, KV/session storage, embedding placement and the cold-KV control; every claim is scoped in the status matrix |
| Current research | The Flash-Next auxiliary prefiller has a validated metadata planner and architecture, but no trainer, runtime path or quality/speed claim |

See [project status](docs/architecture.md#project-status-and-governance) for every row, receipt and non-claim.

## How fast is it?

We measured Strata on an ordinary RTX 5070 PC and an AMD RX 9070 XT PC. Q2_0 writes about 94 tokens per second
and reads about 2,650 tokens per second on the RTX 5070; the full matched tables and context lengths are in
[model speed](docs/MODELS.md#how-fast-is-each-size), [implementation details](docs/DETAILS.md#speed-measured) and
[community results](docs/COMMUNITY_BENCHMARKS.md).

## What you need

| | Short answer |
| --- | --- |
| **Graphics card** | NVIDIA RTX 20-50 or a supported AMD Radeon; 12 GB VRAM is recommended, NVIDIA 8 GB runs slowly |
| **RAM** | 32 GB or more; 64 GB runs every normal size |
| **Disk** | About 80 GB free, preferably NVMe SSD |
| **System** | Windows 10/11 or Linux with a current NVIDIA/AMD driver |

Experimental older GPUs, Intel Arc, Strix Halo and pre-AVX2 CPUs have separate
[hardware guides](docs/README.md#nvidia-amd-and-intel-hardware). The full requirement table is
[installation requirements](docs/INSTALL.md#what-you-need).

## Install

### Let your AI set it up

Paste this into an AI coding assistant:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk, chooses a model that fits, installs and starts Strata, then explains
how to connect your applications. Agents can also use the [MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it, or clone it.
On Windows, double-click `START-HERE.bat`. On Linux, run `./setup.sh`.

The installer chooses the NVIDIA or AMD engine, asks a few model/context/image questions, downloads about 70 GB
and opens `http://127.0.0.1:8080`. Run the same launcher next time to start immediately. Every platform, Docker,
update, storage and tuning detail is in [INSTALL.md](docs/INSTALL.md).

> The first start can make the PC look busy for 1-3 minutes while Strata loads and locks model memory. Wait for
> the launcher to finish; do not close its window.

## Which model should I pick?

Not sure? Choose **IQ2_XS** with 48-64 GB RAM, or the **Coder** when only 32 GB is available. The Coder is
strongest for code but weaker outside it; full-expert Q2_0, IQ2_XS or IQ3_S are better general choices. IQ3_S is
the highest-quality normal size, and Unsloth's 4-bit variants need more storage and RAM.

The RAM table, size comparisons, licenses and variant tradeoffs are in [MODELS.md](docs/MODELS.md).

## Using it

- **Browser:** open `http://127.0.0.1:8080` for Chat, the live Monitor and current settings.
- **Apps and coding agents:** use an OpenAI-compatible base URL at `http://127.0.0.1:8080/v1`; any API key and
  model name work.
- **Anthropic API apps:** use `http://127.0.0.1:8080/v1/messages`.
- **Responses API apps:** use `/v1/responses` ([Codex setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** select off, low, medium or high in Chat or through reasoning-effort controls.
- **Pictures:** enable Images during setup, then attach them in Chat or an application.
- **Other devices:** expose deliberately with `--host 0.0.0.0 --api-key <secret>`; never expose the default
  server without a key.
- **Concurrency and long prompts:** see [batching](docs/BATCHING.md) and [API details](docs/DETAILS.md#using-it).

## Something went wrong?

If installation stops, run the launcher again; it resumes. If the PC freezes at first start, wait, then reduce
other memory use or pick a smaller model. If it is slow with heavy disk activity, close other programs or choose
Q2_0 or IQ2_XS. If port 8080 is busy, Strata may already be running.

The complete failure guide and log locations are [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## How does it work?

Strata divides the 24,576-expert model between the GPU, RAM, CPU and SSD. Hot experts stay on the graphics card,
the full set remains available in host memory, a lookup table lives on disk, and a small helper guesses words that
the large model verifies together.

The plain-language explanation is [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md); the detailed numbers and API reference
are in [DETAILS.md](docs/DETAILS.md).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was
compressed by ISTA-DASLab, UkisAI and Unsloth. Strata uses parts of llama.cpp/ggml. Full credits and license
boundaries are in [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
