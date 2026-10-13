# Quantization calibration

This page records the operational distinction between the public iMatrix used
by the first experiment and the RESMP-DEV standard calibration corpus. The
promotion decision remains in [architecture](architecture.md).

## Current identities

The first 3.01-BPW experiment did **not** use the local standard corpus. It
consumed `mradermacher/Qwen3.8-Flash-Next-i1-GGUF@9547b1c8`, file
`Qwen3.8-Flash-Next.imatrix.gguf`, SHA-256
`78661751ccf1e2e301c4313d1feb1394cfa978784daf518e4fe2e0cccdc1c290`, whose
metadata names dataset `imatrix-training-full-3`.
That matrix used 319 chunks of 512 tokens (163,328 calibration tokens).

The RESMP-DEV standard is
`RESMP-DEV/ptq-calibration-corpus@6ef2bbf949b94192ee8bb78d45a2fc840a10cff5`.
For Qwen3.8 targets, the default view is
`data/tokenizers/qwen3.8-27b/calibration_256x2048.npy`, SHA-256
`d7dd281b1f1756a6492361e1dfb0c65fed259f8a3d65eabbcb51809b50bd35d7`, tokenized
by `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It contains
256 disjoint windows of 2,048 token IDs (524,288 tokens), selected by the
dataset's seeded permutation with seed 17. The larger
`corpus/full_calibration_corpus.txt` is a 2,286,227-model-token pool for
building other views; treating a sequential read of that pool as the default
view would silently change sample selection.
The standard view supplies 3.21 times as many calibration tokens as the public
matrix and preserves 2,048-token contexts rather than 512-token contexts.

`tools/standard_imatrix.py` exports that view as
`calibration_256x2048.i32`, a 2,097,152-byte little-endian int32 stream with
SHA-256
`66c6f7e4a68f3e57772064087a22e79e712e5d7c37c2983ec28e00084cab4745`.

The Flash-Next GGUF has 248,320 tokenizer entries, `tokenizer.ggml.pre=qwen35`,
and EOS `248046`; every standard token ID is in range (maximum 248,069). The
pinned `tokenizer.json`, SHA-256
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`, matches
the GGUF token and merge tables; the GGUF's additional entries are explicit
`[PAD...]` rows.

## Exact token path

Raw-text `-f calibration.txt` would retokenize and split a contiguous stream,
which does not preserve the standard view's selected windows. The opt-in
llama.cpp patch adds:

```text
llama-imatrix --token-file tokens.i32
```

where `tokens.i32` is a nonnegative, little-endian int32 stream. The loader
checks every ID against the target vocabulary. `STRATA_TOKEN_IMATRIX=1`
applies the patch to setup's pinned llama.cpp checkout; the normal third-party
source and normal engine path remain unchanged.
A later setup run without the gate reverts the research patch before building.
It can coexist with `STRATA_DETERMINISTIC_IQ=1` when the subsequent quantizer
also needs the validated double-precision IQ2_XS expression.

Prepare, verify, patch, and build the research tool:

```sh
python3 tools/standard_imatrix.py \
  --build-tools \
  --model /path/to/Qwen3.8-Flash-Next.IQ4_XS.gguf \
  --output /path/to/standard.imatrix.gguf \
  --print-command
```

Remove `--print-command` to run synchronously. The default command uses all
256 standard windows at context 2,048, disables perplexity, writes the main
iMatrix every ten chunks, and saves a numbered `.at_N` snapshot every fifty
chunks.
This makes a long run checkpointed without changing llama.cpp's arithmetic.

Resume a checkpoint by adding `--resume`; the helper reads
`imatrix.chunk_count` from the output and asks llama.cpp to skip exactly that
many windows:

```sh
python3 tools/standard_imatrix.py \
  --model model.gguf \
  --output standard.imatrix.gguf \
  --imatrix build/llama-token-tools/bin/llama-imatrix \
  --resume
```

The GGUF metadata key is `imatrix.chunk_count`.

## Exact-copy deterministic rerun

The subsequent routed-expert rerun uses three research gates together:
`STRATA_TOKEN_IMATRIX=1` when building the calibration tool,
`STRATA_DETERMINISTIC_IQ=1` for the IQ2_XS derived-weight expression, and
`STRATA_EXACT_COPY=1` so a complete recipe can preserve every non-target
tensor byte-for-byte while replacing only routed experts. Generate the anchored
recipe from the verified source GGUF:

```sh
python3 tools/exact_copy_recipe.py \
  --model /path/to/Qwen3.8-Flash-Next.IQ4_XS.gguf \
  --output /path/to/standard-iq2xs.recipe
```

For this source, the recipe has 1,224 rows: 96 gate/up rows become IQ2_XS, 48
down rows become Q2_0, and the other 1,080 rows retain their source types.
The anchors and regex escaping are required; llama.cpp treats recipe names as
regular expressions, and an unanchored `ssm_a` also matches `ssm_alpha`.

Build `llama-quantize` with `-ffp-contract=off`, then require a dry run that
loads all 902 iMatrix entries, reports exactly 144 conversions, preserves every
other type, and projects 63,551.82 MiB at 3.01 BPW. Only then run the real
quantization with `--allow-requantize --imatrix ... --tensor-type-file ...
<source> <output> COPY 24`.

## Why this boundary

llama.cpp's native interface is model-tokenized raw text. K-Transformers'
current GPTQ converter instead names a dataset and split, applies the target
chat template, tokenizes and truncates each sample, shuffles with a pinned
seed, and passes a bounded sample count to llm-compressor. Strata's helper
follows the useful part of that contract: pin the dataset and tokenizer, keep
sample boundaries explicit, bound the number of windows, and fail if token IDs
do not belong to the target model.

No quality conclusion follows from changing calibration source. A standard
iMatrix must still feed a reproducible quantization and pass perplexity/output
and same-day performance controls before any setup choice changes.

The corpus is a mixed-source compilation rather than a singly licensed public
dataset. Read its `PROVENANCE.md` before redistributing a derived artifact or
using it outside local PTQ research.
