# Accelerator plugins

This is the operational contract for Strata's opt-in accelerator plugin seam.
The design decision and promotion gates live in
[architecture](architecture.md); this page owns how to build and probe a plugin.

## Boundary

`include/strata/runtime/accelerator_plugin.hpp` defines ABI version 1. The only
operation is `router_logits_bf16`: a plugin copies BF16 router weights when a
router is created, accepts BF16 activation rows, synchronously returns F32
logits, and does not perform top-k selection. Strata retains routing selection
and every default CUDA/CPU path unless an explicit future caller loads a plugin.

The host library resolves ten C symbols with `dlopen`/`LoadLibraryW`, checks ABI
version 1, and rejects missing operations rather than silently falling back.
`strata-plugin-probe` compares plugin output with a CPU BF16
dequantize-and-dot reference and prints one JSON result line.

## Host-safe test

```sh
python3 scripts/check_plugins.py
```

This builds `strata_accelerator_plugin`, a reference CPU plugin, and
`plugin_loader_test`. It does not initialize CUDA, HIP, SYCL, OpenVINO, or an
NPU, and does not claim anything about accelerator performance.

## OpenVINO NPU plugin

The source is `plugins/openvino_npu/router_plugin.cpp`. It is deliberately not
part of the default configure. On the Strix host with the verified OpenVINO
2026.3.1 wheel environment, compile it with the versioned OpenVINO library and
the same include directory:

```sh
OV=/home/kearm/npu-stack/openvino-venv/lib/python3.12/site-packages/openvino
c++ -std=c++20 -O2 -fPIC -shared \
  -Iinclude -I"$OV/include" plugins/openvino_npu/router_plugin.cpp \
  -L"$OV/libs" -Wl,-rpath,"$OV/libs" -Wl,--no-as-needed \
  -l:libopenvino.so.2631 \
  -o build/plugin-remote/libstrata_openvino_npu.so
```

The current Intel NPU driver rejects BF16 as `INFERENCE_PRECISION_HINT` and
supports F16 or I8. The plugin keeps the Strata ABI in BF16, converts weights
and activations to F16, requests F16 inference, and converts the result to F32.
It compiles one decode-token router at a time; the NPU rejects the dynamic token
upper bound used by the first graph attempt.

Probe the representative Qwen router shape:

```sh
./build/plugin-remote/strata-plugin-probe \
  ./build/plugin-remote/libstrata_openvino_npu.so 2560 512 1 0.05
```

The checked 2026-10-10 run reported `Intel(R) AI Boost`, max absolute error
`4.5975e-09`, 0.0214891 s load/compile, and 0.0251889 s execution. That is a
correctness probe, not a speed result and not integration into model generation.
