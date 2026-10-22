// include/strata/core/mtp_format.hpp - validation for tools/mtp_rt.py's dense manifest.
//
// The manifest is deliberately small, but a bad offset or block alignment would otherwise become an asynchronous
// CUDA fault far from the loader.  Keep the format contract here so the loader and host tests check the same rules.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace strata::core::mtp_format {

constexpr int kQ4_0 = 2;
constexpr int kQ8_0 = 8;
constexpr int kIQ4NL = 20;
constexpr int kIQ4XS = 23;

struct Tensor {
    std::string name;
    std::string kind;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t off = 0;
    int64_t bytes = 0;
};

inline bool supported(const std::string& kind) {
    return kind == "f32" || kind == "bf16" || kind == "q8_0" || kind == "iq4_nl" || kind == "iq4_xs";
}

inline bool projection(const std::string& kind) {
    return kind == "q8_0" || kind == "iq4_nl" || kind == "iq4_xs";
}

inline int ggml_type(const std::string& kind, bool& ok) {
    ok = true;
    if (kind == "q8_0") return kQ8_0;
    if (kind == "iq4_nl") return kIQ4NL;
    if (kind == "iq4_xs") return kIQ4XS;
    ok = false;
    return -1;
}

inline bool checked_mul(int64_t a, int64_t b, int64_t& out) {
    if (a < 0 || b < 0 || (a != 0 && b > std::numeric_limits<int64_t>::max() / a)) return false;
    out = a * b;
    return true;
}

inline bool expected_bytes(const Tensor& t, int64_t& out) {
    int64_t block_elems = 0, block_bytes = 0;
    if (t.kind == "f32") {
        block_elems = 1;
        block_bytes = 4;
    } else if (t.kind == "bf16") {
        block_elems = 1;
        block_bytes = 2;
    } else if (t.kind == "q8_0") {
        block_elems = 32;
        block_bytes = 34;
    } else if (t.kind == "iq4_nl") {
        block_elems = 32;
        block_bytes = 18;
    } else if (t.kind == "iq4_xs") {
        block_elems = 256;
        block_bytes = 136;
    } else {
        return false;
    }
    // The MMVQ contract asks for a 4-byte aligned pointer; the 256-byte stride is what tools/mtp_rt.py
    // writes, not a kernel requirement, so a hand-made runtime is not refused for a smaller alignment.
    if (t.rows <= 0 || t.cols <= 0 || t.bytes <= 0 || t.cols % block_elems != 0 || t.off % 4 != 0) return false;
    int64_t rows = 0;
    return checked_mul(t.rows, t.cols / block_elems, rows) && checked_mul(rows, block_bytes, out) && out == t.bytes;
}

inline bool validate(const std::vector<Tensor>& tensors, int64_t blob_bytes, std::string& err) {
    if (tensors.empty()) {
        err = "dense.txt is empty";
        return false;
    }
    std::unordered_set<std::string> names;
    for (const Tensor& t : tensors) {
        if (t.name.empty() || !names.insert(t.name).second) {
            err = "dense.txt has a missing or duplicate tensor name: " + t.name;
            return false;
        }
        int64_t want = 0;
        if (!expected_bytes(t, want)) {
            err = "mtp: invalid dense.txt tensor " + t.name + " (" + t.kind + " " + std::to_string(t.rows) + "x" +
                  std::to_string(t.cols) + ", offset " + std::to_string(t.off) + ", bytes " + std::to_string(t.bytes) +
                  ")";
            return false;
        }
        if (t.off > blob_bytes || t.bytes > blob_bytes - t.off) {
            err = "mtp: " + t.name + " exceeds dense.bin";
            return false;
        }
    }
    return true;
}

}  // namespace strata::core::mtp_format
