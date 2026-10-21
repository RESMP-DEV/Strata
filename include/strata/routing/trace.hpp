#pragma once

#include <bit>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

namespace strata::routing {

inline constexpr uint32_t kTraceMagic = 0x50525453u;   // bytes "STRP" on the supported little-endian hosts
inline constexpr uint16_t kTraceVersion = 2;
inline constexpr uint16_t kTracePhaseBatchedPrefill = 1;
inline constexpr uint32_t kTraceFlagsNone = 0;
inline constexpr uint64_t kBatchedPrefillRecordHeaderBytes = 32;

static_assert(std::endian::native == std::endian::little, "Strata routing traces are little-endian");

#pragma pack(push, 1)
struct TraceHeader {
    uint32_t magic = kTraceMagic;
    uint16_t version = kTraceVersion;
    uint16_t phase = kTracePhaseBatchedPrefill;
    uint32_t flags = kTraceFlagsNone;
    uint64_t layer_count = 0;
    uint64_t top_k = 0;
    uint64_t record_header_bytes = kBatchedPrefillRecordHeaderBytes;
    uint32_t reserved = 0;
};

struct BatchedPrefillRecordHeader {
    int32_t layer = 0;
    int32_t k = 0;
    int64_t position = 0;
    int32_t token_count = 0;
    int32_t reserved = 0;
    uint64_t request_id = 0;
};
#pragma pack(pop)

static_assert(sizeof(TraceHeader) == 40, "the routing trace header layout is a public contract");
static_assert(sizeof(BatchedPrefillRecordHeader) == 32, "the routing record header layout is a public contract");

[[nodiscard]] inline bool write_trace_header(
    std::FILE* output,
    uint64_t layer_count,
    uint64_t top_k,
    std::string* error
) {
    if (output == nullptr) {
        if (error != nullptr) *error = "routing trace output is closed";
        return false;
    }
    if (layer_count == 0 || top_k == 0) {
        if (error != nullptr) *error = "routing trace geometry must be positive";
        return false;
    }
    const TraceHeader header{.layer_count = layer_count, .top_k = top_k};
    if (std::fwrite(&header, sizeof header, 1, output) != 1) {
        if (error != nullptr) *error = "cannot write the routing trace header";
        return false;
    }
    return true;
}

[[nodiscard]] inline bool write_batched_prefill_routing(
    std::FILE* output,
    uint64_t request_id,
    int64_t layer,
    int64_t position,
    int64_t token_count,
    int64_t top_k,
    const int32_t* ids,
    const float* weights,
    std::string* error
) {
    if (output == nullptr) {
        if (error != nullptr) *error = "routing trace output is closed";
        return false;
    }
    if (layer < 0 || layer > std::numeric_limits<int32_t>::max()) {
        if (error != nullptr) *error = "routing trace layer is out of range";
        return false;
    }
    if (position < 0) {
        if (error != nullptr) *error = "routing trace position is negative";
        return false;
    }
    if (token_count <= 0 || token_count > std::numeric_limits<int32_t>::max()) {
        if (error != nullptr) *error = "routing trace token count is out of range";
        return false;
    }
    if (top_k <= 0 || top_k > std::numeric_limits<int32_t>::max()) {
        if (error != nullptr) *error = "routing trace top-k is out of range";
        return false;
    }
    if (ids == nullptr || weights == nullptr) {
        if (error != nullptr) *error = "routing trace payload is null";
        return false;
    }
    if ((std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(token_count)) <
        static_cast<uint64_t>(top_k)) {
        if (error != nullptr) *error = "routing trace payload size overflows";
        return false;
    }
    const uint64_t entries = static_cast<uint64_t>(token_count) * static_cast<uint64_t>(top_k);
    if (entries > std::numeric_limits<size_t>::max() / sizeof(int32_t)) {
        if (error != nullptr) *error = "routing trace payload size exceeds the host address space";
        return false;
    }
    const size_t count = static_cast<size_t>(entries);
    const BatchedPrefillRecordHeader record{
        .layer = static_cast<int32_t>(layer),
        .k = static_cast<int32_t>(top_k),
        .position = position,
        .token_count = static_cast<int32_t>(token_count),
        .reserved = 0,
        .request_id = request_id,
    };
    if (std::fwrite(&record, sizeof record, 1, output) != 1 ||
        std::fwrite(ids, sizeof(int32_t), count, output) != count ||
        std::fwrite(weights, sizeof(float), count, output) != count) {
        if (error != nullptr) *error = "cannot write a batched-prefill routing record";
        return false;
    }
    return true;
}

}  // namespace strata::routing
