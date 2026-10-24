#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace strata::program::native_dump_guard {

inline constexpr std::string_view windows_suffix = ".windows.tsv";

inline std::filesystem::path normalize(std::string_view path) {
    return std::filesystem::path(std::string(path)).lexically_normal();
}

inline bool same_path(std::string_view left, std::string_view right) {
    if (left.empty() || right.empty()) return false;
    return normalize(left) == normalize(right);
}

inline std::string windows_path(std::string_view logits_path) {
    return std::string(logits_path) + std::string(windows_suffix);
}

// The native logits binary, its windows TSV, and the residual binary must not overlap.
inline bool collides(std::string_view logits_path, std::string_view residual_path) {
    if (logits_path.empty() || residual_path.empty()) return false;
    return same_path(logits_path, residual_path) ||
           same_path(windows_path(logits_path), residual_path);
}

// Cap emitted rows at the first EOS token, but include that token.
inline int accepted_rows(
    const std::vector<int32_t>& tokens,
    int accepted,
    bool stop_eos,
    const std::vector<int64_t>& eos_ids
) {
    if (accepted < 0) return 0;
    if (!stop_eos || eos_ids.empty()) return accepted;
    const int limit = std::min(accepted, static_cast<int>(tokens.size()));
    for (int i = 0; i < limit; ++i) {
        const auto token = static_cast<int64_t>(tokens[static_cast<size_t>(i)]);
        if (std::find(eos_ids.begin(), eos_ids.end(), token) != eos_ids.end()) return i + 1;
    }
    return accepted;
}

} // namespace strata::program::native_dump_guard
