// include/strata/spec/confidence_varlen.hpp - confidence-ranked varlen row budget.
//
// A slot always contributes the golden token it is about to verify.  Its draft proposals are
// ranked by the cumulative product of their conditional confidence values, the allocation
// principle DeepInfer uses for confidence-rate varlen scheduling: taking a later proposal also
// pays for every earlier proposal in that slot.  This pure policy chooses which of those groups
// fit a global row budget and emits the contiguous metadata the engine expects.  It neither
// drafts, verifies, nor touches runtime state.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::spec {

/// Proposals are slot-major: `[proposal_offsets[i], proposal_offsets[i + 1])` indexes
/// `confidences`.  `draft_maxima[i]` is an independent per-slot cap on proposal rows.
struct ConfidenceVarlenRequest {
    const float* confidences = nullptr;
    const size_t* proposal_offsets = nullptr;   ///< slot_count + 1 monotonic offsets from 0
    const int* draft_maxima = nullptr;
    size_t slot_count = 0;
    int64_t total_rows = 0;                    ///< maximum rows, including the golden rows
};

/// `row_counts[i]` is 1 + drafts for slot i.  `row_offsets[i]` is where that contiguous group
/// starts, so consecutive offsets cover exactly `allocated_rows` rows.
struct ConfidenceVarlenPlan {
    std::vector<int> row_counts;
    std::vector<int64_t> row_offsets;
    int64_t allocated_rows = 0;
};

/// Fill `plan` with at most `request.total_rows` rows.  On failure, clear it and, when `error`
/// is non-null, point it at a static reason.  Equal priorities keep slot-major proposal order;
/// zero cumulative confidence proposes no row beyond the golden token.
bool plan_confidence_varlen(const ConfidenceVarlenRequest& request, ConfidenceVarlenPlan* plan,
                            const char** error = nullptr);

}  // namespace strata::spec
