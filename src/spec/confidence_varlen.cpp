// src/spec/confidence_varlen.cpp - see include/strata/spec/confidence_varlen.hpp.
#include "strata/spec/confidence_varlen.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace strata::spec {
namespace {

struct Candidate {
    float confidence = 0.0F;
    size_t slot = 0;
    size_t position = 0;
};

// Lower confidence loses; on a tie, the smaller slot-major key was seen first.  In a min-heap this
// puts the currently worst retained candidate on top.
bool worse_than(const Candidate& left, const Candidate& right) {
    if (left.confidence != right.confidence) return left.confidence < right.confidence;
    if (left.slot != right.slot) return left.slot > right.slot;
    return left.position > right.position;
}

bool fail(ConfidenceVarlenPlan* plan, const char** error, const char* message) {
    if (plan != nullptr) {
        plan->row_counts.clear();
        plan->row_offsets.clear();
        plan->allocated_rows = 0;
    }
    if (error != nullptr) *error = message;
    return false;
}

size_t checked_budget(int64_t extra_budget) {
    const uint64_t budget = static_cast<uint64_t>(extra_budget);
    const uint64_t size_max = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    return budget > size_max ? std::numeric_limits<size_t>::max() : static_cast<size_t>(budget);
}

}  // namespace

bool plan_confidence_varlen(const ConfidenceVarlenRequest& request, ConfidenceVarlenPlan* plan,
                            const char** error) {
    if (plan == nullptr) return fail(nullptr, error, "confidence varlen: no plan output");
    if (request.total_rows < 0) {
        return fail(plan, error, "confidence varlen: total row budget is negative");
    }
    if (request.slot_count > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        static_cast<int64_t>(request.slot_count) > request.total_rows) {
        return fail(plan, error, "confidence varlen: budget below one golden row per slot");
    }
    const int64_t golden_rows = static_cast<int64_t>(request.slot_count);
    const int64_t extra_rows_wanted = request.total_rows - golden_rows;
    const size_t extra_budget = checked_budget(extra_rows_wanted);

    if (request.slot_count != 0 &&
        (request.proposal_offsets == nullptr || request.draft_maxima == nullptr)) {
        return fail(plan, error, "confidence varlen: missing proposal metadata");
    }
    if (request.proposal_offsets != nullptr && request.proposal_offsets[0] != 0) {
        return fail(plan, error, "confidence varlen: first proposal offset is not zero");
    }
    for (size_t slot = 0; slot < request.slot_count; ++slot) {
        const size_t left = request.proposal_offsets[slot];
        const size_t right = request.proposal_offsets[slot + 1];
        if (left > right) {
            return fail(plan, error, "confidence varlen: proposal offsets decrease");
        }
        if (request.draft_maxima[slot] < 0) {
            return fail(plan, error, "confidence varlen: draft maximum is negative");
        }
    }
    if (request.confidences == nullptr && request.proposal_offsets != nullptr &&
        request.proposal_offsets[request.slot_count] != 0) {
        return fail(plan, error, "confidence varlen: missing confidence values");
    }

    // The input is bounded to [0, 1], so cumulative products cannot overflow: zero ends a group
    // and ordinary underflow is just the same no-confidence boundary.
    std::vector<Candidate> selected;
    selected.reserve(extra_budget);
    // std::heap's comparator says which element precedes another, so "better before worse" makes
    // the root of this bounded selection the worst candidate still retained.
    const auto retained_before = [](const Candidate& left, const Candidate& right) {
        return worse_than(right, left);
    };
    for (size_t slot = 0; slot < request.slot_count; ++slot) {
        const size_t proposal_end = request.proposal_offsets[slot + 1];
        size_t proposal_count = proposal_end - request.proposal_offsets[slot];
        const int draft_maximum = request.draft_maxima[slot];
        const size_t capped_count = draft_maximum < 0 || static_cast<size_t>(draft_maximum) < proposal_count
                                        ? static_cast<size_t>(draft_maximum)
                                        : proposal_count;
        if (capped_count > extra_budget) proposal_count = extra_budget;
        else proposal_count = capped_count;

        float cumulative = 1.0F;
        for (size_t position = 0; position < proposal_count; ++position) {
            const float confidence = request.confidences[request.proposal_offsets[slot] + position];
            if (!(confidence >= 0.0F && confidence <= 1.0F) || !std::isfinite(confidence)) {
                return fail(plan, error, "confidence varlen: confidence outside [0, 1]");
            }
            cumulative *= confidence;
            if (!(cumulative > 0.0F)) break;
            const Candidate candidate{cumulative, slot, position};
            if (selected.size() < extra_budget) {
                selected.push_back(candidate);
                std::push_heap(selected.begin(), selected.end(), retained_before);
            } else if (worse_than(selected.front(), candidate)) {
                std::pop_heap(selected.begin(), selected.end(), retained_before);
                selected.back() = candidate;
                std::push_heap(selected.begin(), selected.end(), retained_before);
            }
        }
    }

    // Restore global priority order, then original order for exact ties.
    std::sort(selected.begin(), selected.end(), [](const Candidate& left, const Candidate& right) {
        return worse_than(right, left);
    });

    plan->row_counts.assign(request.slot_count, 1);
    plan->row_offsets.resize(request.slot_count);
    for (const Candidate& candidate : selected) {
        // Cumulative confidence is nonincreasing through a slot, so a selected position cannot
        // pass an unselected earlier proposal.  Keep the allocator honest if that invariant slips.
        if (candidate.position >= static_cast<size_t>(plan->row_counts[candidate.slot])) {
            return fail(plan, error, "confidence varlen: draft prefix invariant failed");
        }
        plan->row_counts[candidate.slot] += 1;
    }
    int64_t next_offset = 0;
    for (size_t slot = 0; slot < request.slot_count; ++slot) {
        plan->row_offsets[slot] = next_offset;
        next_offset += plan->row_counts[slot];
    }
    plan->allocated_rows = next_offset;
    return true;
}

}  // namespace strata::spec
