// src/spec/confidence_varlen_test.cpp - host contract for the portable varlen draft budget.
#include "strata/spec/confidence_varlen.hpp"

#include <cstddef>
#include <cstdio>
#include <limits>
#include <string_view>
#include <vector>

using strata::spec::ConfidenceVarlenPlan;
using strata::spec::ConfidenceVarlenRequest;
using strata::spec::plan_confidence_varlen;

namespace {
int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

ConfidenceVarlenRequest make_request(const std::vector<float>& values,
                                     const std::vector<size_t>& offsets,
                                     const std::vector<int>& maxima,
                                     int64_t total_rows) {
    ConfidenceVarlenRequest request;
    request.confidences = values.data();
    request.proposal_offsets = offsets.data();
    request.draft_maxima = maxima.data();
    request.slot_count = maxima.size();
    request.total_rows = total_rows;
    return request;
}

bool plan_is(const ConfidenceVarlenPlan& plan, const std::vector<int>& counts,
             const std::vector<int64_t>& offsets, int64_t allocated) {
    return plan.row_counts == counts && plan.row_offsets == offsets &&
           plan.allocated_rows == allocated;
}
}  // namespace

int main() {
    std::printf("confidence_varlen_test\n");

    {
        // DeepInfer's cumulative products: 0.8 then 0.8 is stronger than a fresh 0.6 proposal.
        const std::vector<float> values{1.0F, 0.5F, 0.25F, 0.8F, 0.8F, 0.6F};
        const std::vector<size_t> offsets{0, 3, 5, 6};
        const std::vector<int> maxima{3, 2, 1};
        auto request = make_request(values, offsets, maxima, 8);
        ConfidenceVarlenPlan plan;
        const char* error = nullptr;
        // Budget 7 leaves four extra rows; cumulative confidence selects .8, .64, .6, then .5.
        request.total_rows = 7;
        check(plan_confidence_varlen(request, &plan, &error), "valid request plans");
        check(error == nullptr, "valid request reports no error");
        check(plan_is(plan, {2, 3, 2}, {0, 2, 5}, 7),
              "cumulative confidence chooses the global prefix");
    }

    {
        const std::vector<float> values{1.0F, 1.0F, 1.0F, 1.0F};
        const std::vector<size_t> offsets{0, 2, 4};
        const std::vector<int> maxima{2, 2};
        ConfidenceVarlenPlan plan;
        check(plan_confidence_varlen(make_request(values, offsets, maxima, 3), &plan) &&
                  plan_is(plan, {2, 1}, {0, 2}, 3),
              "equal confidence keeps slot-major proposal order");

        const std::vector<float> capped{1.0F, 0.9F, 0.9F, 1.0F};
        const std::vector<int> caps{0, 2};
        const bool capped_ok = plan_confidence_varlen(make_request(capped, offsets, caps, 4), &plan);
        check(capped_ok && plan_is(plan, {1, 3}, {0, 1}, 4),
              "a per-slot maximum truncates its proposal prefix");
    }

    {
        const std::vector<float> values{0.0F, 0.9F, 0.5F, 0.0F};
        const std::vector<size_t> offsets{0, 2, 4};
        const std::vector<int> maxima{2, 2};
        ConfidenceVarlenPlan plan;
        check(plan_confidence_varlen(make_request(values, offsets, maxima, 8), &plan) &&
                  plan_is(plan, {1, 2}, {0, 1}, 3),
              "zero confidence ends the slot at its golden row");
    }

    {
        const std::vector<float> values{1.0F, 1.0F};
        const std::vector<size_t> offsets{0, 1, 2};
        const std::vector<int> maxima{1, 1};
        ConfidenceVarlenPlan plan;
        check(plan_confidence_varlen(make_request(values, offsets, maxima, 2), &plan) &&
                  plan_is(plan, {1, 1}, {0, 1}, 2),
              "an exact golden-row budget proposes no drafts");
    }

    {
        const std::vector<float> values{1.5F};
        const std::vector<size_t> offsets{0, 1};
        const std::vector<int> maxima{1};
        ConfidenceVarlenPlan plan;
        const char* error = nullptr;
        check(!plan_confidence_varlen(make_request(values, offsets, maxima, 2), &plan, &error),
              "confidence above one is rejected");
        check(plan.row_counts.empty() && plan.row_offsets.empty() && plan.allocated_rows == 0,
              "a failed request leaves no partial plan");
    }

    {
        // These extents exceed the tiny budget and would overflow a small signed row count, but
        // only budgeted rows are ranked or emitted.
        const float confidence = 1.0F;
        const std::vector<size_t> offsets{0, std::numeric_limits<size_t>::max(),
                                          std::numeric_limits<size_t>::max()};
        const std::vector<int> maxima{std::numeric_limits<int>::max(), 0};
        ConfidenceVarlenRequest request;
        request.confidences = &confidence;
        request.proposal_offsets = offsets.data();
        request.draft_maxima = maxima.data();
        request.slot_count = 2;
        request.total_rows = 2;
        ConfidenceVarlenPlan plan;
        check(plan_confidence_varlen(request, &plan) &&
                  plan_is(plan, {1, 1}, {0, 1}, 2),
              "oversized extents are capped without signed-count overflow");
    }

    {
        ConfidenceVarlenRequest request;
        request.slot_count = std::numeric_limits<int64_t>::max();
        request.total_rows = 1;
        ConfidenceVarlenPlan plan;
        const char* error = nullptr;
        check(!plan_confidence_varlen(request, &plan, &error) &&
                  std::string_view("confidence varlen: budget below one golden row per slot") ==
                      error,
              "an impossible golden-row count is rejected");
    }

    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
