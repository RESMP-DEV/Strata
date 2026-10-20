// src/core/kv_placement.cpp - see include/strata/core/kv_placement.hpp.
#include "strata/core/kv_placement.hpp"

#include <limits>

namespace strata::core {

const char* KvPlacementPolicyState::name() const {
    switch (policy_) {
        case KvPlacementPolicy::kFrequency: return "frequency";
        case KvPlacementPolicy::kScan: return "scan";
        case KvPlacementPolicy::kClock:
        default: return "clock";
    }
}

void KvPlacementPolicyState::admit(int32_t group, uint64_t now, bool pinned) {
    Entry& entry = facts_[group];
    entry.last_used = now;
    entry.pinned = pinned;
}

void KvPlacementPolicyState::touch(int32_t group, uint64_t now) {
    const auto it = facts_.find(group);
    if (it == facts_.end()) return;
    it->second.last_used = now;
    ++it->second.read_count;
}

void KvPlacementPolicyState::remove(int32_t group) { facts_.erase(group); }

KvPlacementDecision KvPlacementPolicyState::victim() const {
    KvPlacementDecision decision;
    if (facts_.empty()) return decision;
    int32_t best_group = -1;
    uint64_t best_key = std::numeric_limits<uint64_t>::max();
    uint64_t best_recency = std::numeric_limits<uint64_t>::max();
    for (const auto& entry : facts_) {
        if (entry.second.pinned) continue;   // the open tail is never a victim
        // kFrequency: the read count is the primary key and the recency breaks the tie. kClock: the recency is
        // the key itself. The recency is a SEPARATE variable, because comparing it against the read count would
        // compare a timestamp with a count and every tie would take the same branch.
        const uint64_t key = policy_ == KvPlacementPolicy::kFrequency ? entry.second.read_count : entry.second.last_used;
        const uint64_t recency = entry.second.last_used;
        if (best_group < 0 || key < best_key ||
            (key == best_key && (recency < best_recency ||
                                 (recency == best_recency && entry.first < best_group)))) {
            best_key = key;
            best_recency = recency;
            best_group = entry.first;
        }
    }
    decision.victim = best_group;
    decision.admit = best_group >= 0;
    return decision;
}

// ---- the matched control ----

namespace {

/// The deterministic access pattern: every eighth step reads the next group of the hot prefix, and the other
/// steps scan the working set. It is the shape a long attention session has: a small hot region and a mostly
/// sequential scan.
///
/// The hot region is indexed by the HOT-ACCESS COUNT, not by the raw step. Indexing it by the step would make
/// every hot access a multiple of gcd(8, hot_prefix), so a hot prefix of 4 would read only group 0 and the
/// control would model one hot group rather than a hot region.
int32_t pattern_group(uint64_t step, int32_t working_set, int32_t hot_prefix) {
    if (hot_prefix > 0 && (step % 8) == 0) return (int32_t) ((step / 8) % (uint64_t) hot_prefix);
    return (int32_t) (step % (uint64_t) working_set);
}

}  // namespace

KvPlacementControlResult kv_placement_control(KvPlacementPolicy policy, int32_t cache_groups, int32_t working_set,
                                               int32_t hot_prefix, uint64_t steps) {
    KvPlacementControlResult result;
    result.policy = policy;
    if (cache_groups <= 0 || working_set <= 0) return result;

    KvPlacementPolicyState state(policy);
    // The cache holds the first `cache_groups` groups, which is the state after the working set is first scanned.
    for (int32_t g = 0; g < cache_groups && g < working_set; ++g) state.admit(g, 0, false);

    uint64_t now = 1;
    for (uint64_t step = 0; step < steps; ++step) {
        const int32_t group = pattern_group(step, working_set, hot_prefix);
        ++now;
        if (state.size() == 0) {
            ++result.misses;
            ++result.cold_reads;
            state.admit(group, now, false);
            continue;
        }
        if (state.holds(group)) {
            ++result.hits;
            state.touch(group, now);
            continue;
        }
        ++result.misses;
        ++result.cold_reads;
        const KvPlacementDecision decision = state.victim();
        if (!decision.admit) {
            state.admit(group, now, false);
            continue;
        }
        ++result.evictions;
        state.remove(decision.victim);
        state.admit(group, now, false);
    }
    return result;
}

KvPlacementComparison kv_placement_compare(KvPlacementPolicy baseline, KvPlacementPolicy candidate, int32_t cache_groups,
                                           int32_t working_set, int32_t hot_prefix, uint64_t steps) {
    KvPlacementComparison comparison;
    comparison.baseline = kv_placement_control(baseline, cache_groups, working_set, hot_prefix, steps);
    comparison.candidate = kv_placement_control(candidate, cache_groups, working_set, hot_prefix, steps);
    comparison.hit_difference =
        (int64_t) comparison.candidate.hits - (int64_t) comparison.baseline.hits;
    comparison.candidate_wins = comparison.hit_difference > 0;
    return comparison;
}

}  // namespace strata::core
