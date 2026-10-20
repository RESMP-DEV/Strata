// include/strata/core/kv_placement.hpp - the L1 admission policy, and the control that must beat CLOCK.
//
// The admission policy decides which groups stay in the L1 cache. CLOCK is the baseline: the engine's own device
// residency map uses it, and matching it keeps the cold tier comparable with the tier the engine already has.
//
// A replacement policy is only promoted from a MATCHED CONTROL: the same workload, the same cache size, the same
// store, and the same read pattern, with only the policy differing. The control harness below is what makes that
// comparison possible without a GPU, so a policy can be measured before it is allowed near the serving path.
#pragma once

#include "strata/core/kv_coordinator.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// What the policy knows about a group.
struct KvPlacementFacts {
    int32_t layer = 0;
    int32_t group = 0;
    uint64_t last_used = 0;     ///< the sequence number of the last read or write
    uint64_t read_count = 0;    ///< how many times the group has been read
    bool pinned = false;        ///< the open tail is pinned and never a victim
};

/// The result of asking the policy for a victim.
struct KvPlacementDecision {
    int32_t victim = -1;        ///< the group to evict, or -1 when the policy has no opinion
    bool admit = false;         ///< should the incoming group be admitted at all
};

/// The admission policies.
enum class KvPlacementPolicy : int {
    kClock = 0,        ///< the baseline: the least recently used unpinned group
    kFrequency = 1,    ///< the least frequently used unpinned group
    kScan = 2,         ///< the least recently used, but the scan is a full pass over the cache
};

/// One policy's state over the cache's groups.
class KvPlacementPolicyState {
public:
    explicit KvPlacementPolicyState(KvPlacementPolicy policy) : policy_(policy) {}

    /// The policy's name, for the report.
    const char* name() const;

    /// Note a group the cache holds.
    void admit(int32_t group, uint64_t now, bool pinned);

    /// Note a group that was read.
    void touch(int32_t group, uint64_t now);

    /// Note a group that was evicted.
    void remove(int32_t group);

    /// Ask the policy for a victim among the groups that are not pinned.
    KvPlacementDecision victim() const;

    /// Does the policy hold the group? The control asks before it counts a hit.
    bool holds(int32_t group) const { return facts_.find(group) != facts_.end(); }

    /// The number of groups the policy holds.
    size_t size() const { return facts_.size(); }

private:
    struct Entry {
        uint64_t last_used = 0;
        uint64_t read_count = 0;
        bool pinned = false;
    };

    KvPlacementPolicy policy_;
    std::map<int32_t, Entry> facts_;
};

/// The result of a matched control: the same workload, the same cache size, and one policy.
struct KvPlacementControlResult {
    KvPlacementPolicy policy = KvPlacementPolicy::kClock;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t cold_reads = 0;    ///< the reads that had to go to the store

    /// The hit rate, as a fraction.
    double hit_rate() const {
        const uint64_t total = hits + misses;
        return total == 0 ? 0.0 : (double) hits / (double) total;
    }
};

/// Run the control: a deterministic access pattern over a cache of `cache_groups`, with one policy.
///
/// The pattern is a scan of `working_set` groups plus a hot prefix, which is the shape a long attention session
/// has: a mostly sequential scan of the context with a small hot region at the end.
KvPlacementControlResult kv_placement_control(KvPlacementPolicy policy, int32_t cache_groups, int32_t working_set,
                                               int32_t hot_prefix, uint64_t steps);

/// Compare two policies in the same control. The result reports the difference in hits, which is the evidence a
/// promotion needs: a policy that does not beat CLOCK is not promoted.
struct KvPlacementComparison {
    KvPlacementControlResult baseline;
    KvPlacementControlResult candidate;
    int64_t hit_difference = 0;   ///< the candidate's hits minus the baseline's
    bool candidate_wins = false;
};

KvPlacementComparison kv_placement_compare(KvPlacementPolicy baseline, KvPlacementPolicy candidate, int32_t cache_groups,
                                           int32_t working_set, int32_t hot_prefix, uint64_t steps);

}  // namespace strata::core
