// src/core/kv_placement_test.cpp - the placement policy's rules and the matched control that a promotion needs.
#include "strata/core/kv_placement.hpp"

#include <cstdio>
#include <string>

using namespace strata::core;

int main() {
    // 1. The policy's name is stable, because the report names it.
    {
        KvPlacementPolicyState clock(KvPlacementPolicy::kClock);
        KvPlacementPolicyState frequency(KvPlacementPolicy::kFrequency);
        if (std::string(clock.name()) != "clock") return 1;
        if (std::string(frequency.name()) != "frequency") return 2;
    }

    // 2. A pinned group is never a victim.
    {
        KvPlacementPolicyState state(KvPlacementPolicy::kClock);
        state.admit(1, 10, true);    // the open tail
        state.admit(2, 20, false);
        const KvPlacementDecision decision = state.victim();
        if (decision.victim != 2) return 10;
        if (!decision.admit) return 11;
    }

    // 3. CLOCK evicts the least recently used group.
    {
        KvPlacementPolicyState state(KvPlacementPolicy::kClock);
        state.admit(1, 10, false);
        state.admit(2, 20, false);
        state.admit(3, 30, false);
        state.touch(1, 40);
        const KvPlacementDecision decision = state.victim();
        if (decision.victim != 2) return 20;   // group 1 was used more recently
    }

    // 4. The frequency policy evicts the least frequently read group.
    {
        KvPlacementPolicyState state(KvPlacementPolicy::kFrequency);
        state.admit(1, 10, false);
        state.admit(2, 20, false);
        state.touch(1, 30);
        state.touch(1, 40);
        const KvPlacementDecision decision = state.victim();
        if (decision.victim != 2) return 30;   // group 2 was never read
    }

    // 5. The control is deterministic: the same policy and the same pattern give the same numbers.
    {
        const KvPlacementControlResult one =
            kv_placement_control(KvPlacementPolicy::kClock, 16, 64, 4, 1000);
        const KvPlacementControlResult two =
            kv_placement_control(KvPlacementPolicy::kClock, 16, 64, 4, 1000);
        if (one.hits != two.hits || one.misses != two.misses) return 40;
        if (one.evictions != two.evictions) return 41;
        if (one.hits + one.misses != 1000) return 42;
    }

    // 6. The comparison reports the difference, and a policy that does not beat the baseline does not win.
    {
        const KvPlacementComparison comparison =
            kv_placement_compare(KvPlacementPolicy::kClock, KvPlacementPolicy::kFrequency, 16, 64, 4, 1000);
        if (comparison.hit_difference !=
            (int64_t) comparison.candidate.hits - (int64_t) comparison.baseline.hits) return 50;
        if (comparison.candidate_wins != (comparison.hit_difference > 0)) return 51;

        // The same policy against itself is a tie, never a win: a control that "wins" against itself is a broken
        // control.
        const KvPlacementComparison tie =
            kv_placement_compare(KvPlacementPolicy::kClock, KvPlacementPolicy::kClock, 16, 64, 4, 1000);
        if (tie.hit_difference != 0) return 52;
        if (tie.candidate_wins) return 53;
    }

    // 7. A cache that holds the whole working set has no misses at all.
    {
        const KvPlacementControlResult result =
            kv_placement_control(KvPlacementPolicy::kClock, 64, 64, 4, 500);
        if (result.misses != 0) return 60;
        if (result.hits != 500) return 61;
        if (result.cold_reads != 0) return 62;
    }

    std::puts("kv placement: OK");
    return 0;
}
