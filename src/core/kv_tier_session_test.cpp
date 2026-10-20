// src/core/kv_tier_session_test.cpp - the session's ownership of the coordinators and the tier's opt-in contract.
#include "strata/core/kv_tier_session.hpp"

#include <cstdio>
#include <string>

using namespace strata::core;

int main() {
    // The tier is off by default: the serving path's branch must cost nothing when it is not used.
    if (KvTierSession::enabled()) return 1;
    if (KvTierSession::coordinator(0) != nullptr) return 2;

    KvTierSessionConfig config;
    config.store_path = "/tmp/strata-kv-tier-test";
    config.session_id = 1;
    config.layer = 0;
    config.l1_groups_per_layer = 2;
    config.max_groups = 16;
    config.geometry_id = 0xabc;
    config.format_id = 0xdef;
    config.blocks_per_group = 4;
    config.block_bytes = 64;
    config.runs = {KvBlockRun{32}, KvBlockRun{32}};

    std::string error;
    // A configuration without a store path or a geometry is refused.
    KvTierSessionConfig bad = config;
    bad.store_path.clear();
    if (KvTierSession::enable(bad, error)) return 3;
    bad = config;
    bad.geometry_id = 0;
    if (KvTierSession::enable(bad, error)) return 4;
    bad = config;
    bad.l1_groups_per_layer = 0;
    if (KvTierSession::enable(bad, error)) return 5;

    if (!KvTierSession::enable(config, error)) { std::printf("enable failed: %s\n", error.c_str()); return 6; }
    if (!KvTierSession::enabled()) return 7;
    // A second enable is refused: the session owns one configuration.
    if (KvTierSession::enable(config, error)) return 8;

    // The coordinators are per layer, and two layers never share state.
    KvTierCoordinator* a = KvTierSession::coordinator(0);
    KvTierCoordinator* b = KvTierSession::coordinator(1);
    if (a == nullptr || b == nullptr) return 9;
    if (a == b) return 10;
    if (KvTierSession::coordinator(0) != a) return 11;   // the same layer gives the same coordinator

    // The counters are summed over the coordinators and start empty.
    const KvTierCounters counters = KvTierSession::counters();
    if (counters.l1_hits != 0 || counters.l2_reads != 0) return 12;

    // A reset advances the epoch of every coordinator.
    if (a->epoch() != 0) return 13;
    if (!KvTierSession::reset(error)) return 14;
    if (a->epoch() != 1) return 15;

    KvTierSession::shutdown();
    if (KvTierSession::enabled()) return 16;

    std::puts("kv tier session: OK");
    return 0;
}
