// include/strata/core/kv_tier_session.hpp - the serving-path owner of the disaggregated KV tiers.
//
// The engine's KV streaming (kv_stream.hpp) already gives a streamed QSA layer an L0 VRAM pool of page slots and
// an L1 pinned host copy of the authoritative K/V, with a device-side CLOCK resolver. The coordinator adds the part
// that engine does not have: a cold tier (L2) that a group can be published to and read back from, the epoch
// rules that keep a rewind from reading stale bytes, and the counters that prove a cold read happened.
//
// This session owns the coordinators: one per session, stage, and batch slot, so the same layer ordinal in two
// stages never shares state.
#pragma once

#include "strata/core/kv_coordinator.hpp"
#include "strata/core/kv_file_store.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

/// The configuration of the cold tier. The tier is opt-in: with no store, the session is inert and the engine's
/// own streaming path is untouched.
struct KvTierSessionConfig {
    std::string store_path;         ///< the local store's path; empty means no cold tier
    int32_t layer = 0;              ///< the QSA layer the store holds
    int32_t l1_groups_per_layer = 0;///< how many groups the L1 cache holds per layer; 0 means no L1 cache
    uint64_t geometry_id = 0;
    uint64_t format_id = 0;
    int32_t blocks_per_group = kKvGroupBlocks;
    uint64_t block_bytes = 0;
    std::vector<KvBlockRun> runs;
};

/// The session's own state. It is a singleton because the engine's serving path has one session per process, and
/// a global keeps the verify path's call sites free of plumbing.
class KvTierSession {
public:
    /// Enable the cold tier. The call is idempotent, and a second call with a different path is refused.
    static bool enable(const KvTierSessionConfig& config, std::string& error);

    /// Is the cold tier enabled?
    static bool enabled();

    /// The coordinator for a layer, or null when the tier is off. The coordinator is created on first use, so a
    /// layer that never runs costs nothing.
    static KvTierCoordinator* coordinator(int32_t layer);

    /// The counters, summed over every coordinator the session created.
    static KvTierCounters counters();

    /// One line on stderr with the counters. A run that claims to have used the cold tier must show reads here.
    static void report(const char* tag);

    /// Zero every coordinator's counters, so the next report describes ONE sequence.
    static void reset_counters();

    /// Forget the session's state: the cold tier's groups are invalidated and the epoch advances.
    static bool reset(std::string& error);

    /// Disable the tier and release the session's state.
    static void shutdown();

private:
    static KvTierSession& instance();
    KvTierCoordinator* coordinator_for(int32_t layer);
    KvTierSessionConfig config_;
    KvFileStore store_;              ///< the L2 local file store
    bool store_open_ = false;
    std::vector<std::unique_ptr<KvTierCoordinator>> coordinators_;
    bool enabled_ = false;
};

}  // namespace strata::core
