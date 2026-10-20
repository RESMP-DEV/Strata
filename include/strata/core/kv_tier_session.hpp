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
    uint64_t session_id = 0;        ///< the session's identity; the store and the coordinators share it
    int32_t layer = 0;              ///< the QSA layer the store holds
    int32_t l1_groups_per_layer = 0;///< how many groups the L1 cache holds per layer; 0 means no L1 cache
    uint64_t geometry_id = 0;
    uint64_t format_id = 0;
    int32_t blocks_per_group = kKvGroupBlocks;
    uint64_t block_bytes = 0;
    uint64_t max_groups = 0;    ///< how many groups the store's directory may hold; 0 refuses every write
    std::vector<KvBlockRun> runs;
};

/// The graph-split proof. A CUDA graph cannot perform host-side file I/O or update a host cache from a CPU
/// thread, so the captured sequence must split at the selection boundary: the captured selection produces the read
/// plan, the coordinator makes the L1 slots ready on the host, and the captured resolve consumes only ready slots.
///
/// The proof counts the phases the captured windows actually went through. A run that claims the cold tier is in
/// the serving path must show a selection phase, a host readiness phase, and a resolve phase, in that order, and
/// no split violation.
struct KvTierGraphPhases {
    uint64_t selection = 0;   ///< the captured selection produced a read plan
    uint64_t readiness = 0;   ///< the host made the plan's groups ready, between the two graph phases
    uint64_t resolve = 0;     ///< the captured resolve consumed the ready groups
    uint64_t windows = 0;     ///< the windows that went through all three
    uint64_t split_violations = 0;  ///< a resolve without a preceding host readiness: a graph doing host work
};

/// The phases, in the order the hooks call them.
enum KvTierPhase : int {
    kKvTierPhaseSelection = 0,
    kKvTierPhaseReadiness = 1,
    kKvTierPhaseResolve = 2,
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

    /// The graph-split proof's counters.
    static KvTierGraphPhases graph_phases();

    /// Note a phase of the captured window. The order is what the proof checks: a resolve that is not preceded by
    /// a host readiness in the same window is a split violation, because that means the graph tried to do host
    /// work.
    static void note_phase(int phase);

    /// One line on stderr with the counters. A run that claims to have used the cold tier must show reads here.
    /// The line also carries the graph-split proof: the number of windows that went through a captured selection,
    /// a host readiness phase, and a captured resolve, and the number of split violations.
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
