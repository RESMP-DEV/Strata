// src/core/kv_tier_session.cpp - see include/strata/core/kv_tier_session.hpp.
#include "strata/core/kv_tier_session.hpp"

#include "strata/core/kv_tier_hooks.hpp"

#include <cstdio>
#include <mutex>

namespace strata::core {

namespace {

std::mutex g_mutex;
KvTierGraphPhases g_phases;
int g_window_stage = 0;   // 0: nothing, 1: selection seen, 2: readiness seen

}  // namespace

KvTierSession& KvTierSession::instance() {
    static KvTierSession session;
    return session;
}

bool KvTierSession::enable(const KvTierSessionConfig& config, std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    KvTierSession& session = instance();
    if (session.enabled_) {
        error = "kv tier: the cold tier is already enabled";
        return false;
    }
    if (config.store_path.empty()) {
        error = "kv tier: the cold tier needs a store path";
        return false;
    }
    if (config.l1_groups_per_layer <= 0) {
        error = "kv tier: the cold tier needs at least one L1 group per layer";
        return false;
    }
    if (config.geometry_id == 0 || config.format_id == 0) {
        error = "kv tier: the cold tier needs a geometry and a format identity";
        return false;
    }
    session.config_ = config;

    // The L2 store is a file: it holds whole groups, commits a group's valid marker only after its payload is
    // durable, and refuses a key of another geometry, format, layer, or epoch.
    KvFileStore::Identity identity;
    // The store's session identity is the SESSION's, not the geometry's: the coordinator's scope carries the same
    // value, so the key the coordinator writes is the key the store accepts. A store whose identity is a
    // different number from the coordinator's scope refuses every publish.
    identity.session_id = config.session_id;
    identity.geometry_id = config.geometry_id;
    identity.format_id = config.format_id;
    identity.layer = config.layer;
    identity.blocks_per_group = config.blocks_per_group;
    uint64_t block_bytes = 0;
    for (const auto& run : config.runs) block_bytes += run.bytes;
    identity.group_bytes = block_bytes * (uint64_t) config.blocks_per_group;
    // The directory is a FIXED area, so the store must be told how many groups it may ever hold. It is sized from
    // the context: the number of blocks the layer can hold, over the group's blocks.
    identity.max_groups = config.max_groups;
    if (!session.store_.open(config.store_path, identity, config.runs, error)) return false;
    session.store_open_ = true;
    session.enabled_ = true;
    return true;
}

bool KvTierSession::enabled() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return instance().enabled_;
}

KvTierCoordinator* KvTierSession::coordinator_for(int32_t layer) {
    KvTierSession& session = instance();
    // One coordinator per LAYER: the layers have different stores and different geometry identities, so a layer
    // must never share a coordinator with another one.
    for (auto& coordinator : session.coordinators_) {
        if (coordinator->scope().stage == 0 && coordinator->scope().slot == 0 &&
            coordinator->scope().session_id == session.config_.session_id &&
            coordinator->layer() == layer)
            return coordinator.get();
    }
    KvTierCoordinator::Options options;
    options.layer = layer;
    options.l1_groups = session.config_.l1_groups_per_layer;
    options.geometry_id = session.config_.geometry_id;
    options.format_id = session.config_.format_id;
    options.blocks_per_group = session.config_.blocks_per_group;
    options.block_bytes = session.config_.block_bytes;
    options.runs = session.config_.runs;
    session.coordinators_.push_back(std::make_unique<KvTierCoordinator>(
        KvCoordinatorScope{session.config_.session_id, 0, 0}, options, &session.store_));
    return session.coordinators_.back().get();
}

KvTierCoordinator* KvTierSession::coordinator(int32_t layer) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!instance().enabled_) return nullptr;
    return instance().coordinator_for(layer);
}

KvTierCounters KvTierSession::counters() {
    std::lock_guard<std::mutex> lock(g_mutex);
    KvTierCounters total;
    for (const auto& coordinator : instance().coordinators_) {
        const KvTierCounters& c = coordinator->counters();
        total.l1_hits += c.l1_hits;
        total.l1_misses += c.l1_misses;
        total.l1_evictions += c.l1_evictions;
        total.l2_reads += c.l2_reads;
        total.l2_read_failures += c.l2_read_failures;
        total.l2_write_bytes += c.l2_write_bytes;
        total.l2_read_bytes += c.l2_read_bytes;
        total.l3_reads += c.l3_reads;
        total.l3_write_bytes += c.l3_write_bytes;
        total.published_groups += c.published_groups;
        total.checksum_errors += c.checksum_errors;
        total.readiness_waits += c.readiness_waits;
    }
    return total;
}

KvTierGraphPhases KvTierSession::graph_phases() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_phases;
}

void KvTierSession::note_phase(int phase) {
    std::lock_guard<std::mutex> lock(g_mutex);
    switch (phase) {
        case kKvTierPhaseSelection:
            ++g_phases.selection;
            g_window_stage = 1;
            break;
        case kKvTierPhaseReadiness:
            if (g_window_stage < 1) ++g_phases.split_violations;   // readiness without a selection
            ++g_phases.readiness;
            g_window_stage = 2;
            break;
        case kKvTierPhaseResolve:
            if (g_window_stage < 2) ++g_phases.split_violations;   // a resolve with no host readiness
            ++g_phases.resolve;
            if (g_window_stage >= 2) ++g_phases.windows;
            g_window_stage = 0;
            break;
        default:
            break;
    }
}

void KvTierSession::report(const char* tag) {
    const KvTierCounters c = counters();
    const KvTierGraphPhases phases = graph_phases();
    const KvReadBatchStats batches = kv_tier_batch_stats();
    std::fprintf(stderr,
                 "strata kv tier (%s): l1_hits=%llu l1_misses=%llu l1_evictions=%llu l2_reads=%llu "
                 "l2_read_failures=%llu l2_read_bytes=%llu l2_write_bytes=%llu published_groups=%llu "
                 "checksum_errors=%llu readiness_waits=%llu | graph_split: selection=%llu readiness=%llu "
                 "resolve=%llu windows=%llu violations=%llu | batches: batches=%llu groups=%llu requested=%llu "
                 "foreground=%llu\n",
                 tag ? tag : "report", (unsigned long long) c.l1_hits, (unsigned long long) c.l1_misses,
                 (unsigned long long) c.l1_evictions, (unsigned long long) c.l2_reads,
                 (unsigned long long) c.l2_read_failures, (unsigned long long) c.l2_read_bytes,
                 (unsigned long long) c.l2_write_bytes, (unsigned long long) c.published_groups,
                 (unsigned long long) c.checksum_errors, (unsigned long long) c.readiness_waits,
                 (unsigned long long) phases.selection, (unsigned long long) phases.readiness,
                 (unsigned long long) phases.resolve, (unsigned long long) phases.windows,
                 (unsigned long long) phases.split_violations, (unsigned long long) batches.batches,
                 (unsigned long long) batches.groups, (unsigned long long) batches.requested,
                 (unsigned long long) batches.foreground);
}

void KvTierSession::reset_counters() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto& coordinator : instance().coordinators_) coordinator->reset_counters();
}

bool KvTierSession::reset(std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    KvTierSession& session = instance();
    bool ok = true;
    for (auto& coordinator : session.coordinators_) {
        // The first failure's message is the one the caller sees; the rest still run, so one coordinator's
        // problem does not leave the others holding an old epoch.
        std::string one;
        if (!coordinator->reset(one)) {
            ok = false;
            if (error.empty()) error = one;
        }
    }
    return ok;
}

void KvTierSession::shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    KvTierSession& session = instance();
    session.coordinators_.clear();
    if (session.store_open_) session.store_.close();
    session.store_open_ = false;
    session.enabled_ = false;
}

}  // namespace strata::core
