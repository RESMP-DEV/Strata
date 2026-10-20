// src/core/kv_tier_session.cpp - see include/strata/core/kv_tier_session.hpp.
#include "strata/core/kv_tier_session.hpp"

#include "strata/core/kv_tier_hooks.hpp"

#include <cstdio>
#include <iomanip>
#include <mutex>
#include <sstream>

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

KvTierSession::~KvTierSession() {
    if (!enabled_) return;
    std::string error;
    if (!flush(error))
        std::fprintf(stderr, "strata kv tier (shutdown): background I/O failed: %s\n", error.c_str());
    report("shutdown");
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

    session.enabled_ = true;
    return true;
}

bool KvTierSession::enabled() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return instance().enabled_;
}

KvTierCoordinator* KvTierSession::coordinator_for(int32_t layer) {
    KvTierSession& session = instance();
    // One coordinator and one fixed-layout file per LAYER. A file store's directory is indexed by group, so
    // mixing layers would make the file grow without bound and make one layer's directory capacity a lie.
    for (auto& coordinator : session.coordinators_) {
        if (coordinator->scope().stage == 0 && coordinator->scope().slot == 0 &&
            coordinator->scope().session_id == session.config_.session_id &&
            coordinator->layer() == layer)
            return coordinator.get();
    }
    uint64_t block_bytes = 0;
    for (const auto& run : session.config_.runs) block_bytes += run.bytes;
    KvFileStore::Identity identity;
    identity.session_id = session.config_.session_id;
    identity.geometry_id = session.config_.geometry_id;
    identity.format_id = session.config_.format_id;
    identity.layer = layer;
    identity.blocks_per_group = session.config_.blocks_per_group;
    identity.group_bytes = block_bytes * (uint64_t) session.config_.blocks_per_group;
    identity.max_groups = session.config_.max_groups;
    std::ostringstream path;
    path << session.config_.store_path << "/layer-" << std::setfill('0') << std::setw(4) << layer;
    auto store = std::make_unique<KvFileStore>();
    std::string error;
    if (!store->open(path.str(), identity, session.config_.runs, error)) {
        session.last_error_ = error;
        return nullptr;
    }
    auto [store_it, inserted] = session.stores_.emplace(layer, std::move(store));
    if (!inserted) return nullptr;

    KvTierCoordinator::Options options;
    options.layer = layer;
    options.l1_groups = session.config_.l1_groups_per_layer;
    options.geometry_id = session.config_.geometry_id;
    options.format_id = session.config_.format_id;
    options.blocks_per_group = session.config_.blocks_per_group;
    options.block_bytes = session.config_.block_bytes;
    options.runs = session.config_.runs;
    options.async_io = session.config_.async_io;
    session.coordinators_.push_back(std::make_unique<KvTierCoordinator>(
        KvCoordinatorScope{session.config_.session_id, 0, 0}, options, store_it->second.get()));
    return session.coordinators_.back().get();
}

KvTierCoordinator* KvTierSession::coordinator(int32_t layer) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!instance().enabled_) return nullptr;
    KvTierSession& session = instance();
    KvTierCoordinator* coordinator = session.coordinator_for(layer);
    if (coordinator == nullptr) std::fprintf(stderr, "kv tier: coordinator %d failed: %s\n", layer, session.last_error_.c_str());
    return coordinator;
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
        total.engine_host_restores += c.engine_host_restores;
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
                "checksum_errors=%llu readiness_waits=%llu engine_host_restores=%llu | graph_split: selection=%llu readiness=%llu "
                 "resolve=%llu windows=%llu violations=%llu | batches: batches=%llu groups=%llu requested=%llu "
                 "foreground=%llu\n",
                 tag ? tag : "report", (unsigned long long) c.l1_hits, (unsigned long long) c.l1_misses,
                 (unsigned long long) c.l1_evictions, (unsigned long long) c.l2_reads,
                 (unsigned long long) c.l2_read_failures, (unsigned long long) c.l2_read_bytes,
                 (unsigned long long) c.l2_write_bytes, (unsigned long long) c.published_groups,
                 (unsigned long long) c.checksum_errors, (unsigned long long) c.readiness_waits,
                (unsigned long long) c.engine_host_restores,
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

bool KvTierSession::flush(std::string& error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    KvTierSession& session = instance();
    bool ok = true;
    for (const auto& coordinator : session.coordinators_) {
        std::string one;
        if (!coordinator->flush(one)) {
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
    session.stores_.clear();
    session.enabled_ = false;
}

}  // namespace strata::core
