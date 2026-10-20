// src/core/kv_tier_session.cpp - see include/strata/core/kv_tier_session.hpp.
#include "strata/core/kv_tier_session.hpp"

#include <cstdio>
#include <mutex>

namespace strata::core {

namespace {

std::mutex g_mutex;

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
    session.store_ = new KvMemoryStore();   // the local file store arrives with the L2 adapter
    session.enabled_ = true;
    return true;
}

bool KvTierSession::enabled() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return instance().enabled_;
}

KvTierCoordinator* KvTierSession::coordinator_for(int32_t layer) {
    KvTierSession& session = instance();
    for (auto& coordinator : session.coordinators_) {
        if (coordinator->scope().stage == 0 && coordinator->scope().slot == 0 &&
            coordinator->scope().session_id == (uint64_t) layer)
            return coordinator.get();
    }
    KvTierCoordinator::Options options;
    options.l1_groups = session.config_.l1_groups_per_layer;
    options.geometry_id = session.config_.geometry_id;
    options.format_id = session.config_.format_id;
    options.blocks_per_group = session.config_.blocks_per_group;
    options.block_bytes = session.config_.block_bytes;
    options.runs = session.config_.runs;
    session.coordinators_.push_back(std::make_unique<KvTierCoordinator>(
        KvCoordinatorScope{(uint64_t) layer, 0, 0}, options, session.store_));
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

void KvTierSession::report(const char* tag) {
    const KvTierCounters c = counters();
    std::fprintf(stderr,
                 "strata kv tier (%s): l1_hits=%llu l1_misses=%llu l1_evictions=%llu l2_reads=%llu "
                 "l2_read_failures=%llu l2_read_bytes=%llu l2_write_bytes=%llu published_groups=%llu "
                 "checksum_errors=%llu readiness_waits=%llu\n",
                 tag ? tag : "report", (unsigned long long) c.l1_hits, (unsigned long long) c.l1_misses,
                 (unsigned long long) c.l1_evictions, (unsigned long long) c.l2_reads,
                 (unsigned long long) c.l2_read_failures, (unsigned long long) c.l2_read_bytes,
                 (unsigned long long) c.l2_write_bytes, (unsigned long long) c.published_groups,
                 (unsigned long long) c.checksum_errors, (unsigned long long) c.readiness_waits);
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
    delete session.store_;
    session.store_ = nullptr;
    session.enabled_ = false;
}

}  // namespace strata::core
