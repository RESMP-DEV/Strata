// src/core/kv_coordinator.cpp - see include/strata/core/kv_coordinator.hpp.
#include "strata/core/kv_coordinator.hpp"

#include <cstring>
#include <sstream>

namespace strata::core {

namespace {

}  // namespace

std::string key_string(const KvBlockKey& key) {
    std::ostringstream out;
    out << key.session_id << '/' << key.geometry_id << '/' << key.format_id << '/' << key.layer << '/'
        << key.group << '/' << key.epoch;
    return out.str();
}

namespace {

uint64_t fnv1a(const void* data, size_t bytes);

uint64_t fnv1a(const void* data, size_t bytes) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

}  // namespace

// ---- KvMemoryStore ----

bool KvMemoryStore::contains(const KvBlockKey& key) const {
    const auto it = groups_.find(key_string(key));
    return it != groups_.end() && !it->second.corrupt;
}

bool KvMemoryStore::get(const KvBlockRequest& request, std::string& error) {
    ++reads_;
    const auto it = groups_.find(key_string(request.key));
    if (it == groups_.end()) {
        error = "kv store: the group is absent";
        return false;
    }
    if (it->second.corrupt || (has_corrupt_ && corrupt_ == request.key)) {
        error = "kv store: the group's checksum does not match";
        return false;
    }
    // The store verifies its own payload: a group whose bytes do not match the checksum it was written with is
    // refused, never returned as a zero fill or as the bytes the caller happens to hold.
    {
        const uint64_t actual = fnv1a(it->second.payload.data(), it->second.payload.size());
        if (actual != it->second.checksum) {
            error = "kv store: the group's checksum does not match";
            return false;
        }
    }
    if (request.dst == nullptr || request.bytes == 0) {
        error = "kv store: the request has no destination";
        return false;
    }
    if (request.offset + request.bytes > it->second.payload.size()) {
        error = "kv store: the request runs past the group's payload";
        return false;
    }
    std::memcpy(request.dst, it->second.payload.data() + request.offset, request.bytes);
    return true;
}

bool KvMemoryStore::put(const KvBlockPut& put_request, std::string& error) {
    if (put_request.src == nullptr || put_request.bytes == 0) {
        error = "kv store: the put has no payload";
        return false;
    }
    ++writes_;
    Entry entry;
    entry.payload.assign(static_cast<const uint8_t*>(put_request.src),
                         static_cast<const uint8_t*>(put_request.src) + put_request.bytes);
    entry.checksum = put_request.checksum;
    entry.corrupt = (has_corrupt_ && corrupt_ == put_request.key);
    groups_[key_string(put_request.key)] = std::move(entry);
    return true;
}

bool KvMemoryStore::invalidate(const KvBlockKey& key, std::string& error) {
    const auto it = groups_.find(key_string(key));
    if (it == groups_.end()) {
        error = "kv store: the group is absent";
        return false;
    }
    groups_.erase(it);
    return true;
}

// ---- KvTierCoordinator ----

KvTierCoordinator::KvTierCoordinator(KvCoordinatorScope scope, Options options, KvBlockStore* store)
    : scope_(scope), options_(std::move(options)), store_(store) {
    // The coordinator's layer is its own: the scope names the session, the stage, and the slot, and the layer is
    // a separate dimension because two layers of the same session never share a coordinator.
    layer_ = options_.layer;
    if (options_.l1_groups < 0) options_.l1_groups = 0;
    if (options_.blocks_per_group <= 0) options_.blocks_per_group = kKvGroupBlocks;
    slots_.resize((size_t) options_.l1_groups);
    slot_bytes_.resize((size_t) options_.l1_groups * (size_t) group_bytes());
}

KvTierCoordinator::~KvTierCoordinator() = default;

uint64_t KvTierCoordinator::block_bytes() const {
    uint64_t block = 0;
    for (const auto& run : options_.runs) block += run.bytes;
    if (block == 0) block = options_.block_bytes;
    return block;
}

uint64_t KvTierCoordinator::group_bytes() const {
    return block_bytes() * (uint64_t) options_.blocks_per_group;
}

int32_t KvTierCoordinator::slot_index(int32_t layer, int32_t group) const {
    // A slot is a flat LRU over groups; the layer and group identify what the slot holds.
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].group >= 0 && slots_[i].layer == layer && slots_[i].group == group &&
            slots_[i].epoch == epoch_)
            return (int32_t) i;
    }
    return -1;
}

int32_t KvTierCoordinator::find_group(int32_t layer, int32_t group) const { return slot_index(layer, group); }

int32_t KvTierCoordinator::victim_slot(std::string& error) const {
    // CLOCK: the least recently used unpinned slot. A pinned slot is the open tail and is never a victim.
    int32_t best = -1;
    uint64_t best_stamp = UINT64_MAX;
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].pins > 0) continue;
        if (slots_[i].group < 0) return (int32_t) i;   // a free slot
        if (slots_[i].stamp < best_stamp) {
            best_stamp = slots_[i].stamp;
            best = (int32_t) i;
        }
    }
    if (best < 0) {
        error = "kv coordinator: every L1 slot is pinned";
        return -1;
    }
    return best;
}

void KvTierCoordinator::touch(int32_t slot) {
    if (slot < 0 || (size_t) slot >= slots_.size()) return;
    slots_[(size_t) slot].stamp = ++stamp_;
}

uint64_t KvTierCoordinator::group_checksum(int32_t layer, int32_t group) const {
    const int32_t slot = slot_index(layer, group);
    if (slot < 0) return 0;
    const uint64_t bytes = group_bytes();
    if (bytes == 0) return 0;
    return fnv1a(slot_bytes_.data() + (size_t) slot * (size_t) bytes, (size_t) bytes);
}

bool KvTierCoordinator::rewind(std::string& error) {
    (void) error;
    if (store_ != nullptr) {
        // The store's groups of the old epoch are invalidated BEFORE the epoch advances, so a stale group can
        // never be read as current data and the store's quota is not consumed by an epoch nobody may read.
        for (const KvBlockKey& key : published_) {
            std::string ignored;
            (void) store_->invalidate(key, ignored);
        }
    }
    published_.clear();
    published_checksums_.clear();
    ++epoch_;
    for (auto& slot : slots_) {
        slot.group = -1;
        slot.layer = -1;
        slot.epoch = 0;
        slot.stamp = 0;
        slot.pins = 0;
    }
    completed_.clear();
    stamp_ = 0;
    return true;
}

bool KvTierCoordinator::reset(std::string& error) {
    if (store_ != nullptr) {
        // The store's groups of the current epoch are invalidated before the epoch advances, so a stale group can
        // never be read as current data. A group that is already gone is not an error: the reset still succeeds.
        for (const KvBlockKey& key : published_) {
            std::string ignored;
            store_->invalidate(key, ignored);
        }
    }
    const bool ok = rewind(error);
    completed_.clear();
    return ok;
}

bool KvTierCoordinator::append(int32_t layer, int32_t block, std::string& error) {
    if (store_ == nullptr) {
        error = "kv coordinator: no store is attached";
        return false;
    }
    const int32_t group = block / options_.blocks_per_group;
    const int32_t slot = slot_index(layer, group);
    if (slot < 0) {
        // The open tail must live in L1: it is pinned until the group is published.
        const int32_t victim = victim_slot(error);
        if (victim < 0) return false;
        KvL1Slot& entry = slots_[(size_t) victim];
        if (entry.group >= 0) ++counters_.l1_evictions;
        entry.group = group;
        entry.layer = layer;
        entry.epoch = epoch_;
        entry.pins = 1;
        touch(victim);
    } else {
        slots_[(size_t) slot].pins = 1;
        touch(slot);
    }
    auto& bits = completed_[layer];
    const size_t need = (size_t) block + 1;
    if (bits.size() < need) bits.resize(need, 0);
    bits[(size_t) block] = 0;   // the block is open again
    return true;
}

bool KvTierCoordinator::write_block(int32_t layer, int32_t block, const void* data, uint64_t bytes, std::string& error) {
    if (data == nullptr) {
        error = "kv coordinator: the block has no data";
        return false;
    }
    const uint64_t want = block_bytes();
    if (bytes != want) {
        error = "kv coordinator: the block's size does not match the geometry";
        return false;
    }
    const int32_t group = group_of(block);
    int32_t slot = slot_index(layer, group);
    if (slot < 0) {
        const int32_t victim = victim_slot(error);
        if (victim < 0) return false;
        KvL1Slot& entry = slots_[(size_t) victim];
        if (entry.group >= 0) ++counters_.l1_evictions;
        entry.group = group;
        entry.layer = layer;
        entry.epoch = epoch_;
        entry.pins = 0;
        touch(victim);
        slot = victim;
    }
    const uint64_t group_len = group_bytes();
    const uint64_t offset = (uint64_t) (block % options_.blocks_per_group) * want;
    std::memcpy(slot_bytes_.data() + (size_t) slot * (size_t) group_len + offset, data, (size_t) bytes);
    touch(slot);
    return true;
}

bool KvTierCoordinator::read_block(int32_t layer, int32_t block, void* dst, uint64_t bytes, std::string& error) {
    const uint64_t want = block_bytes();
    if (dst == nullptr) {
        error = "kv coordinator: the block has no destination";
        return false;
    }
    if (bytes != want) {
        error = "kv coordinator: the block's size does not match the geometry";
        return false;
    }
    const int32_t group = group_of(block);
    const int32_t slot = slot_index(layer, group);
    if (slot < 0) {
        error = "kv coordinator: the group is not in L1";
        return false;
    }
    const uint64_t group_len = group_bytes();
    const uint64_t offset = (uint64_t) (block % options_.blocks_per_group) * want;
    std::memcpy(dst, slot_bytes_.data() + (size_t) slot * (size_t) group_len + offset, (size_t) bytes);
    return true;
}

bool KvTierCoordinator::complete(int32_t layer, int32_t block, std::string& error) {
    (void) error;
    const int32_t group = block / options_.blocks_per_group;
    auto& bits = completed_[layer];
    const size_t need = (size_t) block + 1;
    if (bits.size() < need) bits.resize(need, 0);
    bits[(size_t) block] = 1;

    // The group is whole when every one of its blocks is complete.
    const int32_t base = group * options_.blocks_per_group;
    bool whole = true;
    for (int32_t i = 0; i < options_.blocks_per_group; ++i) {
        const size_t at = (size_t) (base + i);
        if (at >= bits.size() || bits[at] == 0) { whole = false; break; }
    }
    if (!whole) return true;   // the group is not publishable yet

    // A whole group is published as one store put, and only then does it become evictable.
    return store_group(layer, group, error);
}

bool KvTierCoordinator::store_group(int32_t layer, int32_t group, std::string& error) {
    const int32_t slot = slot_index(layer, group);
    if (slot < 0) {
        error = "kv coordinator: the group is not in L1";
        return false;
    }
    const uint64_t bytes = group_bytes();
    KvBlockKey key;
    key.session_id = scope_.session_id;
    key.geometry_id = options_.geometry_id;
    key.format_id = options_.format_id;
    key.layer = layer;
    key.group = group;
    key.epoch = epoch_;

    KvBlockPut put_request;
    put_request.key = key;
    put_request.src = slot_bytes_.data() + (size_t) slot * (size_t) bytes;
    put_request.bytes = bytes;
    put_request.checksum = group_checksum(layer, group);
    if (!store_->put(put_request, error)) return false;
    ++counters_.published_groups;
    counters_.l2_write_bytes += bytes;
    published_.push_back(key);
    published_checksums_[key_string(key)] = put_request.checksum;

    // The group's blocks are no longer open: the tail is released and the slot becomes evictable.
    auto& bits = completed_[layer];
    const int32_t base = group * options_.blocks_per_group;
    for (int32_t i = 0; i < options_.blocks_per_group; ++i) {
        const size_t at = (size_t) (base + i);
        if (at < bits.size()) bits[at] = 0;
    }
    slots_[(size_t) slot].pins = 0;
    return true;
}

bool KvTierCoordinator::load_group(int32_t layer, int32_t group, int32_t slot, std::string& error) {
    const uint64_t bytes = group_bytes();
    KvBlockKey key;
    key.session_id = scope_.session_id;
    key.geometry_id = options_.geometry_id;
    key.format_id = options_.format_id;
    key.layer = layer;
    key.group = group;
    key.epoch = epoch_;

    KvBlockRequest request;
    request.key = key;
    request.offset = 0;
    request.bytes = bytes;
    request.dst = slot_bytes_.data() + (size_t) slot * (size_t) bytes;
    // The coordinator names the checksum of the group it PUBLISHED, so a store that can verify what it received
    // does. The L1 slot holds whatever was there before, so its bytes are not the group's bytes.
    const auto published = published_checksums_.find(key_string(key));
    request.checksum = published == published_checksums_.end() ? 0 : published->second;

    // The counters move when the read is ISSUED, not when it succeeds: a run that claims to have used the cold
    // tier must show a foreground cold read, and a refused read is still a cold read.
    ++counters_.l2_reads;
    counters_.l2_read_bytes += bytes;
    if (!store_->get(request, error)) {
        ++counters_.l2_read_failures;
        if (error.find("checksum") != std::string::npos) ++counters_.checksum_errors;
        return false;
    }
    return true;
}

bool KvTierCoordinator::read(int32_t layer, int32_t group, std::string& error) {
    if (store_ == nullptr) {
        error = "kv coordinator: no store is attached";
        return false;
    }
    const int32_t existing = slot_index(layer, group);
    if (existing >= 0) {
        ++counters_.l1_hits;
        touch(existing);
        return true;
    }
    ++counters_.l1_misses;
    ++counters_.readiness_waits;

    const int32_t victim = victim_slot(error);
    if (victim < 0) return false;
    if (!load_group(layer, group, victim, error)) return false;

    KvL1Slot& entry = slots_[(size_t) victim];
    if (entry.group >= 0) ++counters_.l1_evictions;
    entry.group = group;
    entry.layer = layer;
    entry.epoch = epoch_;
    entry.pins = 0;
    touch(victim);
    return true;
}

std::vector<KvBlockKey> KvTierCoordinator::published_keys() const { return published_; }

KvTier KvTierCoordinator::tier_of(int32_t layer, int32_t group) const {
    if (slot_index(layer, group) >= 0) return KvTier::kL1;
    return KvTier::kNone;
}

int32_t KvTierCoordinator::slot_of(int32_t layer, int32_t group) const { return slot_index(layer, group); }

int32_t KvTierCoordinator::completed_blocks(int32_t layer, int32_t group) const {
    const auto it = completed_.find(layer);
    if (it == completed_.end()) return 0;
    const int32_t base = group * options_.blocks_per_group;
    int32_t count = 0;
    for (int32_t i = 0; i < options_.blocks_per_group; ++i) {
        const size_t at = (size_t) (base + i);
        if (at < it->second.size() && it->second[at] == 1) ++count;
    }
    return count;
}

}  // namespace strata::core
