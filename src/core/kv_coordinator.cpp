// src/core/kv_coordinator.cpp - see include/strata/core/kv_coordinator.hpp.
#include "strata/core/kv_coordinator.hpp"
#include "strata/core/kv_read_batch.hpp"

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
    if (options_.async_io) {
        async_enabled_ = true;
        if (options_.async_workers == 0) options_.async_workers = 1;
        worker_ = std::thread([this] { worker_loop(); });
    }
}

KvTierCoordinator::~KvTierCoordinator() {
    {
        std::lock_guard<std::mutex> lock(io_mutex_);
        shutdown_ = true;
    }
    if (worker_.joinable()) {
        io_ready_.notify_all();
        worker_.join();
    }
}

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
    std::string async_error;
    if (!drain_async_completions(async_error) && error.empty()) error = async_error;
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
    std::string async_error;
    if (!drain_async_completions(async_error) && error.empty()) error = async_error;
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
    if (async_enabled_ && !drain_async_completions(error)) return false;
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
    if (async_enabled_ && !drain_async_completions(error)) return false;
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
    if (async_enabled_ && !drain_async_completions(error)) return false;
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

    if (async_enabled_) return submit_group(layer, group, error);

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

KvBlockKey KvTierCoordinator::operation_key(int32_t layer, int32_t group) const {
    KvBlockKey key;
    key.session_id = scope_.session_id;
    key.geometry_id = options_.geometry_id;
    key.format_id = options_.format_id;
    key.layer = layer;
    key.group = group;
    key.epoch = epoch_;
    return key;
}

bool KvTierCoordinator::has_pending_io_locked() const {
    for (const auto& [id, record] : operations_) {
        if (record.state != KvAsyncState::kCanceled) return true;
    }
    return !queue_.empty();
}

uint64_t KvTierCoordinator::pending_read_locked(int32_t layer, int32_t group) const {
    for (const auto& [id, record] : operations_) {
        if (record.layer == layer && record.group == group && record.kind == AsyncJob::Kind::kRead &&
            record.state != KvAsyncState::kCanceled)
            return id;
    }
    return 0;
}

uint64_t KvTierCoordinator::pending_put_locked(int32_t layer, int32_t group) const {
    for (const auto& [id, record] : operations_) {
        if (record.layer == layer && record.group == group && record.kind == AsyncJob::Kind::kPut &&
            record.state != KvAsyncState::kCanceled)
            return id;
    }
    return 0;
}

bool KvTierCoordinator::group_io_ready(int32_t layer, int32_t group) const {
    if (!async_enabled_) return true;
    std::lock_guard<std::mutex> lock(io_mutex_);
    return pending_read_locked(layer, group) == 0 && pending_put_locked(layer, group) == 0;
}

bool KvTierCoordinator::submit_group(int32_t layer, int32_t group, std::string& error) {
    const int32_t slot = slot_index(layer, group);
    if (slot < 0) {
        error = "kv coordinator: the group is not in L1";
        return false;
    }
    const uint64_t bytes = group_bytes();
    auto job = std::make_shared<AsyncJob>();
    job->kind = AsyncJob::Kind::kPut;
    job->layer = layer;
    job->group = group;
    job->slot = slot;
    job->key = operation_key(layer, group);
    job->payload.assign(slot_bytes_.begin() + (ptrdiff_t) ((size_t) slot * (size_t) bytes),
                        slot_bytes_.begin() + (ptrdiff_t) (((size_t) slot + 1) * (size_t) bytes));
    job->checksum = fnv1a(job->payload.data(), job->payload.size());

    std::lock_guard<std::mutex> lock(io_mutex_);
    if (pending_put_locked(layer, group) != 0 || pending_read_locked(layer, group) != 0) {
        error = "kv coordinator: the group still has pending I/O";
        return false;
    }
    job->operation = next_operation_++;
    AsyncRecord record;
    record.kind = job->kind;
    record.state = KvAsyncState::kQueued;
    record.layer = layer;
    record.group = group;
    record.slot = slot;
    record.key = job->key;
    record.checksum = job->checksum;
    operations_[job->operation] = record;
    queue_.push_back(std::move(job));
    io_ready_.notify_one();
    ++counters_.async_writes_submitted;
    return true;
}

void KvTierCoordinator::finish_operation_locked(uint64_t operation, KvAsyncState state, std::string error) {
    const auto it = operations_.find(operation);
    if (it == operations_.end()) return;
    it->second.state = state;
    it->second.error = std::move(error);
}

void KvTierCoordinator::worker_loop() {
    for (;;) {
        std::shared_ptr<AsyncJob> job;
        {
            std::unique_lock<std::mutex> lock(io_mutex_);
            io_ready_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });
            if (shutdown_ && queue_.empty()) return;
            job = queue_.front();
            queue_.pop_front();
            const auto record = operations_.find(job->operation);
            if (record == operations_.end()) continue;
            if (record->second.cancel_requested) {
                record->second.state = KvAsyncState::kCanceled;
                record->second.error = "kv coordinator: the operation was canceled";
                io_ready_.notify_all();
                continue;
            }
            record->second.state = KvAsyncState::kRunning;
        }

        bool ok = false;
        std::string error;
        if (job->kind == AsyncJob::Kind::kPut) {
            KvBlockPut request;
            request.key = job->key;
            request.src = job->payload.data();
            request.bytes = job->payload.size();
            request.checksum = job->checksum;
            ok = store_ != nullptr && store_->put(request, error);
        } else {
            KvBlockRequest request;
            request.key = job->key;
            request.offset = 0;
            request.bytes = job->payload.size();
            request.dst = job->payload.data();
            request.checksum = job->checksum;
            ok = store_ != nullptr && store_->get(request, error);
        }

        {
            std::lock_guard<std::mutex> lock(io_mutex_);
            const auto completed = operations_.find(job->operation);
            if (completed != operations_.end() && completed->second.cancel_requested) {
                completed->second.state = KvAsyncState::kCanceled;
                completed->second.error = "kv coordinator: the operation was canceled";
                io_ready_.notify_all();
                continue;
            }
            if (completed != operations_.end() && job->kind == AsyncJob::Kind::kRead)
                completed->second.payload = std::move(job->payload);
            finish_operation_locked(job->operation, ok ? KvAsyncState::kSucceeded : KvAsyncState::kFailed,
                                    std::move(error));
            io_ready_.notify_all();
        }
    }
}

bool KvTierCoordinator::read(int32_t layer, int32_t group, std::string& error) {
    if (store_ == nullptr) {
        error = "kv coordinator: no store is attached";
        return false;
    }
    if (async_enabled_ && !drain_async_completions(error)) return false;
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

bool KvTierCoordinator::submit_read(KvReadBatch& batch, std::string& error) {
    if (!async_enabled_) {
        error = "kv coordinator: asynchronous I/O is not enabled";
        return false;
    }
    if (batch.layer() != layer_) {
        error = "kv coordinator: the batch belongs to another layer";
        return false;
    }
    std::lock_guard<std::mutex> lock(io_mutex_);
    for (size_t index = 0; index < batch.items().size(); ++index) {
        const KvReadItem& item = batch.items()[index];
        if (item.layer != layer_) {
            error = "kv coordinator: the batch mixes layers";
            return false;
        }
        if (slot_index(item.layer, item.group) >= 0) continue;
        if (pending_read_locked(item.layer, item.group) != 0) continue;

        const int32_t victim = victim_slot(error);
        if (victim < 0) return false;
        auto job = std::make_shared<AsyncJob>();
        job->kind = AsyncJob::Kind::kRead;
        job->layer = item.layer;
        job->group = item.group;
        job->slot = victim;
        job->key = operation_key(item.layer, item.group);
        job->payload.resize((size_t) group_bytes());
        const auto published = published_checksums_.find(key_string(job->key));
        job->checksum = published == published_checksums_.end() ? 0 : published->second;

        ++counters_.l1_misses;
        ++counters_.l2_reads;
        counters_.l2_read_bytes += group_bytes();
        job->operation = next_operation_++;
        AsyncRecord record;
        record.kind = job->kind;
        record.state = KvAsyncState::kQueued;
        record.layer = item.layer;
        record.group = item.group;
        record.slot = victim;
        record.key = job->key;
        record.checksum = job->checksum;
        operations_[job->operation] = record;
        const uint64_t operation = job->operation;
        queue_.push_back(std::move(job));
        io_ready_.notify_one();
        ++counters_.async_reads_submitted;
        batch.set_operation(index, operation);
    }
    return true;
}

bool KvTierCoordinator::wait_read(uint64_t operation, int32_t layer, int32_t group, std::string& error) {
    if (!wait(operation, error)) {
        std::lock_guard<std::mutex> lock(io_mutex_);
        const auto failed = operations_.find(operation);
        if (failed != operations_.end()) {
            if (failed->second.kind == AsyncJob::Kind::kRead) ++counters_.l2_read_failures;
            operations_.erase(failed);
        }
        return false;
    }
    std::lock_guard<std::mutex> lock(io_mutex_);
    const auto it = operations_.find(operation);
    if (it == operations_.end()) {
        error = "kv coordinator: the operation is unknown";
        return false;
    }
    const int32_t slot = it->second.slot;
    const bool ok = it->second.state == KvAsyncState::kSucceeded;
    if (ok) {
        const uint64_t bytes = group_bytes();
        std::memcpy(slot_bytes_.data() + (size_t) slot * (size_t) bytes,
                    it->second.payload.data(), (size_t) bytes);
        KvL1Slot& entry = slots_[(size_t) slot];
        if (entry.group >= 0 && (entry.layer != layer || entry.group != group)) ++counters_.l1_evictions;
        entry.layer = layer;
        entry.group = group;
        entry.epoch = epoch_;
        entry.pins = 0;
        touch(slot);
    }
    operations_.erase(it);
    return ok;
}

bool KvTierCoordinator::wait_read(const KvReadBatch& batch, std::string& error) {
    bool ok = true;
    std::string first;
    for (const KvReadItem& item : batch.items()) {
        if (item.operation == 0) continue;
        std::string one;
        if (!wait_read(item.operation, item.layer, item.group, one) && ok) {
            ok = false;
            first = one;
        }
    }
    if (!ok && error.empty()) error = first;
    return ok;
}

bool KvTierCoordinator::cancel_read(uint64_t operation, std::string& error) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    const auto it = operations_.find(operation);
    if (it == operations_.end() || it->second.kind != AsyncJob::Kind::kRead) {
        error = "kv coordinator: the read operation is unknown";
        return false;
    }
    if (it->second.state == KvAsyncState::kSucceeded || it->second.state == KvAsyncState::kFailed ||
        it->second.state == KvAsyncState::kCanceled)
        return true;
    it->second.cancel_requested = true;
    ++counters_.async_cancellations;
    io_ready_.notify_all();
    return true;
}

bool KvTierCoordinator::completion(uint64_t operation, KvAsyncCompletion& state) const {
    std::lock_guard<std::mutex> lock(io_mutex_);
    const auto it = operations_.find(operation);
    if (it == operations_.end()) return false;
    state.operation = operation;
    state.state = it->second.state;
    state.error = it->second.error;
    return true;
}

bool KvTierCoordinator::wait(uint64_t operation, std::string& error) {
    if (!async_enabled_) {
        error = "kv coordinator: asynchronous I/O is not enabled";
        return false;
    }
    std::unique_lock<std::mutex> lock(io_mutex_);
    for (;;) {
        const auto it = operations_.find(operation);
        if (it == operations_.end()) {
            error = "kv coordinator: the operation is unknown";
            return false;
        }
        if (it->second.state != KvAsyncState::kQueued && it->second.state != KvAsyncState::kRunning) {
            if (it->second.state == KvAsyncState::kSucceeded) return true;
            error = it->second.error.empty() ? "kv coordinator: the operation did not succeed" : it->second.error;
            return false;
        }
        io_ready_.wait(lock);
    }
}

void KvTierCoordinator::clear_slot_locked(int32_t slot) {
    if (slot < 0 || (size_t) slot >= slots_.size()) return;
    slots_[(size_t) slot].group = -1;
    slots_[(size_t) slot].layer = -1;
    slots_[(size_t) slot].epoch = 0;
    slots_[(size_t) slot].stamp = 0;
    slots_[(size_t) slot].pins = 0;
}

bool KvTierCoordinator::drain_async_completions(std::string& error) {
    if (!async_enabled_) return true;
    std::unique_lock<std::mutex> lock(io_mutex_);
    io_ready_.wait(lock, [this] {
        if (!queue_.empty()) return false;
        for (const auto& entry : operations_) {
            const auto state = entry.second.state;
            if (state == KvAsyncState::kQueued || state == KvAsyncState::kRunning) return false;
        }
        return true;
    });
    bool ok = true;
    for (auto it = operations_.begin(); it != operations_.end();) {
        AsyncRecord& record = it->second;
        if (record.state == KvAsyncState::kFailed || record.state == KvAsyncState::kCanceled) {
            if (ok) {
                error = record.error;
                ok = false;
            }
            if (record.kind == AsyncJob::Kind::kRead) ++counters_.l2_read_failures;
            it = operations_.erase(it);
            continue;
        }
        if (record.state == KvAsyncState::kSucceeded && record.kind == AsyncJob::Kind::kRead) {
            const uint64_t bytes = group_bytes();
            if (record.payload.size() == bytes && record.slot >= 0 &&
                (size_t) record.slot < slots_.size()) {
                std::memcpy(slot_bytes_.data() + (size_t) record.slot * (size_t) bytes, record.payload.data(),
                            (size_t) bytes);
                KvL1Slot& entry_slot = slots_[(size_t) record.slot];
                if (entry_slot.group >= 0 && (entry_slot.layer != record.layer || entry_slot.group != record.group))
                    ++counters_.l1_evictions;
                entry_slot.layer = record.layer;
                entry_slot.group = record.group;
                entry_slot.epoch = epoch_;
                entry_slot.pins = 0;
                touch(record.slot);
            } else {
                ok = false;
                error = "kv coordinator: a completed read has no usable payload";
            }
        } else if (record.state == KvAsyncState::kSucceeded && record.kind == AsyncJob::Kind::kPut) {
            ++counters_.published_groups;
            counters_.l2_write_bytes += group_bytes();
            published_.push_back(record.key);
            published_checksums_[key_string(record.key)] = record.checksum;
            auto& bits = completed_[record.layer];
            const int32_t base = record.group * options_.blocks_per_group;
            for (int32_t i = 0; i < options_.blocks_per_group; ++i) {
                const size_t at = (size_t) (base + i);
                if (at < bits.size()) bits[at] = 0;
            }
            if (record.slot >= 0 && (size_t) record.slot < slots_.size())
                slots_[(size_t) record.slot].pins = 0;
        }
        if (record.state == KvAsyncState::kSucceeded) {
            it = operations_.erase(it);
        } else {
            ++it;
        }
    }
    return ok;
}

bool KvTierCoordinator::flush(std::string& error) {
    if (!async_enabled_) return true;
    return drain_async_completions(error);
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
