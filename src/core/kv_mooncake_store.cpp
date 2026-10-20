// src/core/kv_mooncake_store.cpp - see include/strata/core/kv_mooncake_store.hpp.
#include "strata/core/kv_mooncake_store.hpp"

#include <cstring>
#include <sstream>

namespace strata::core {

namespace {

// The one checksum function, from the shared header.
uint64_t fnv1a(const void* data, size_t bytes) {
    return kv_payload_checksum(data, bytes);
}

}  // namespace

// ---- KvMemoryTransport ----

bool KvMemoryTransport::put(const std::string& key, const void* data, size_t bytes, std::string& error) {
    if (data == nullptr || bytes == 0) {
        error = "kv shared transport: the put has no payload";
        return false;
    }
    const uint8_t* p = static_cast<const uint8_t*>(data);
    objects_[key] = std::vector<uint8_t>(p, p + bytes);
    return true;
}

bool KvMemoryTransport::get(const std::string& key, std::vector<uint8_t>& out, std::string& error) {
    const auto it = objects_.find(key);
    if (it == objects_.end()) {
        error = "kv shared transport: the object is absent";
        return false;
    }
    if (has_corrupt_ && corrupt_ == key) {
        error = "kv shared transport: the object's checksum does not match";
        return false;
    }
    out = it->second;
    return true;
}

bool KvMemoryTransport::remove(const std::string& key, std::string& error) {
    const auto it = objects_.find(key);
    if (it == objects_.end()) {
        error = "kv shared transport: the object is absent";
        return false;
    }
    objects_.erase(it);
    return true;
}

bool KvMemoryTransport::has(const std::string& key) const { return objects_.find(key) != objects_.end(); }

// ---- KvMooncakePoolTransport ----

void KvMooncakePoolTransport::evict_for_locked(uint64_t incoming_bytes, size_t incoming_objects,
                                               std::string& error) {
    const bool byte_quota = options_.quota_bytes != 0;
    const bool object_quota = options_.max_objects != 0;
    if (byte_quota && incoming_bytes > options_.quota_bytes) {
        ++quota_refusals_;
        error = "kv shared pool: one object exceeds the pool quota";
        return;
    }
    while ((byte_quota && used_bytes_ + incoming_bytes > options_.quota_bytes) ||
           (object_quota && objects_.size() + incoming_objects > options_.max_objects)) {
        if (objects_.empty()) break;
        auto victim = objects_.begin();
        for (auto it = objects_.begin(); it != objects_.end(); ++it) {
            if (it->second.stamp < victim->second.stamp) victim = it;
        }
        used_bytes_ -= victim->second.payload.size();
        objects_.erase(victim);
        ++evictions_;
    }
    if ((byte_quota && used_bytes_ + incoming_bytes > options_.quota_bytes) ||
        (object_quota && objects_.size() + incoming_objects > options_.max_objects)) {
        ++quota_refusals_;
        error = "kv shared pool: the quota could not be satisfied";
    }
}

bool KvMooncakePoolTransport::put(const std::string& key, const void* data, size_t bytes, std::string& error) {
    if (data == nullptr || bytes == 0) {
        error = "kv shared pool: the put has no payload";
        return false;
    }
    const uint8_t* payload = static_cast<const uint8_t*>(data);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = objects_.find(key);
    if (existing != objects_.end()) {
        if (existing->second.payload.size() == bytes &&
            std::memcmp(existing->second.payload.data(), payload, bytes) == 0) {
            existing->second.stamp = ++tick_;
            return true;
        }
        ++immutable_refusals_;
        error = "kv shared pool: an existing object is immutable";
        return false;
    }

    std::string quota_error;
    evict_for_locked((uint64_t) bytes, 1, quota_error);
    if (!quota_error.empty()) {
        error = quota_error;
        return false;
    }
    Entry entry;
    entry.payload.assign(payload, payload + bytes);
    entry.stamp = ++tick_;
    objects_[key] = std::move(entry);
    used_bytes_ += (uint64_t) bytes;
    ++admissions_;
    return true;
}

bool KvMooncakePoolTransport::get(const std::string& key, std::vector<uint8_t>& out, std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = objects_.find(key);
    if (it == objects_.end()) {
        error = "kv shared pool: the object is absent";
        return false;
    }
    it->second.stamp = ++tick_;
    out = it->second.payload;
    return true;
}

bool KvMooncakePoolTransport::remove(const std::string& key, std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = objects_.find(key);
    if (it == objects_.end()) {
        error = "kv shared pool: the object is absent";
        return false;
    }
    used_bytes_ -= it->second.payload.size();
    objects_.erase(it);
    return true;
}

bool KvMooncakePoolTransport::has(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return objects_.find(key) != objects_.end();
}

uint64_t KvMooncakePoolTransport::used_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return used_bytes_;
}

size_t KvMooncakePoolTransport::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return objects_.size();
}

uint64_t KvMooncakePoolTransport::admissions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return admissions_;
}

uint64_t KvMooncakePoolTransport::evictions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return evictions_;
}

uint64_t KvMooncakePoolTransport::quota_refusals() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return quota_refusals_;
}

uint64_t KvMooncakePoolTransport::immutable_refusals() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return immutable_refusals_;
}

// ---- KvMooncakeStore ----

KvMooncakeStore::KvMooncakeStore(KvSharedTransport* transport, Options options)
    : transport_(transport), options_(options) {}

KvMooncakeStore::~KvMooncakeStore() = default;

std::string KvMooncakeStore::object_key(const KvBlockKey& key) {
    std::ostringstream out;
    // The key carries everything semantic: the session, the geometry, the format, the layer, the group, and the
    // epoch. The transport carries the bytes and nothing else, so a stale epoch's object is a different object.
    out << "strata-kv/" << key.session_id << '/' << key.geometry_id << '/' << key.format_id << '/' << key.layer
        << '/' << key.group << '/' << key.epoch;
    return out.str();
}

bool KvMooncakeStore::matches(const KvBlockKey& key) const {
    return key.session_id == options_.session_id && key.geometry_id == options_.geometry_id &&
           key.format_id == options_.format_id && (options_.layer < 0 || key.layer == options_.layer);
}

uint64_t KvMooncakeStore::checksum_of(const void* data, size_t bytes) const {
    return kv_payload_checksum(data, bytes);
}

uint64_t KvMooncakeStore::remaining_bytes() const {
    if (options_.quota_bytes == 0) return UINT64_MAX;
    return options_.quota_bytes > used_bytes_ ? options_.quota_bytes - used_bytes_ : 0;
}

bool KvMooncakeStore::contains(const KvBlockKey& key) const {
    if (!matches(key)) return false;
    return transport_ != nullptr && transport_->has(object_key(key));
}

bool KvMooncakeStore::get(const KvBlockRequest& request, std::string& error) {
    ++reads_;
    if (transport_ == nullptr) {
        error = "kv shared store: no transport";
        return false;
    }
    if (!matches(request.key)) {
        error = "kv shared store: the key is not this store's identity";
        return false;
    }
    if (request.dst == nullptr || request.bytes == 0) {
        error = "kv shared store: the request has no destination";
        return false;
    }
    if (request.offset + request.bytes > options_.group_bytes) {
        error = "kv shared store: the request runs past the group's payload";
        return false;
    }
    std::vector<uint8_t> payload;
    if (!transport_->get(object_key(request.key), payload, error)) return false;
    if (payload.size() != options_.group_bytes) {
        ++checksum_errors_;
        error = "kv shared store: the object's size does not match the geometry";
        return false;
    }
    // The store verifies the payload it received, when the caller named the checksum it expects. A shared pool
    // is a place where a group can be corrupted in transit, and a corrupted group is refused, never returned.
    if (request.checksum != 0) {
        const uint64_t actual = checksum_of(payload.data(), payload.size());
        if (actual != request.checksum) {
            ++checksum_errors_;
            error = "kv shared store: the object's checksum does not match";
            return false;
        }
    }
    std::memcpy(request.dst, payload.data() + request.offset, request.bytes);
    return true;
}

bool KvMooncakeStore::put(const KvBlockPut& put_request, std::string& error) {
    if (transport_ == nullptr) {
        error = "kv shared store: no transport";
        return false;
    }
    if (!matches(put_request.key)) {
        error = "kv shared store: the key is not this store's identity";
        return false;
    }
    if (put_request.src == nullptr || put_request.bytes != options_.group_bytes) {
        error = "kv shared store: the put is not one whole group";
        return false;
    }
    const std::string key = object_key(put_request.key);

    // The quota is checked BEFORE the write, so a full pool refuses the write rather than filling and then
    // reporting the overflow. The count moves only AFTER the transport write succeeds: a failed write must not
    // leave the pool's accounting short of what it holds, or a later invalidate would subtract the same bytes
    // twice and the count would wrap.
    uint64_t current = 0;
    if (options_.quota_bytes != 0) {
        const auto it = sizes_.find(key);
        current = it == sizes_.end() ? 0 : it->second;
        if (used_bytes_ - current + put_request.bytes > options_.quota_bytes) {
            ++quota_refusals_;
            error = "kv shared store: the quota would be exceeded";
            return false;
        }
    }

    if (!transport_->put(key, put_request.src, (size_t) put_request.bytes, error)) return false;
    if (options_.quota_bytes != 0) {
        sizes_[key] = put_request.bytes;
        used_bytes_ = used_bytes_ - current + put_request.bytes;
    }
    ++writes_;
    return true;
}

bool KvMooncakeStore::invalidate(const KvBlockKey& key, std::string& error) {
    if (transport_ == nullptr) {
        error = "kv shared store: no transport";
        return false;
    }
    if (!matches(key)) {
        error = "kv shared store: the key is not this store's identity";
        return false;
    }
    const std::string object = object_key(key);
    if (!transport_->remove(object, error)) return false;
    const auto it = sizes_.find(object);
    if (it != sizes_.end()) {
        used_bytes_ -= it->second;
        sizes_.erase(it);
    }
    return true;
}

}  // namespace strata::core
