// include/strata/core/kv_mooncake_store.hpp - the L3 shared store, behind the same KvBlockStore ABI.
//
// Mooncake's Transfer Engine and Store are a shared object pool with placement, replication, and asynchronous
// batch transfer. They do not understand Strata's QSA block layout, and they must not: the correct boundary is
// that the store moves OPAQUE BYTES under a key Strata owns.
//
// So this adapter owns everything semantic:
//   - the key, which carries the session, the geometry, the format, the layer, the group, and the epoch;
//   - the epoch, so a rewind's bytes are never served as current data;
//   - the payload's checksum, so a group that arrived corrupted is refused rather than returned;
//   - the quota, so a shared pool cannot be filled past its budget.
//
// The transport underneath is a narrow interface, `KvSharedTransport`, so the same adapter works against a real
// Mooncake client and against the in-process transport the tests use. The adapter's contract is testable without
// a second host, which is exactly what a shared tier needs before a second host exists.
#pragma once

#include "strata/core/kv_coordinator.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

/// The narrow transport the shared store needs. A real Mooncake client implements this; the tests implement it
/// in process. The payload is opaque bytes: the transport does not parse it.
class KvSharedTransport {
public:
    virtual ~KvSharedTransport() = default;

    /// Store the bytes under the key. The transport may place a copy wherever it wants.
    virtual bool put(const std::string& key, const void* data, size_t bytes, std::string& error) = 0;

    /// Fetch the bytes. A key the transport does not hold is a failure with an error, never a zero fill.
    virtual bool get(const std::string& key, std::vector<uint8_t>& out, std::string& error) = 0;

    /// Drop the key.
    virtual bool remove(const std::string& key, std::string& error) = 0;

    /// Does the transport hold the key?
    virtual bool has(const std::string& key) const = 0;
};

/// The shared store over a transport.
class KvMooncakeStore : public KvBlockStore {
public:
    struct Options {
        uint64_t session_id = 0;
        uint64_t geometry_id = 0;
        uint64_t format_id = 0;
        int32_t layer = -1;              ///< the layer the store holds; another layer's key is refused
        int32_t blocks_per_group = kKvGroupBlocks;
        uint64_t group_bytes = 0;
        uint64_t quota_bytes = 0;         ///< 0 means no quota
    };

    KvMooncakeStore(KvSharedTransport* transport, Options options);
    ~KvMooncakeStore() override;

    bool contains(const KvBlockKey& key) const override;
    bool get(const KvBlockRequest& request, std::string& error) override;
    bool put(const KvBlockPut& put_request, std::string& error) override;
    bool invalidate(const KvBlockKey& key, std::string& error) override;

    /// The bytes the store holds, and the quota it allows.
    uint64_t used_bytes() const { return used_bytes_; }
    uint64_t quota_bytes() const { return options_.quota_bytes; }
    uint64_t remaining_bytes() const;

    /// The counters.
    uint64_t reads() const { return reads_; }
    uint64_t writes() const { return writes_; }
    uint64_t checksum_errors() const { return checksum_errors_; }
    uint64_t quota_refusals() const { return quota_refusals_; }

    /// The object key for a block key. It is the key Strata owns and the transport only carries.
    static std::string object_key(const KvBlockKey& key);

private:
    bool matches(const KvBlockKey& key) const;
    uint64_t checksum_of(const void* data, size_t bytes) const;

    KvSharedTransport* transport_;
    Options options_;
    std::map<std::string, uint64_t> sizes_;   ///< the object key to its size, for the quota
    uint64_t used_bytes_ = 0;
    uint64_t reads_ = 0;
    uint64_t writes_ = 0;
    uint64_t checksum_errors_ = 0;
    uint64_t quota_refusals_ = 0;
};

/// An in-process transport. It is the test double for the shared store, and it is also a local reference for what
/// a real transport must provide.
class KvMemoryTransport : public KvSharedTransport {
public:
    bool put(const std::string& key, const void* data, size_t bytes, std::string& error) override;
    bool get(const std::string& key, std::vector<uint8_t>& out, std::string& error) override;
    bool remove(const std::string& key, std::string& error) override;
    bool has(const std::string& key) const override;

    /// Corrupt the stored bytes of a key, so the store's refusal path is testable.
    void corrupt(const std::string& key) { corrupt_ = key; has_corrupt_ = true; }

    size_t size() const { return objects_.size(); }

private:
    std::map<std::string, std::vector<uint8_t>> objects_;
    std::string corrupt_;
    bool has_corrupt_ = false;
};

/// A reference implementation of the pool semantics a Mooncake Master owns: one quota and one eviction order
/// across every object identity, not one budget per Strata store. It is intentionally in-process and dependency
/// free. A real Mooncake client remains the transport for cross-host placement, replication, and batch RDMA.
class KvMooncakePoolTransport : public KvSharedTransport {
public:
    struct Options {
        uint64_t quota_bytes = 0;   ///< zero means unbounded
        size_t max_objects = 0;     ///< zero means unbounded
    };

    KvMooncakePoolTransport() = default;
    explicit KvMooncakePoolTransport(Options options) : options_(options) {}

    bool put(const std::string& key, const void* data, size_t bytes, std::string& error) override;
    bool get(const std::string& key, std::vector<uint8_t>& out, std::string& error) override;
    bool remove(const std::string& key, std::string& error) override;
    bool has(const std::string& key) const override;

    uint64_t used_bytes() const;
    uint64_t quota_bytes() const { return options_.quota_bytes; }
    size_t size() const;
    uint64_t admissions() const;
    uint64_t evictions() const;
    uint64_t quota_refusals() const;
    uint64_t immutable_refusals() const;

private:
    struct Entry {
        std::vector<uint8_t> payload;
        uint64_t stamp = 0;
    };

    void evict_for_locked(uint64_t incoming_bytes, size_t incoming_objects, std::string& error);

    Options options_;
    mutable std::mutex mutex_;
    std::map<std::string, Entry> objects_;
    uint64_t used_bytes_ = 0;
    uint64_t tick_ = 0;
    uint64_t admissions_ = 0;
    uint64_t evictions_ = 0;
    uint64_t quota_refusals_ = 0;
    uint64_t immutable_refusals_ = 0;
};

}  // namespace strata::core
