// include/strata/core/kv_file_store.hpp - the L2 local file store (a KvBlockStore over a file).
//
// The file holds immutable GROUPS of K/V blocks. A group is written whole and its valid marker is committed only
// after the entire payload is durable, so a reader never sees a torn group: a group with an unset marker or a bad
// checksum is ABSENT, never garbage.
//
// The header is the single source of truth for the layout. The reader adopts the header's own payload offset and
// entry count instead of recomputing them, because a reader that recomputes a second layout can disagree with the
// writer about a corrupt header and pass it as clean.
//
// The payload runs are 16-byte aligned, so a direct transfer needs no padding. The group size is the coordinator's
// group size, and a group is exactly its blocks' runs in layout order.
#pragma once

#include "strata/core/kv_coordinator.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {

/// The local file store. It is created for one session, one geometry, and one format, and it refuses any key that
/// does not match that identity: a group of another geometry or format is not readable here.
class KvFileStore : public KvBlockStore {
public:
    /// The identity the store holds. A key outside it is absent, not an error.
    struct Identity {
        uint64_t session_id = 0;
        uint64_t geometry_id = 0;
        uint64_t format_id = 0;
        int32_t layer = -1;        ///< the layer the store holds; a key of another layer is refused
        int32_t blocks_per_group = kKvGroupBlocks;
        uint64_t group_bytes = 0;
    };

    /// Open (or create) the store at `path`. The file is truncated and rewritten when it does not match the
    /// identity, because a store of another geometry is not readable and must not be trusted.
    bool open(const std::string& path, const Identity& identity, const std::vector<KvBlockRun>& runs, std::string& error);

    bool contains(const KvBlockKey& key) const override;
    bool get(const KvBlockRequest& request, std::string& error) override;
    bool put(const KvBlockPut& put_request, std::string& error) override;
    bool invalidate(const KvBlockKey& key, std::string& error) override;

    /// The number of groups the store holds.
    size_t size() const { return entries_.size(); }

    /// The counters: the reads and writes the store served.
    uint64_t reads() const { return reads_; }
    uint64_t writes() const { return writes_; }
    uint64_t checksum_errors() const { return checksum_errors_; }

    /// The path.
    const std::string& path() const { return path_; }

    /// The checksum over a group's payload, in layout order. It is public so a writer and a reader use the SAME
    /// function: two checksum implementations are two chances to disagree about what "the same bytes" means.
    static uint64_t payload_checksum(const void* data, size_t bytes);

    /// Close the file and release the state.
    void close();

private:
    struct Entry {
        uint32_t valid = 0;
        uint64_t checksum = 0;
        uint32_t epoch = 0;
    };

    bool matches(const KvBlockKey& key) const;
    uint64_t group_index(const KvBlockKey& key) const;
    bool read_exact(uint64_t offset, void* dst, size_t bytes, std::string& error) const;
    bool write_exact(uint64_t offset, const void* src, size_t bytes, std::string& error);
    bool read_entry(uint64_t group, Entry& entry, std::string& error) const;
    bool write_entry(uint64_t group, const Entry& entry, std::string& error);

    std::string path_;
    Identity identity_;
    std::vector<KvBlockRun> runs_;
    uint64_t block_bytes_ = 0;
    uint64_t group_bytes_ = 0;
    int fd_ = -1;
    uint64_t entries_offset_ = 0;
    uint64_t payload_offset_ = 0;
    uint64_t group_count_ = 0;
    std::unordered_map<uint64_t, Entry> entries_;
    uint64_t reads_ = 0;
    uint64_t writes_ = 0;
    uint64_t checksum_errors_ = 0;
};

}  // namespace strata::core
