// src/core/kv_file_store.cpp - see include/strata/core/kv_file_store.hpp.
#include "strata/core/kv_file_store.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// The L2 file store uses POSIX file calls (pread, pwrite, ftruncate). The Windows equivalents are not provided
// yet, so on Windows the store REFUSES every operation rather than compiling calls it cannot make. The tier is
// opt-in and off on Windows until this changes; the refusal is explicit so a caller knows why.
#if defined(_WIN32)
#define STRATA_KV_FILE_STORE_POSIX 0
#else
#define STRATA_KV_FILE_STORE_POSIX 1
#endif

namespace strata::core {

namespace {

constexpr uint64_t kHeaderSize = 4096;
constexpr uint64_t kEntryBytes = 40;
constexpr uint32_t kFormatVersion = 1;

uint64_t crc32(const uint8_t* data, size_t bytes) {
    return kv_payload_checksum(data, bytes);
}

uint64_t align_up(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

#pragma pack(push, 1)
struct Header {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint64_t session_id;
    int32_t layer;
    uint64_t geometry_id;
    uint64_t format_id;
    uint64_t blocks_per_group;
    uint64_t block_bytes;
    uint64_t group_bytes;
    uint64_t run_count;
    uint64_t max_groups;
    uint64_t groups;
    uint64_t entries_offset;
    uint64_t payload_offset;
    uint64_t payload_bytes;
    uint64_t reserved[4];
    uint64_t checksum;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 156, "the KV file header layout changed");

#pragma pack(push, 1)
struct GroupEntry {
    uint32_t valid;
    uint32_t epoch;
    uint64_t checksum;
    uint64_t written_at;
    uint64_t reserved[2];
};
#pragma pack(pop)
static_assert(sizeof(GroupEntry) == kEntryBytes, "the KV group entry layout changed");

}  // namespace

uint64_t KvFileStore::payload_checksum(const void* data, size_t bytes) {
    return crc32(static_cast<const uint8_t*>(data), bytes);
}

void KvFileStore::close() {
#if STRATA_KV_FILE_STORE_POSIX
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    entries_.clear();
}

bool KvFileStore::open(const std::string& path, const Identity& identity, const std::vector<KvBlockRun>& runs,
                       std::string& error) {
    close();
#if !STRATA_KV_FILE_STORE_POSIX
    (void) path;
    (void) identity;
    (void) runs;
    error = "kv file store: the local file store is not available on this platform";
    return false;
#else
    path_ = path;
    identity_ = identity;
    runs_ = runs;
    block_bytes_ = 0;
    for (const auto& run : runs_) {
        if (run.bytes == 0) {
            error = "kv file store: a run is empty";
            return false;
        }
        if (run.bytes % 16 != 0) {
            error = "kv file store: a run is not 16-byte aligned";
            return false;
        }
        block_bytes_ += run.bytes;
    }
    if (identity.blocks_per_group <= 0 || identity.group_bytes == 0) {
        error = "kv file store: the geometry is empty";
        return false;
    }
    if (identity.group_bytes != block_bytes_ * (uint64_t) identity.blocks_per_group) {
        error = "kv file store: the group's size does not match the runs";
        return false;
    }
    group_bytes_ = identity.group_bytes;

    std::error_code fs_error;
    const std::filesystem::path parent = std::filesystem::path(path_).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, fs_error);
    if (fs_error) {
        error = "kv file store: the directory could not be created: " + fs_error.message();
        return false;
    }

    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd_ < 0) {
        error = std::string("kv file store: the file could not be opened: ") + std::strerror(errno);
        return false;
    }

    // The file is rewritten when it does not match this identity: a store of another geometry or format is not
    // readable here, and its bytes must not be served as if they were.
    Header header{};
    bool reuse = false;
    if (::pread(fd_, &header, sizeof(header), 0) == (ssize_t) sizeof(header) &&
        std::memcmp(header.magic, "STRATKV1", 8) == 0 && header.version == kFormatVersion &&
        header.header_size == kHeaderSize && header.session_id == identity.session_id &&
        header.layer == identity.layer && header.geometry_id == identity.geometry_id &&
        header.format_id == identity.format_id &&
        header.blocks_per_group == (uint64_t) identity.blocks_per_group &&
        header.block_bytes == block_bytes_ && header.group_bytes == group_bytes_ &&
        header.run_count == runs_.size()) {
        reuse = true;
    }

    if (reuse) {
        // The header is the single source of truth for the layout: the reader adopts ITS offsets.
        entries_offset_ = header.entries_offset;
        payload_offset_ = header.payload_offset;
        group_count_ = header.groups;
        for (uint64_t g = 0; g < group_count_; ++g) {
            GroupEntry entry{};
            if (::pread(fd_, &entry, sizeof(entry), (off_t) (entries_offset_ + g * kEntryBytes)) ==
                (ssize_t) sizeof(entry)) {
                if (entry.valid != 0) {
                    Entry one;
                    one.valid = entry.valid;
                    one.epoch = entry.epoch;
                    one.checksum = entry.checksum;
                    entries_[g] = one;
                }
            }
        }
        return true;
    }

    // A fresh file: a fixed header, then the group directory, then the payload area.
    if (::ftruncate(fd_, 0) != 0) {
        error = std::string("kv file store: the file could not be truncated: ") + std::strerror(errno);
        close();
        return false;
    }
    entries_offset_ = kHeaderSize;
    // The directory is a FIXED area, sized for every group the store can ever hold. A directory that grew with
    // the group count would put a later group's entry into an earlier group's payload.
    payload_offset_ = align_up(entries_offset_ + identity.max_groups * kEntryBytes, 4096);
    group_count_ = 0;   // the store grows as groups are written
    std::memcpy(header.magic, "STRATKV1", 8);
    header.version = kFormatVersion;
    header.header_size = (uint32_t) kHeaderSize;
    header.session_id = identity.session_id;
    header.layer = identity.layer;
    header.geometry_id = identity.geometry_id;
    header.format_id = identity.format_id;
    header.blocks_per_group = (uint64_t) identity.blocks_per_group;
    header.block_bytes = block_bytes_;
    header.group_bytes = group_bytes_;
    header.run_count = runs_.size();
    header.max_groups = identity.max_groups;
    header.groups = group_count_;
    header.entries_offset = entries_offset_;
    header.payload_offset = payload_offset_;
    header.payload_bytes = 0;
    header.checksum = crc32(reinterpret_cast<const uint8_t*>(&header), sizeof(header) - sizeof(header.checksum));
    if (::pwrite(fd_, &header, sizeof(header), 0) != (ssize_t) sizeof(header)) {
        error = std::string("kv file store: the header could not be written: ") + std::strerror(errno);
        close();
        return false;
    }
    return true;
#endif
}

bool KvFileStore::matches(const KvBlockKey& key) const {
    return key.session_id == identity_.session_id && key.geometry_id == identity_.geometry_id &&
           key.format_id == identity_.format_id && key.layer == identity_.layer;
}

uint64_t KvFileStore::group_index(const KvBlockKey& key) const {
    // The store holds ONE layer's groups, so the index is the group number itself. A key of another layer is
    // refused by `matches`, so a mixed index would only make the file's size explode.
    return (uint64_t) (uint32_t) key.group;
}

bool KvFileStore::contains(const KvBlockKey& key) const {
    if (!matches(key)) return false;
    const auto it = entries_.find(group_index(key));
    return it != entries_.end() && it->second.valid != 0 && it->second.epoch == key.epoch;
}

bool KvFileStore::read_exact(uint64_t offset, void* dst, size_t bytes, std::string& error) const {
#if !STRATA_KV_FILE_STORE_POSIX
    (void) offset;
    (void) dst;
    (void) bytes;
    error = "kv file store: the local file store is not available on this platform";
    return false;
#else
    size_t done = 0;
    auto* out = static_cast<uint8_t*>(dst);
    while (done < bytes) {
        const ssize_t got = ::pread(fd_, out + done, bytes - done, (off_t) (offset + done));
        if (got <= 0) {
            error = std::string("kv file store: the read failed: ") + std::strerror(errno);
            return false;
        }
        done += (size_t) got;
    }
    return true;
#endif
}

bool KvFileStore::write_exact(uint64_t offset, const void* src, size_t bytes, std::string& error) {
#if !STRATA_KV_FILE_STORE_POSIX
    (void) offset;
    (void) src;
    (void) bytes;
    error = "kv file store: the local file store is not available on this platform";
    return false;
#else
    size_t done = 0;
    const auto* in = static_cast<const uint8_t*>(src);
    while (done < bytes) {
        const ssize_t put = ::pwrite(fd_, in + done, bytes - done, (off_t) (offset + done));
        if (put <= 0) {
            error = std::string("kv file store: the write failed: ") + std::strerror(errno);
            return false;
        }
        done += (size_t) put;
    }
    return true;
#endif
}

bool KvFileStore::read_entry(uint64_t group, Entry& entry, std::string& error) const {
    GroupEntry raw{};
    if (!read_exact(entries_offset_ + group * kEntryBytes, &raw, sizeof(raw), error)) return false;
    entry.valid = raw.valid;
    entry.epoch = raw.epoch;
    entry.checksum = raw.checksum;
    return true;
}

bool KvFileStore::write_entry(uint64_t group, const Entry& entry, std::string& error) {
    GroupEntry raw{};
    raw.valid = entry.valid;
    raw.epoch = entry.epoch;
    raw.checksum = entry.checksum;
    raw.written_at = (uint64_t) time(nullptr);
    if (!write_exact(entries_offset_ + group * kEntryBytes, &raw, sizeof(raw), error)) return false;
    return true;
}

bool KvFileStore::get(const KvBlockRequest& request, std::string& error) {
    ++reads_;
    if (fd_ < 0) {
        error = "kv file store: the store is closed";
        return false;
    }
    if (!matches(request.key)) {
        error = "kv file store: the key is not this store's geometry";
        return false;
    }
    const uint64_t group = group_index(request.key);
    const auto it = entries_.find(group);
    if (it == entries_.end() || it->second.valid == 0 || it->second.epoch != request.key.epoch) {
        // A group of an older epoch is absent, not stale data: a rewind must never read the bytes of the
        // sequence before it.
        error = "kv file store: the group is absent";
        return false;
    }
    if (request.dst == nullptr || request.bytes == 0) {
        error = "kv file store: the request has no destination";
        return false;
    }
    if (request.offset + request.bytes > group_bytes_) {
        error = "kv file store: the request runs past the group's payload";
        return false;
    }
    // The group's checksum covers the WHOLE group payload, so a partial read cannot verify it. A whole-group
    // read verifies; a partial read is refused, because an unverifiable read is exactly the torn read the
    // valid marker is there to prevent.
    if (request.offset != 0 || request.bytes != group_bytes_) {
        error = "kv file store: the group's checksum covers the whole group, so a partial read is refused";
        return false;
    }
    std::vector<uint8_t> buffer((size_t) request.bytes);
    const uint64_t offset = payload_offset_ + group * group_bytes_;
    if (!read_exact(offset, buffer.data(), buffer.size(), error)) return false;
    const uint64_t actual = crc32(buffer.data(), buffer.size());
    if (actual != it->second.checksum) {
        ++checksum_errors_;
        error = "kv file store: the group's checksum does not match";
        return false;
    }
    std::memcpy(request.dst, buffer.data(), buffer.size());
    return true;
}

bool KvFileStore::put(const KvBlockPut& put_request, std::string& error) {
    if (fd_ < 0) {
        error = "kv file store: the store is closed";
        return false;
    }
    if (!matches(put_request.key)) {
        error = "kv file store: the key is not this store's geometry";
        return false;
    }
    if (put_request.src == nullptr || put_request.bytes != group_bytes_) {
        error = "kv file store: the put is not one whole group";
        return false;
    }
    const uint64_t group = group_index(put_request.key);
    if (identity_.max_groups == 0 || group >= identity_.max_groups) {
        error = "kv file store: the group is past the store's directory capacity";
        return false;
    }
    if (group + 1 > group_count_) {
        // The store grows: the group directory and the payload area follow.
        const uint64_t old_groups = group_count_;
        group_count_ = group + 1;
#if STRATA_KV_FILE_STORE_POSIX
        if (::ftruncate(fd_, (off_t) (payload_offset_ + group_count_ * group_bytes_)) != 0) {
            error = std::string("kv file store: the file could not be extended: ") + std::strerror(errno);
            group_count_ = old_groups;
            return false;
        }
        Header header{};
        if (::pread(fd_, &header, sizeof(header), 0) == (ssize_t) sizeof(header)) {
            header.groups = group_count_;
            header.payload_bytes = group_count_ * group_bytes_;
            header.checksum = crc32(reinterpret_cast<const uint8_t*>(&header), sizeof(header) - sizeof(header.checksum));
            if (::pwrite(fd_, &header, sizeof(header), 0) != (ssize_t) sizeof(header)) {
                error = std::string("kv file store: the header could not be updated: ") + std::strerror(errno);
                group_count_ = old_groups;
                return false;
            }
        }
#endif
    }

    // The payload first, and the valid marker only after it: a reader never sees a group whose marker is set but
    // whose payload is incomplete.
    const uint64_t offset = payload_offset_ + group * group_bytes_;
    if (!write_exact(offset, put_request.src, (size_t) put_request.bytes, error)) return false;
    // The payload must be durable BEFORE the valid marker is committed, so a reader never sees a group whose
    // marker is set but whose payload is incomplete.
#if STRATA_KV_FILE_STORE_POSIX
    if (::fsync(fd_) != 0) {
        error = std::string("kv file store: the payload could not be flushed: ") + std::strerror(errno);
        return false;
    }
#endif
    Entry entry;
    entry.valid = 1;
    entry.checksum = put_request.checksum;
    entry.epoch = put_request.key.epoch;
    if (!write_entry(group, entry, error)) return false;
    entries_[group] = entry;
    ++writes_;
    return true;
}

bool KvFileStore::invalidate(const KvBlockKey& key, std::string& error) {
    if (fd_ < 0) {
        error = "kv file store: the store is closed";
        return false;
    }
    if (!matches(key)) {
        error = "kv file store: the key is not this store's geometry";
        return false;
    }
    const uint64_t group = group_index(key);
    const auto it = entries_.find(group);
    if (it == entries_.end() || it->second.valid == 0) {
        error = "kv file store: the group is absent";
        return false;
    }
    Entry entry;
    entry.valid = 0;
    entry.checksum = it->second.checksum;
    entry.epoch = it->second.epoch;
    if (!write_entry(group, entry, error)) return false;
    entries_[group] = entry;
    return true;
}

}  // namespace strata::core
