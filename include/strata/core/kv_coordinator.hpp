// include/strata/core/kv_coordinator.hpp - the disaggregated KV tier coordinator (docs/KV_DISAGGREGATION.md).
//
// One coordinator owns the KV residency graph for ONE session, stage, and batch slot. It is the only component
// allowed to move a logical block between tiers. Layer-split and batch modes need one coordinator per stage and
// slot: the same layer ordinal refers to different states in those modes, so sharing a coordinator is forbidden.
//
// This header is the engine-neutral half of the design: block identity, the store ABI, the coordinator's state
// machine, readiness, and the counters. It has no CUDA dependency, so every rule here is testable on a host.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {

/// One immutable group of K/V blocks. `epoch` changes on a rewind or a conversation reset; a new epoch never
/// reinterprets the bytes of an older one, so a stale key can never be served as current data.
struct KvBlockKey {
    uint64_t session_id = 0;
    uint64_t geometry_id = 0;   ///< the model's K/V geometry fingerprint
    uint64_t format_id = 0;     ///< the K/V numeric format fingerprint (fp16, int8, q4_0, K8V4)
    int32_t layer = 0;          ///< the QSA layer ordinal
    int32_t group = 0;          ///< the group index inside the layer
    uint32_t epoch = 0;

    bool operator==(const KvBlockKey& other) const {
        return session_id == other.session_id && geometry_id == other.geometry_id &&
               format_id == other.format_id && layer == other.layer && group == other.group &&
               epoch == other.epoch;
    }
};

/// The identity of a coordinator's own state: a session, a stage, and a batch slot. A layer-split stage and a
/// batch slot each get their own coordinator, so their keys never collide even for the same layer ordinal.
struct KvCoordinatorScope {
    uint64_t session_id = 0;
    int32_t stage = 0;
    int32_t slot = 0;

    bool operator==(const KvCoordinatorScope& other) const {
        return session_id == other.session_id && stage == other.stage && slot == other.slot;
    }
};

/// The number of blocks in one immutable group. A group is the unit of a store transfer: it is written whole and
/// its valid marker is committed only after the entire payload.
inline constexpr int32_t kKvGroupBlocks = 32;

/// The bytes of one K/V block for a format, as the runs that make it up, in layout order. The coordinator never
/// interprets the payload; only the store and the engine do.
struct KvBlockRun {
    uint64_t bytes = 0;
};

/// The geometry a coordinator manages: the block granularity, the group size, and the format's run layout.
struct KvTierGeometry {
    uint64_t geometry_id = 0;
    uint64_t format_id = 0;
    int32_t blocks_per_group = kKvGroupBlocks;
    uint64_t block_bytes = 0;
    std::vector<KvBlockRun> runs;   ///< the block's runs, in layout order

    bool valid() const { return geometry_id != 0 && format_id != 0 && block_bytes > 0 && !runs.empty(); }
};

/// A read of one group. The payload is opaque bytes: the store moves it, the engine interprets it.
struct KvBlockRequest {
    KvBlockKey key;
    int64_t offset = 0;      ///< the byte offset inside the group's payload
    uint64_t bytes = 0;      ///< how many bytes to read
    void* dst = nullptr;     ///< where the bytes go
};

/// A write of one group. The whole group is submitted at once; the store commits its valid marker only after the
/// entire payload is durable.
struct KvBlockPut {
    KvBlockKey key;
    const void* src = nullptr;
    uint64_t bytes = 0;
    uint64_t checksum = 0;   ///< over the payload, in layout order
};

/// The store's ABI. `get` and `put` are request objects, not synchronous file calls: the first implementation may
/// complete them synchronously behind this ABI, but the ABI must not expose that shortcut as architecture, because
/// the target is asynchronous read batches overlapped with independent GPU work.
class KvBlockStore {
public:
    virtual ~KvBlockStore() = default;

    /// Does the store hold this key? A key from an older epoch is absent, never reinterpreted.
    virtual bool contains(const KvBlockKey& key) const = 0;

    /// Read one group. A group that is absent, whose valid marker is unset, or whose checksum does not match is a
    /// failure with an error, never a zero fill.
    virtual bool get(const KvBlockRequest& request, std::string& error) = 0;

    /// Write one group whole.
    virtual bool put(const KvBlockPut& put_request, std::string& error) = 0;

    /// Drop a key. Used on a rewind or a reset, so a stale group is never served.
    virtual bool invalidate(const KvBlockKey& key, std::string& error) = 0;
};

/// Where a block's bytes live right now.
enum class KvTier : int {
    kNone = 0,   ///< nowhere: it must be read before use
    kL1 = 1,     ///< the coordinator's pinned host slot
    kL2 = 2,     ///< a local immutable group in the store
    kL3 = 3,     ///< a shared group in the store
};

/// The coordinator's counters. A run that claims to have used a cold tier must show reads here: identical output
/// alone cannot tell the disk path from the host path it replaced.
struct KvTierCounters {
    uint64_t l1_hits = 0;
    uint64_t l1_misses = 0;
    uint64_t l1_evictions = 0;
    uint64_t l2_reads = 0;         ///< cold reads ISSUED to the store, successful or not
    uint64_t l2_read_failures = 0; ///< cold reads the store refused
    uint64_t l2_write_bytes = 0;
    uint64_t l2_read_bytes = 0;
    uint64_t l3_reads = 0;
    uint64_t l3_write_bytes = 0;
    uint64_t published_groups = 0;
    uint64_t checksum_errors = 0;
    uint64_t readiness_waits = 0;
};

/// One entry of the coordinator's L1: the pinned host slot that holds one group's bytes.
struct KvL1Slot {
    int32_t group = -1;
    int32_t layer = -1;
    uint32_t epoch = 0;
    uint64_t stamp = 0;    ///< the CLOCK reference stamp
    uint32_t pins = 0;     ///< the open tail is pinned and never evicted
};

/// The coordinator for one session, stage, and slot.
///
/// The state machine is deliberately small and explicit:
///
///   append(block)   -> the block is open in L1 (pinned while it is the tail)
///   complete(block) -> the block may be published once its group is whole
///   publish(group)  -> the group is written to the store and becomes evictable
///   read(group)     -> the group is in L1, and the caller waits for readiness
///
/// A group is only readable when it is whole. The tail group is never published and never evicted, so a partially
/// written group can never reach the resolver.
class KvTierCoordinator {
public:
    struct Options {
        int32_t l1_groups = 0;      ///< how many groups the L1 cache holds
        uint64_t geometry_id = 0;
        uint64_t format_id = 0;
        int32_t blocks_per_group = kKvGroupBlocks;
        uint64_t block_bytes = 0;
        std::vector<KvBlockRun> runs;
    };

    KvTierCoordinator(KvCoordinatorScope scope, Options options, KvBlockStore* store);
    ~KvTierCoordinator();

    KvTierCoordinator(const KvTierCoordinator&) = delete;
    KvTierCoordinator& operator=(const KvTierCoordinator&) = delete;

    /// The scope this coordinator owns. Two coordinators with different scopes never share state.
    const KvCoordinatorScope& scope() const { return scope_; }

    /// The current epoch. A rewind or a reset advances it, so every earlier key becomes unreachable.
    uint32_t epoch() const { return epoch_; }

    /// Advance the epoch. The L1 cache is dropped and the store's keys of the old epoch are invalidated, so a
    /// rewind never reads bytes that were written before it.
    bool rewind(std::string& error);

    /// Forget everything, including the store's groups of the current epoch. This is the conversation reset.
    bool reset(std::string& error);

    /// The keys the coordinator published in the current epoch, for the tests and for a store sweep.
    std::vector<KvBlockKey> published_keys() const;

    /// Note that a block has been appended. The block stays in L1 and, while it is the open tail, it is pinned.
    bool append(int32_t layer, int32_t block, std::string& error);

    /// Write a block's bytes into the coordinator's L1 slot. The coordinator owns the L1 bytes, so the engine
    /// hands the block over and the coordinator is what serializes, checksums, and publishes it.
    bool write_block(int32_t layer, int32_t block, const void* data, uint64_t bytes, std::string& error);

    /// Read a block's bytes back out of L1, for the resolve step that fills the VRAM slot.
    bool read_block(int32_t layer, int32_t block, void* dst, uint64_t bytes, std::string& error);

    /// Note that a block's last cell has been written. When every block of the group is complete, the group is
    /// serialized and submitted to the store; until then it is not readable.
    bool complete(int32_t layer, int32_t block, std::string& error);

    /// Make a group readable in L1, reading it from the store when it is not there. The call returns only when the
    /// group is ready, so the resolver never waits on a missing slot.
    bool read(int32_t layer, int32_t group, std::string& error);

    /// Where a group's bytes are right now.
    KvTier tier_of(int32_t layer, int32_t group) const;

    /// The L1 slot that holds a group, or -1.
    int32_t slot_of(int32_t layer, int32_t group) const;

    /// The counters. A run that claims to have used the cold tier must show `l2_reads` here.
    const KvTierCounters& counters() const { return counters_; }
    void reset_counters() { counters_ = KvTierCounters{}; }

    /// The number of groups the L1 cache holds.
    int32_t l1_groups() const { return options_.l1_groups; }

    /// The bytes of one group.
    uint64_t group_bytes() const;

    /// The number of blocks in a group.
    int32_t blocks_per_group() const { return options_.blocks_per_group; }

    /// The number of complete blocks in a group, and whether the group is whole. The caller supplies the block
    /// completion state, because the coordinator learns it through `complete`.
    int32_t completed_blocks(int32_t layer, int32_t group) const;

private:
    int32_t slot_index(int32_t layer, int32_t group) const;
    int32_t find_group(int32_t layer, int32_t group) const;
    int32_t victim_slot(std::string& error) const;
    bool load_group(int32_t layer, int32_t group, int32_t slot, std::string& error);
    bool store_group(int32_t layer, int32_t group, std::string& error);
    void touch(int32_t slot);
    uint64_t group_checksum(int32_t layer, int32_t group) const;
    uint64_t block_bytes() const;
    int32_t group_of(int32_t block) const { return block / options_.blocks_per_group; }

    KvCoordinatorScope scope_;
    Options options_;
    KvBlockStore* store_;
    uint32_t epoch_ = 0;
    uint64_t stamp_ = 0;
    std::vector<KvL1Slot> slots_;
    std::vector<uint8_t> slot_bytes_;    ///< the pinned host bytes of each slot
    /// The per-layer completion state of every block: one flag per block index, so a group is whole when all of
    /// its blocks' flags are set. The index is the block itself, not the block inside its group: two groups of the
    /// same layer must never share a flag.
    std::map<int32_t, std::vector<uint8_t>> completed_;
    /// The groups this coordinator published in the current epoch, so a reset can invalidate exactly them.
    std::vector<KvBlockKey> published_;
    KvTierCounters counters_;
};

/// A store that keeps everything in memory. It is the reference implementation of the ABI and the test double for
/// the coordinator's own tests; it is also what the L3 shared store's contract is checked against.
class KvMemoryStore : public KvBlockStore {
public:
    bool contains(const KvBlockKey& key) const override;
    bool get(const KvBlockRequest& request, std::string& error) override;
    bool put(const KvBlockPut& put_request, std::string& error) override;
    bool invalidate(const KvBlockKey& key, std::string& error) override;

    /// The groups the store holds, for the tests.
    size_t size() const { return groups_.size(); }

    /// Force a checksum mismatch on the next read of a key, so the coordinator's refusal path is testable. The
    /// store verifies the payload's checksum on every read, so a corrupted group is refused even when the
    /// coordinator holds a copy: the copy is not trusted over the store's own check.
    void corrupt(const KvBlockKey& key) { corrupt_ = key; has_corrupt_ = true; }

    /// The number of reads and writes the store served.
    uint64_t reads() const { return reads_; }
    uint64_t writes() const { return writes_; }

private:
    struct Entry {
        std::vector<uint8_t> payload;
        uint64_t checksum = 0;
        bool corrupt = false;
    };
    std::map<std::string, Entry> groups_;
    KvBlockKey corrupt_;
    bool has_corrupt_ = false;
    uint64_t reads_ = 0;
    uint64_t writes_ = 0;
};

}  // namespace strata::core
