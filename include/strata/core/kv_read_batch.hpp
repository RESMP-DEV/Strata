// include/strata/core/kv_read_batch.hpp - the read batch: the unit of the cold tier's I/O scheduling.
//
// One token's attention selection can name hundreds of blocks. Issuing a separate store read for each of them
// turns a single token into hundreds of tiny I/Os, which is the difference between a cold tier that helps and one
// that destroys throughput. A read batch collects the selection's blocks, deduplicates them, and issues one
// request per GROUP.
//
// The batch also carries the priority: a foreground read (the selection the attention is waiting for) is not the
// same as a prefetch (the next expected append group) or a migration (a group moving between stores). The
// scheduler is expected to serve foreground first, and a foreground batch may not be starved by prefetch traffic.
#pragma once

#include "strata/core/kv_coordinator.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// What a read is for. The order is the priority order.
enum class KvReadKind : int {
    kForeground = 0,   ///< the selection the attention is waiting for
    kPrefetch = 1,     ///< the next expected group, read ahead of the selection
    kMigration = 2,    ///< a group moving between the local and the shared store
};

/// One group a batch needs.
struct KvReadItem {
    int32_t layer = 0;
    int32_t group = 0;
    int32_t slot = -1;   ///< the L1 slot the group's bytes go into; -1 when the slot is not known yet
    uint64_t operation = 0; ///< the background read's explicit completion, or 0 for a synchronous item
};

/// A batch of group reads. The batch owns the deduplication: the same group named twice appears once.
class KvReadBatch {
public:
    KvReadBatch() = default;
    KvReadBatch(int32_t layer, KvReadKind kind) : layer_(layer), kind_(kind) {}

    /// Add a group. A group already in the batch is not added twice, and the first addition keeps its slot.
    bool add(int32_t layer, int32_t group, int32_t slot);

    /// The groups in the batch, in the order they were added.
    const std::vector<KvReadItem>& items() const { return items_; }

    /// How many groups the batch holds.
    size_t size() const { return items_.size(); }
    bool empty() const { return items_.empty(); }

    /// The layer the batch's groups belong to. A batch never mixes layers: the layers have different stores and
    /// different geometry identities, and one batch that mixed them could not be scheduled as one unit.
    int32_t layer() const { return layer_; }

    /// The kind of read, and so the priority.
    KvReadKind kind() const { return kind_; }
    void set_kind(KvReadKind kind) { kind_ = kind; }

    /// The bytes the batch will move, when the geometry is known. The scheduler uses it to weigh a batch.
    uint64_t bytes() const { return bytes_; }
    void set_bytes(uint64_t bytes) { bytes_ = bytes; }

    /// A deadline, in microseconds from the batch's creation. A foreground batch with a deadline is one the
    /// scheduler must not starve.
    int64_t deadline_us() const { return deadline_us_; }
    void set_deadline_us(int64_t deadline_us) { deadline_us_ = deadline_us; }

    /// The number of groups the batch asked for, before deduplication. The difference is the work the
    /// deduplication saved, and it is what a run can show instead of a throughput claim.
    uint64_t requested() const { return requested_; }

private:
    int32_t layer_ = 0;
    KvReadKind kind_ = KvReadKind::kForeground;
    std::vector<KvReadItem> items_;
    uint64_t bytes_ = 0;
    int64_t deadline_us_ = 0;
    uint64_t requested_ = 0;
};

}  // namespace strata::core
