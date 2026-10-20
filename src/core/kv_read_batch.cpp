// src/core/kv_read_batch.cpp - see include/strata/core/kv_read_batch.hpp.
#include "strata/core/kv_read_batch.hpp"

#include <algorithm>

namespace strata::core {

bool KvReadBatch::add(int32_t layer, int32_t group, int32_t slot) {
    ++requested_;
    // A batch never mixes layers: the layers have different stores and different geometry identities, and one
    // batch that mixed them could not be scheduled as one unit.
    if (items_.empty()) layer_ = layer;
    else if (layer != layer_) return false;

    for (auto& item : items_) {
        if (item.group != group) continue;
        // The group is already in the batch. The first addition keeps its slot: a later duplicate must not move
        // a group that a reader is already waiting on.
        if (item.slot < 0 && slot >= 0) item.slot = slot;
        return true;
    }
    items_.push_back(KvReadItem{layer, group, slot});
    return true;
}

}  // namespace strata::core
