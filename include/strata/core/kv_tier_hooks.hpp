// include/strata/core/kv_tier_hooks.hpp - the serving-path hooks of the disaggregated KV tiers.
//
// The hooks sit where the serving path actually moves K/V, not where a prototype happened to be able to reach:
//   append  - verify.cpp's kv_append_* calls, after the engine has written the window's cells
//   resolve - verify.cpp's qsa_kv_resolve calls, before the attention reads the blocks
//
// Every hook is a no-op when the cold tier is off, so the default runtime path is byte-identical.
#pragma once

#include "strata/core/kv_coordinator.hpp"
#include "strata/core/kv_read_batch.hpp"
#include "strata/core/kv_tier_session.hpp"
#include "strata/core/layer.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

/// The number of blocks a layer has, and the geometry the coordinator needs. The engine's own block layout is the
/// only layout: the coordinator never invents a second one.
struct KvTierLayerGeometry {
    int64_t blocks = 0;
    uint64_t block_bytes = 0;
    std::vector<KvBlockRun> runs;
};

/// The block geometry for a state, derived from the engine's own layout function, so the cold tier's payload is
/// exactly the bytes the engine's pages hold.
KvTierLayerGeometry kv_tier_geometry(const QsaState& st, const ModelGeometry& g);

/// Is the cold tier on? The serving path checks this on every window, so the default path costs one branch.
bool kv_tier_session_on();

/// The append hook. The engine has written the window's cells; the coordinator notes the blocks and publishes the
/// groups that became whole. It returns false only on a hard error (a store write that failed), and the caller
/// treats that as a failed window rather than silently continuing without a cold tier.
bool kv_tier_append(int32_t layer, const QsaState& st, const ModelGeometry& g, int64_t first_block, int64_t n_blocks,
                    std::string& error);

/// The token path's hook. The token path runs inside a CAPTURED graph, so it cannot do host I/O: the publish
/// happens at the commit boundary, where the window's writes have become visible to the host. The hook notes the
/// cells the committed window produced, so the same coordinator that serves the window also publishes the tokens.
bool kv_tier_commit(int32_t layer, const QsaState& st, const ModelGeometry& g, int64_t first_pos, int64_t n_cells,
                    std::string& error);

/// The resolve hook. The selection names the blocks the attention needs; the coordinator makes the groups that are
/// not in L1 ready, so the attention never waits on a missing group.
///
/// The selection's cells are collected into ONE read batch, deduplicated by group, so a token that names hundreds
/// of blocks does not issue hundreds of separate store reads.
bool kv_tier_resolve(int32_t layer, const QsaState& st, const ModelGeometry& g, const int32_t* ids, int64_t n_ids,
                     std::string& error);

/// The read batches the session has issued, for the telemetry and for the test.
struct KvReadBatchStats {
    uint64_t batches = 0;          ///< the batches the session issued
    uint64_t groups = 0;           ///< the groups the batches asked for, after deduplication
    uint64_t requested = 0;        ///< the groups the selections named, before deduplication
    uint64_t foreground = 0;       ///< the foreground batches
    uint64_t prefetch = 0;         ///< the prefetch batches
};

/// The read-batch statistics.
KvReadBatchStats kv_tier_batch_stats();

}  // namespace strata::core
