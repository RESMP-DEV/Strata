// src/core/kv_tier_hooks.cpp - see include/strata/core/kv_tier_hooks.hpp.
#include "strata/core/kv_tier_hooks.hpp"

#include "strata/core/kv_tier_session.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <algorithm>
#include <cstring>
#include <set>

namespace strata::core {

namespace {

KvReadBatchStats g_batch_stats;


/// The runs of one block, in the engine's layout order. This is the same order `kv_stream.cu` copies, so the
/// cold tier's payload is the engine's bytes and not a second layout.
std::vector<KvBlockRun> block_runs(const QsaState& st, const ModelGeometry& g) {
    const auto s = qsa_shapes(g);
    const int fmt = qsa_kv_format(st);
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    std::vector<KvBlockRun> runs;
    const uint64_t q4 = rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim);
    const uint64_t scales = rows * (uint64_t) (s.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
    if (fmt == strata::kernels::kKvHybrid) {
        runs.push_back(KvBlockRun{rows * (uint64_t) s.head_dim});   // the int8 K codes
        runs.push_back(KvBlockRun{scales});                        // the int8 K scales
        runs.push_back(KvBlockRun{q4});                            // the rotated q4_0 V
    } else if (fmt == strata::kernels::kKvQ4) {
        runs.push_back(KvBlockRun{q4});
        runs.push_back(KvBlockRun{q4});
    } else if (fmt == strata::kernels::kKvInt8) {
        runs.push_back(KvBlockRun{rows * (uint64_t) s.head_dim});
        runs.push_back(KvBlockRun{rows * (uint64_t) s.head_dim});
        runs.push_back(KvBlockRun{scales});
        runs.push_back(KvBlockRun{scales});
    } else {
        runs.push_back(KvBlockRun{rows * (uint64_t) s.head_dim * 2});
        runs.push_back(KvBlockRun{rows * (uint64_t) s.head_dim * 2});
    }
    return runs;
}

uint64_t total_bytes(const std::vector<KvBlockRun>& runs) {
    uint64_t sum = 0;
    for (const auto& run : runs) sum += run.bytes;
    return sum;
}

/// The layer's host copy, gathered as the per-run source pointers and their per-block lengths, in the same run
/// order the coordinator stores. The engine's own layout is the only layout: there is no second one.
struct HostRuns {
    const uint8_t* sources[4] = {nullptr, nullptr, nullptr, nullptr};
    uint64_t lengths[4] = {0, 0, 0, 0};
    int count = 0;
};

HostRuns host_runs(const QsaState& st, const ModelGeometry& g) {
    const auto s = qsa_shapes(g);
    const int fmt = qsa_kv_format(st);
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    const uint64_t q4 = rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim);
    const uint64_t scales = rows * (uint64_t) (s.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
    HostRuns runs;
    if (fmt == strata::kernels::kKvHybrid) {
        runs.sources[0] = (const uint8_t*) st.host.k_q;     runs.lengths[0] = rows * (uint64_t) s.head_dim;
        runs.sources[1] = (const uint8_t*) st.host.k_scale; runs.lengths[1] = scales;
        runs.sources[2] = (const uint8_t*) st.host.v_q4;    runs.lengths[2] = q4;
        runs.count = 3;
    } else if (fmt == strata::kernels::kKvQ4) {
        runs.sources[0] = (const uint8_t*) st.host.k_q4; runs.lengths[0] = q4;
        runs.sources[1] = (const uint8_t*) st.host.v_q4; runs.lengths[1] = q4;
        runs.count = 2;
    } else if (fmt == strata::kernels::kKvInt8) {
        runs.sources[0] = (const uint8_t*) st.host.k_q;     runs.lengths[0] = rows * (uint64_t) s.head_dim;
        runs.sources[1] = (const uint8_t*) st.host.v_q;     runs.lengths[1] = runs.lengths[0];
        runs.sources[2] = (const uint8_t*) st.host.k_scale; runs.lengths[2] = scales;
        runs.sources[3] = (const uint8_t*) st.host.v_scale; runs.lengths[3] = scales;
        runs.count = 4;
    } else {
        runs.sources[0] = (const uint8_t*) st.host.k_pool; runs.lengths[0] = rows * (uint64_t) s.head_dim * 2;
        runs.sources[1] = (const uint8_t*) st.host.v_pool; runs.lengths[1] = runs.lengths[0];
        runs.count = 2;
    }
    return runs;
}

/// Gather one block's bytes from the engine's host copy, in layout order.
bool gather_block(const HostRuns& runs, int64_t block, std::vector<uint8_t>& out, uint64_t block_bytes,
                  std::string& error) {
    out.clear();
    for (int r = 0; r < runs.count; ++r) {
        if (runs.sources[r] == nullptr) {
            error = "kv tier: the layer has no host copy for this run";
            return false;
        }
        const uint64_t offset = (uint64_t) block * runs.lengths[r];
        out.insert(out.end(), runs.sources[r] + offset, runs.sources[r] + offset + runs.lengths[r]);
    }
    if (out.size() != block_bytes) {
        error = "kv tier: the block's bytes do not match the geometry";
        return false;
    }
    return true;
}

/// Scatter one block back into the engine's own pinned host copy. This is the consumption boundary: the captured
/// `qsa_kv_resolve` still reads `QsaState::host`, so a cold group is not "used" until its bytes are here.
bool scatter_block(const HostRuns& runs, int64_t block, const std::vector<uint8_t>& in, uint64_t block_bytes,
                   std::string& error) {
    if (in.size() != block_bytes) {
        error = "kv tier: the restored block's bytes do not match the geometry";
        return false;
    }
    size_t at = 0;
    for (int r = 0; r < runs.count; ++r) {
        if (runs.sources[r] == nullptr) {
            error = "kv tier: the layer has no host copy for this run";
            return false;
        }
        const uint64_t offset = (uint64_t) block * runs.lengths[r];
        uint8_t* dst = const_cast<uint8_t*>(runs.sources[r]);
        std::memcpy(dst + offset, in.data() + at, (size_t) runs.lengths[r]);
        at += (size_t) runs.lengths[r];
    }
    return true;
}

}  // namespace

bool kv_tier_session_on() { return KvTierSession::enabled(); }

KvTierLayerGeometry kv_tier_geometry(const QsaState& st, const ModelGeometry& g) {
    KvTierLayerGeometry geometry;
    const auto s = qsa_shapes(g);
    geometry.blocks = (st.max_cells + s.idx_block - 1) / (s.idx_block > 0 ? s.idx_block : 1);
    geometry.runs = block_runs(st, g);
    geometry.block_bytes = total_bytes(geometry.runs);
    return geometry;
}

bool kv_tier_append(int32_t layer, const QsaState& st, const ModelGeometry& g, int64_t first_block, int64_t n_blocks,
                    std::string& error) {
    if (!KvTierSession::enabled()) return true;
    KvTierCoordinator* coordinator = KvTierSession::coordinator(layer);
    if (coordinator == nullptr) return true;

    const KvTierLayerGeometry geometry = kv_tier_geometry(st, g);
    if (geometry.block_bytes == 0) return true;
    const HostRuns runs = host_runs(st, g);
    if (runs.count == 0) return true;   // not a streamed layer: the engine's own path owns it

    // The blocks this window appended. The block's BYTES go into the coordinator's L1 slot before the block is
    // completed, because a group published without its bytes would store another group's payload.
    std::vector<uint8_t> block_bytes;
    for (int64_t i = 0; i < n_blocks; ++i) {
        const int64_t block = first_block + i;
        if (!coordinator->append(layer, (int32_t) block, error)) return false;
        if (!gather_block(runs, block, block_bytes, geometry.block_bytes, error)) return false;
        if (!coordinator->write_block(layer, (int32_t) block, block_bytes.data(), geometry.block_bytes, error))
            return false;
        if (!coordinator->complete(layer, (int32_t) block, error)) return false;
    }
    if (!coordinator->flush(error)) return false;
    return true;
}

bool kv_tier_commit(int32_t layer, const QsaState& st, const ModelGeometry& g, int64_t first_pos, int64_t n_cells,
                    std::string& error) {
    if (!KvTierSession::enabled()) return true;
    KvTierCoordinator* coordinator = KvTierSession::coordinator(layer);
    if (coordinator == nullptr) return true;
    if (n_cells <= 0) return true;

    const int64_t cells_per_block = qsa_shapes(g).idx_block;
    if (cells_per_block <= 0) return true;
    const KvTierLayerGeometry geometry = kv_tier_geometry(st, g);
    if (geometry.block_bytes == 0) return true;
    const HostRuns runs = host_runs(st, g);
    if (runs.count == 0) return true;

    const int64_t first_block = first_pos / cells_per_block;
    const int64_t last_block = (first_pos + n_cells - 1) / cells_per_block;
    std::vector<uint8_t> block_bytes;
    for (int64_t block = first_block; block <= last_block; ++block) {
        const int32_t block_index = (int32_t) block;
        if (!coordinator->append(layer, block_index, error)) return false;
        // The block's BYTES go in before the block is completed, for the same reason as in the append hook.
        if (!gather_block(runs, block, block_bytes, geometry.block_bytes, error)) return false;
        if (!coordinator->write_block(layer, block_index, block_bytes.data(), geometry.block_bytes, error)) return false;
        // Only a block whose cells are all written is complete. The tail block is the one still being filled.
        const int64_t block_end = (block + 1) * cells_per_block;
        if (block_end <= first_pos + n_cells) {
            if (!coordinator->complete(layer, block_index, error)) return false;
        }
    }
    if (!coordinator->flush(error)) return false;
    return true;
}

bool kv_tier_resolve(int32_t layer, const QsaState& st, const ModelGeometry& g, const int32_t* ids, int64_t n_ids,
                     std::string& error) {
    if (!KvTierSession::enabled()) return true;
    KvTierCoordinator* coordinator = KvTierSession::coordinator(layer);
    if (coordinator == nullptr) return true;
    if (ids == nullptr || n_ids <= 0) return true;

    // The selection names CELLS; the coordinator's granularity is the block, so the cells are mapped to the
    // blocks that contain them and collected into ONE batch, deduplicated by group. A token that names hundreds
    // of blocks then issues one request per GROUP, not one per block.
    const int64_t cells_per_block = qsa_shapes(g).idx_block;
    if (cells_per_block <= 0) return true;
    const KvTierLayerGeometry geometry = kv_tier_geometry(st, g);
    const HostRuns runs = host_runs(st, g);
    if (geometry.block_bytes == 0 || runs.count == 0) return true;
    KvReadBatch batch(layer, KvReadKind::kForeground);
    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t cell = ids[i];
        if (cell < 0) continue;
        const int32_t block = cell / (int32_t) cells_per_block;
        batch.add(layer, block / coordinator->blocks_per_group(), -1);
    }
    if (batch.empty()) return true;

    ++g_batch_stats.batches;
    g_batch_stats.groups += batch.size();
    g_batch_stats.requested += batch.requested();
    ++g_batch_stats.foreground;

    for (const KvReadItem& item : batch.items()) {
        if (!coordinator->read(layer, item.group, error)) return false;
        std::vector<uint8_t> block_bytes;
        block_bytes.resize((size_t) geometry.block_bytes);
        const int64_t first_block = (int64_t) item.group * coordinator->blocks_per_group();
        const int64_t end_block =
            std::min<int64_t>(geometry.blocks, first_block + coordinator->blocks_per_group());
        for (int64_t block = first_block; block < end_block; ++block) {
            if (!coordinator->read_block(layer, (int32_t) block, block_bytes.data(), geometry.block_bytes, error))
                return false;
            if (!scatter_block(runs, block, block_bytes, geometry.block_bytes, error)) return false;
        }
        coordinator->note_engine_host_restore();
    }
    return true;
}

KvReadBatchStats kv_tier_batch_stats() { return g_batch_stats; }

}  // namespace strata::core
