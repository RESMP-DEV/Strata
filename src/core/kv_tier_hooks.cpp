// src/core/kv_tier_hooks.cpp - see include/strata/core/kv_tier_hooks.hpp.
#include "strata/core/kv_tier_hooks.hpp"

#include "strata/core/kv_tier_session.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <set>

namespace strata::core {

namespace {

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
    if (geometry.block_bytes == 0) {
        error = "kv tier: the layer's block geometry is empty";
        return false;
    }

    // The bytes the engine holds for this layer's host copy, in the same run order the coordinator stores.
    const int fmt = qsa_kv_format(st);
    const uint64_t rows = (uint64_t) (qsa_shapes(g).n_head_kv * qsa_shapes(g).page_size);
    const uint8_t* sources[4] = {nullptr, nullptr, nullptr, nullptr};
    uint64_t lengths[4] = {0, 0, 0, 0};
    int n_sources = 0;
    const uint64_t q4 = rows * strata::kernels::kv_q4_bytes_per_head((int) qsa_shapes(g).head_dim);
    const uint64_t scales = rows * (uint64_t) (qsa_shapes(g).head_dim / strata::kernels::KV_Q8_GROUP) * 2;
    if (fmt == strata::kernels::kKvHybrid) {
        sources[0] = (const uint8_t*) st.host.k_q;     lengths[0] = rows * (uint64_t) qsa_shapes(g).head_dim;
        sources[1] = (const uint8_t*) st.host.k_scale; lengths[1] = scales;
        sources[2] = (const uint8_t*) st.host.v_q4;    lengths[2] = q4;
        n_sources = 3;
    } else if (fmt == strata::kernels::kKvQ4) {
        sources[0] = (const uint8_t*) st.host.k_q4; lengths[0] = q4;
        sources[1] = (const uint8_t*) st.host.v_q4; lengths[1] = q4;
        n_sources = 2;
    } else if (fmt == strata::kernels::kKvInt8) {
        sources[0] = (const uint8_t*) st.host.k_q;     lengths[0] = rows * (uint64_t) qsa_shapes(g).head_dim;
        sources[1] = (const uint8_t*) st.host.v_q;     lengths[1] = lengths[0];
        sources[2] = (const uint8_t*) st.host.k_scale; lengths[2] = scales;
        sources[3] = (const uint8_t*) st.host.v_scale; lengths[3] = scales;
        n_sources = 4;
    } else {
        sources[0] = (const uint8_t*) st.host.k_pool; lengths[0] = rows * (uint64_t) qsa_shapes(g).head_dim * 2;
        sources[1] = (const uint8_t*) st.host.v_pool; lengths[1] = lengths[0];
        n_sources = 2;
    }
    if (n_sources == 0 || sources[0] == nullptr) return true;   // not a streamed layer: the engine's own path owns it

    // The blocks this window appended.
    for (int64_t i = 0; i < n_blocks; ++i) {
        const int64_t block = first_block + i;
        if (!coordinator->append(layer, (int32_t) block, error)) return false;

        // The block's bytes, gathered from the engine's host copy runs, in layout order.
        std::vector<uint8_t> block_bytes;
        block_bytes.reserve((size_t) geometry.block_bytes);
        for (int r = 0; r < n_sources; ++r) {
            const uint64_t offset = (uint64_t) block * lengths[r];
            block_bytes.insert(block_bytes.end(), sources[r] + offset, sources[r] + offset + lengths[r]);
        }
        if (block_bytes.size() != geometry.block_bytes) {
            error = "kv tier: the block's bytes do not match the geometry";
            return false;
        }
        if (!coordinator->write_block(layer, (int32_t) block, block_bytes.data(), geometry.block_bytes, error))
            return false;
        if (!coordinator->complete(layer, (int32_t) block, error)) return false;
    }
    return true;
}

bool kv_tier_commit(int32_t layer, const QsaState& st, const ModelGeometry& g, int64_t first_pos, int64_t n_cells,
                    std::string& error) {
    if (!KvTierSession::enabled()) return true;
    KvTierCoordinator* coordinator = KvTierSession::coordinator(layer);
    if (coordinator == nullptr) return true;
    if (n_cells <= 0) return true;

    // The token path's cells are the ones the committed window appended. The blocks that are COMPLETE are the
    // ones the group can be published from; a block that is still open stays pinned.
    const int64_t cells_per_block = qsa_shapes(g).idx_block;
    if (cells_per_block <= 0) return true;
    const int64_t first_block = first_pos / cells_per_block;
    const int64_t last_block = (first_pos + n_cells - 1) / cells_per_block;

    const KvTierLayerGeometry geometry = kv_tier_geometry(st, g);
    if (geometry.block_bytes == 0) return true;

    for (int64_t block = first_block; block <= last_block; ++block) {
        const int32_t block_index = (int32_t) block;
        if (!coordinator->append(layer, block_index, error)) return false;
        // Only a block whose cells are all written is complete. The engine's own count says how many blocks
        // have completed, and the tail block is the one still being filled.
        const int64_t block_end = (block + 1) * cells_per_block;
        if (block_end <= first_pos + n_cells) {
            if (!coordinator->complete(layer, block_index, error)) return false;
        }
    }
    return true;
}

bool kv_tier_resolve(int32_t layer, const QsaState& st, const ModelGeometry& g, const int32_t* ids, int64_t n_ids,
                     std::string& error) {
    (void) st;
    if (!KvTierSession::enabled()) return true;
    KvTierCoordinator* coordinator = KvTierSession::coordinator(layer);
    if (coordinator == nullptr) return true;
    if (ids == nullptr || n_ids <= 0) return true;

    // The selection names CELLS; the coordinator's granularity is the block, so the cells are mapped to the
    // blocks that contain them and deduplicated before any read is issued.
    const int64_t cells_per_block = qsa_shapes(g).idx_block;
    if (cells_per_block <= 0) return true;
    std::set<int32_t> groups;
    for (int64_t i = 0; i < n_ids; ++i) {
        const int32_t cell = ids[i];
        if (cell < 0) continue;
        const int32_t block = cell / (int32_t) cells_per_block;
        groups.insert(block / coordinator->blocks_per_group());
    }
    for (const int32_t group : groups) {
        if (!coordinator->read(layer, group, error)) return false;
    }
    return true;
}

}  // namespace strata::core
