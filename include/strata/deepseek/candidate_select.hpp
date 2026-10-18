// include/strata/deepseek/candidate_select.hpp - the CPU reference for DeepSeek V4.1 hierarchical
// candidate selection, the correctness primitive a CUDA kernel must reproduce.
//
// The model does not top-k over the whole compressed index. It selects twice:
//
//   1. choose_candidate_blocks: one coarse source scores every visible row, the rows are pooled
//      into fixed-size blocks, each block scores as its best visible row, and the top
//      `blocks_wanted` blocks are kept. The block holding the newest visible row is pinned: it is
//      the only partly filled block and it could otherwise lose to an older block on score.
//   2. choose_top_rows: a later reader source scores the same rows with different weights, and
//      its top `k` rows are taken only from inside the kept blocks. Blocks bound attention work;
//      they do not change which rows the reader prefers.
//
// Determinism is the whole point of this file. Two runs on the same inputs must return the same
// rows, so every ordering here is total and explicit:
//
//   * Primary key: descending score. NaN is treated as the lowest possible score, so it can never
//     outrank a real value. Negative infinity is an ordinary score, not a sentinel: a masked row
//     loses only by its own value.
//   * Tie key: ascending row index (ascending block index for blocks). A tie therefore resolves
//     toward the oldest row, which makes the result independent of input order and of any partition
//     of the scan. A partial sort with an inconsistent comparator would be worse than useless here.
//   * The pinned newest block always wins, against a strictly higher score elsewhere and against
//     another block with the identical best score: it is pinned before ranking, not appended after.
//
// The composition is `select_candidates`. The stages are separate, pure functions of plain spans so
// a kernel port can be checked one level at a time.
//
// Scoring is opaque here: these functions read whatever floats they are handed. Whether a masked
// row is written as -inf, -1e30f or -3.4e38f changes which rows win, and that is the caller's
// contract.
// Use `kMaskedScore` for the -inf convention the rest of the DeepSeek V4.1 stack uses.
//
// Header-only, C++20, no ggml and no engine dependency: a Strata-owned reference, not a port.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace strata::deepseek_v41 {

/// The mask value. Scores at or below this are unreachable and never selected. The comparison is
/// an ordinary descending comparison with no ordering rule of its own.
inline constexpr float kMaskedScore = -std::numeric_limits<float>::infinity();
inline constexpr float kUnreachableScore = kMaskedScore;

/// A NaN row loses to every real value; an -inf row is simply the worst score there is.
inline bool is_unreachable(float score) noexcept { return !(score > kUnreachableScore); }

/// One chosen block and the row range it covers. `row_begin` is `id * block_size`; `row_end` is
/// exclusive and clamped to the scored row count, so a partial tail block is short.
struct SelectedBlock {
    std::uint32_t id = 0;
    std::uint32_t row_begin = 0;
    std::uint32_t row_end = 0;
    std::uint32_t rows = 0;
    bool pinned = false;
};

/// One block's ranking score and flags. `BlockSelection::ranking` holds one of these per block, in
/// full rank order, which is what a near-tie test needs: it shows the scores on both sides of the
/// block cut and which blocks they belong to.
struct BlockRanking {
    float score = kUnreachableScore;
    bool selected = false;
    bool pinned = false;
};

/// What stage one decided, and what it would take to change the decision.
struct BlockSelection {
    std::vector<SelectedBlock> blocks;   ///< ascending block id, which is ascending row order
    std::vector<BlockRanking> ranking;   ///< every block, in full rank order
    std::uint32_t row_count = 0;         ///< rows considered: the visible count, clamped
    std::uint32_t visible_rows = 0;      ///< visible count after clamping
    std::uint32_t blocks_wanted = 0;
    std::uint32_t blocks_total = 0;      ///< blocks covering `row_count`
    std::uint32_t blocks_selected = 0;
    std::uint32_t pinned_block = 0;      ///< meaningful only when `pinned_block_valid`
    bool pinned_block_valid = false;
    std::uint32_t dropped_rank = 0;      ///< full rank of the first block not selected
    float kept_score = kUnreachableScore;
    float dropped_score = kUnreachableScore;
    float cut_margin = -1.0f;            ///< kept minus dropped; -1 when nothing was dropped
    /// The requested budget was fully spent and a block was left behind, so the last kept block
    /// really does outrank the first dropped one. False when the block count, rather than the
    /// scores, ended the selection, and when nothing was requested or nothing was dropped.
    bool cut_exact = false;
};

/// Stage two: the rows kept, and the boundary they were cut on.
struct RowSelection {
    std::vector<std::uint32_t> rows;     ///< ascending row index
    std::vector<float> row_scores;       ///< parallel to `rows`
    std::uint32_t candidates = 0;        ///< reachable rows inside the selected blocks
    std::uint32_t k = 0;
    std::uint32_t rows_selected = 0;
    std::uint32_t dropped_rank = 0;      ///< full rank of the first candidate not selected
    float kept_score = kUnreachableScore;
    float dropped_score = kUnreachableScore;
    float cut_margin = -1.0f;            ///< kept minus dropped; -1 when nothing was dropped
    /// The same meaning as `BlockSelection::cut_exact`: the last kept row really does outrank the
    /// first dropped candidate, rather than the cut falling where the candidates ran out.
    bool cut_exact = false;
};

/// Both stages from one call, for a caller that checks the block choice and the row choice at
/// once.
struct CandidateSelection {
    BlockSelection blocks;
    RowSelection rows;
};

/// The valid stage-one result for a query that sees nothing.
inline BlockSelection empty_block_selection(std::uint32_t blocks_total,
                                             std::uint32_t blocks_wanted) {
    BlockSelection s;
    s.blocks_total = blocks_total;
    s.blocks_wanted = blocks_wanted;
    return s;
}


/// The valid stage-two result for a query that sees nothing.
inline RowSelection empty_row_selection(std::uint32_t k) {
    RowSelection s;
    s.k = k;
    return s;
}

// THE PINNED BLOCK IS A BLOCK ID, NOT A SCORE. Deriving it in integers from the visible row count
// is exact for every row count. The equivalent float formula, floor(((visible - 1) / ratio) /
// block_size), is only exact while the ratio and block size are powers of two, and rounding there
// moves the pin onto the wrong block without any visible symptom.

/// Stage one. `scores` is one query's coarse source row: the score of compressed row `r` is
/// `scores[r]`. `visible_rows` is how many leading rows the query may see; rows at or beyond it are
/// not part of any block maximum whatever they score. `block_size` and `blocks_wanted` are model
/// geometry.
///
/// The blocks are the `ceil(row_count / block_size)` blocks over `row_count`, where `row_count` is
/// the visible count clamped to `scores.size()`. Rows past `row_count` are never looked at, so they
/// take no part in any block maximum whatever they score, and `blocks_total` counts the blocks
/// over the rows that exist rather than the blocks the score row could have held.
///
/// The edges:
///   * `visible_rows == 0` or an empty score row: nothing is selected, there is no block to pin,
///     and `blocks_total` is 0 because no block holds a visible row.
///   * `block_size == 0` or `blocks_wanted == 0`: nothing is selected, and `ranking` is empty.
///   * Fewer blocks than asked for: every reachable block is selected, nothing was dropped, and
///     `cut_exact` is false because the block count, not the scores, ended the selection.
///   * A partial tail block covers only the rows that exist, and it is the pinned block whenever it
///     is also the newest one.
///   * Every block unreachable except the pin: the pin is still selected. That is the intended
///     behavior rather than a fallback, which is why `pinned_block_valid` is reported separately.
inline BlockSelection choose_candidate_blocks(std::span<const float> scores,
                                               std::uint32_t visible_rows,
                                               std::uint32_t block_size,
                                               std::uint32_t blocks_wanted) {
    const std::uint32_t row_capacity = static_cast<std::uint32_t>(scores.size());
    const std::uint32_t visible = std::min(visible_rows, row_capacity);
    if (visible == 0 || block_size == 0) {
        // With nothing visible no block holds a row, so the count is 0 rather than the count of
        // blocks the score row could have held: reporting those would invite a reader to size an
        // index from blocks that can never be selected.
        const std::uint32_t total = 0;
        BlockSelection s = empty_block_selection(total, blocks_wanted);
        return s;
    }

    const std::uint32_t row_count = visible;
    // Ceiling by division, not by adding `block_size - 1`: that sum wraps to zero for a block size
    // near the top of the range, which would size the block table at zero and index past it.
    const std::uint32_t blocks_total =
        row_count / block_size + (row_count % block_size != 0 ? 1u : 0u);
    const std::uint32_t pinned = (row_count - 1) / block_size;

    std::vector<float> best(blocks_total, kUnreachableScore);
    for (std::uint32_t r = 0; r < row_count; ++r) {
        // std::max ignores NaN, so a NaN row can never become a block's best row.
        best[r / block_size] = std::max(best[r / block_size], scores[r]);
    }
    // Pinned before ranking: the ascending-block-index tie key resolves a tied score in the pin's
    // favor, and no other score can outrank it.
    best[pinned] = std::numeric_limits<float>::infinity();

    std::vector<std::uint32_t> order(blocks_total);
    for (std::uint32_t b = 0; b < blocks_total; ++b) order[b] = b;
    // A total order, not a partial sort: the score comparison is anti-symmetric because NaN left
    // the keys above, and the id comparison settles every remaining tie.
    std::sort(order.begin(), order.end(), [&best](std::uint32_t a, std::uint32_t b) {
        if (best[a] != best[b]) return best[a] > best[b];
        return a < b;
    });

    BlockSelection s;
    s.row_count = row_count;
    s.visible_rows = visible;
    s.blocks_wanted = blocks_wanted;
    s.blocks_total = blocks_total;
    s.pinned_block = pinned;
    s.pinned_block_valid = true;
    s.ranking.resize(blocks_total);
    for (std::uint32_t i = 0; i < blocks_total; ++i) {
        s.ranking[i].score = best[order[i]];
        s.ranking[i].pinned = (order[i] == pinned);
    }

    // `unreachable_ranks` counts the leading run of unreachable blocks, and nothing else. They sort
    // last, so they are never selected, and they cap the request when the query simply cannot reach
    // that many blocks. Counting unreachable blocks anywhere in the order would wrongly retire a
    // request for more blocks than the query can reach.
    std::uint32_t unreachable_ranks = 0;
    for (std::uint32_t b : order) {
        if (is_unreachable(best[b])) ++unreachable_ranks;
        else break;
    }
    const std::uint32_t want = std::min(blocks_wanted, blocks_total);
    const std::uint32_t keep = std::min(want, blocks_total - unreachable_ranks);

    for (std::uint32_t i = 0; i < keep; ++i) {
        const std::uint32_t b = order[i];
        s.ranking[i].selected = true;
        SelectedBlock sb;
        sb.id = b;
        sb.row_begin = b * block_size;
        // Saturating rather than wrapping: a `block_size` large enough to overflow the product
        // would otherwise hand stage two a range it reads past.
        sb.row_end = block_size > row_count - sb.row_begin ? row_count : sb.row_begin + block_size;
        sb.rows = sb.row_end - sb.row_begin;
        sb.pinned = (b == pinned);
        s.blocks.push_back(sb);
    }
    std::sort(s.blocks.begin(), s.blocks.end(),
              [](const SelectedBlock& a, const SelectedBlock& b) { return a.id < b.id; });
    s.blocks_selected = static_cast<std::uint32_t>(s.blocks.size());
    s.dropped_rank = keep;
    if (keep > 0) s.kept_score = best[order[keep - 1]];
    if (keep < blocks_total) {
        s.dropped_score = best[order[keep]];
        s.cut_margin = s.kept_score - s.dropped_score;
    }
    s.cut_exact = want > 0 && keep == want && keep < blocks_total;
    return s;
}

/// Stage two. `scores` is one query's reader row, which may differ from the coarse source's scores;
/// `blocks` is stage one's `BlockSelection`. A row is a candidate when it lies inside a selected
/// block, and a row outside every selected block is unreachable by construction: stage two never
/// looks at a row it was not allowed to see, whatever that row scored.
///
/// The edges: an empty `blocks` selection or an empty candidate set returns
/// `empty_row_selection(k)`. A `k` above the candidate count keeps every candidate and reports
/// `cut_exact == false`. Rows are read only up to the shorter of the selection's row count and the
/// score row, so a mismatched pair degrades to fewer candidates rather than an out-of-range read.
/// `rows` and `row_scores` are ascending, so a consumer uses them directly as an attention index;
/// the rank tie key does not leak into that ordering.
inline RowSelection choose_top_rows(std::span<const float> scores, const BlockSelection& blocks,
                                    std::uint32_t k) {
    const std::uint32_t row_count = blocks.row_count;
    if (blocks.blocks.empty() || row_count == 0) return empty_row_selection(k);

    std::vector<std::uint32_t> order;
    for (const SelectedBlock& b : blocks.blocks) {
        // A caller may hand over a selection built from a longer row; only the rows this query
        // actually has are candidates, and a range that starts past them contributes nothing.
        if (b.row_begin >= row_count) continue;
        const std::uint32_t end = std::min(b.row_end, row_count);
        for (std::uint32_t r = b.row_begin; r < end && r < scores.size(); ++r) {
            if (!is_unreachable(scores[r])) order.push_back(r);
        }
    }
    RowSelection s;
    s.k = k;
    s.candidates = static_cast<std::uint32_t>(order.size());
    if (order.empty()) return s;

    std::sort(order.begin(), order.end(), [&scores](std::uint32_t a, std::uint32_t b) {
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    });
    const std::uint32_t keep = std::min(k, s.candidates);
    s.rows_selected = keep;
    s.dropped_rank = keep;
    if (keep > 0) s.kept_score = scores[order[keep - 1]];
    if (keep < s.candidates) {
        s.dropped_score = scores[order[keep]];
        s.cut_margin = s.kept_score - s.dropped_score;
    }
    s.cut_exact = k > 0 && keep == k && keep < s.candidates;
    for (std::uint32_t i = 0; i < keep; ++i) s.rows.push_back(order[i]);
    // Ascending row order, with the scores re-read in that order so the two stay parallel.
    std::sort(s.rows.begin(), s.rows.end());
    s.row_scores.reserve(s.rows.size());
    for (std::uint32_t r : s.rows) s.row_scores.push_back(scores[r]);
    return s;
}

/// Convenience composition: stage one on the coarse source row, stage two on the reader row. The
/// two score rows may differ in length; stage two's row count comes from the source row that chose
/// the blocks, so a shorter reader row reads only rows the source proved exist.
inline CandidateSelection select_candidates(std::span<const float> source_scores,
                                            std::span<const float> reader_scores,
                                            std::uint32_t visible_rows, std::uint32_t block_size,
                                            std::uint32_t blocks_wanted, std::uint32_t k) {
    CandidateSelection out;
    out.blocks = choose_candidate_blocks(source_scores, visible_rows, block_size, blocks_wanted);
    out.rows = choose_top_rows(reader_scores, out.blocks, k);
    return out;
}

}  // namespace strata::deepseek_v41
