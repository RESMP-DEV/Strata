// src/deepseek/candidate_select_test.cpp - the host test for the DeepSeek V4.1 candidate-selection
// reference. CPU only: no model, no GPU, no ggml, no engine sources.
//
//   candidate_select_test           run every case, exit 0 when all pass
//
// The expected rows are written out by hand from the contract rather than recomputed by a second
// copy of the same algorithm, so a wrong comparator fails here instead of agreeing with itself. The
// two long-range cases do check against an independent oracle, and say so where they do.
#include "strata/deepseek/candidate_select.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace ds = strata::deepseek_v41;

namespace {

int g_fail = 0;

#define CHECK(cond, ...)                                                 \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);   \
            std::fprintf(stderr, __VA_ARGS__);                           \
            std::fprintf(stderr, "\n");                                  \
            ++g_fail;                                                    \
        }                                                                \
    } while (0)

bool g_detail_printed = false;

/// One line of detail from inside a case; the case's own verdict is still printed afterwards.
void detail(const char* text) {
    std::printf("\n      %s", text);
    g_detail_printed = true;
}

/// Run one named case and print its outcome. Failures are already on stderr with file and line.
template <typename F>
void run_case(const char* name, F&& fn) {
    const int before = g_fail;
    g_detail_printed = false;
    std::printf("  %-44s", name);
    std::fflush(stdout);
    fn();
    if (g_detail_printed) std::printf("\n  %-44s", "");
    std::printf("%s\n", g_fail == before ? "ok" : "FAILED");
}

std::string fmt(const std::vector<std::uint32_t>& v) {
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(v[i]);
    }
    return out.empty() ? std::string("-") : out;
}

std::vector<std::uint32_t> block_ids(const ds::BlockSelection& s) {
    std::vector<std::uint32_t> out;
    for (const ds::SelectedBlock& b : s.blocks) out.push_back(b.id);
    return out;
}

bool ascending(const std::vector<std::uint32_t>& v) {
    return std::is_sorted(v.begin(), v.end()) &&
           std::adjacent_find(v.begin(), v.end()) == v.end();
}

bool in_blocks(const ds::BlockSelection& s, std::uint32_t row) {
    for (const ds::SelectedBlock& b : s.blocks) {
        if (row >= b.row_begin && row < b.row_end) return true;
    }
    return false;
}

/// A plain LCG: the test needs reproducible pseudo-randomness, not a statistical one.
class Lcg {
public:
    explicit Lcg(std::uint64_t seed) : state_(seed) {}
    std::uint32_t next() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint32_t>(state_ >> 33);
    }
    float unit() { return static_cast<float>(next()) / 4294967296.0f; }

private:
    std::uint64_t state_;
};

}  // namespace

// The pin is why blocks exist. A newest block scoring far below the rest would lose on score alone;
// it is kept anyway, and pinning it is the only thing that changes the ranking.
static void test_pin_keeps_the_newest_block() {
    run_case("pin keeps the newest block", [] {
        std::vector<float> src(40);
        for (std::uint32_t r = 0; r < 40; ++r) src[r] = 0.1f * static_cast<float>(r / 8);
        for (std::uint32_t r = 32; r < 40; ++r) src[r] = -100.0f;

        const ds::BlockSelection s = ds::choose_candidate_blocks(src, 40, 8, 3);
        CHECK(s.blocks_total == 5, "blocks_total %u", s.blocks_total);
        CHECK(s.pinned_block == 4 && s.pinned_block_valid, "pinned %u", s.pinned_block);
        // The pin leads at infinite, then 0.3, 0.2 beat block 0 at 0.0. Nothing was dropped by the
        // count, so the cut is not the test's subject here.
        CHECK(fmt(block_ids(s)) == "2,3,4", "blocks %s", fmt(block_ids(s)).c_str());
        CHECK(s.blocks.back().pinned, "the highest block id is the pinned one");
        CHECK(std::abs(s.kept_score - 0.2f) < 1e-6f, "kept_score %f", s.kept_score);
        CHECK(std::abs(s.dropped_score - 0.1f) < 1e-6f, "dropped_score %f", s.dropped_score);
        CHECK(s.dropped_rank == 3 && s.cut_exact, "dropped_rank %u exact %d", s.dropped_rank,
              int(s.cut_exact));
        CHECK(std::abs(s.cut_margin - 0.1f) < 1e-6f, "cut_margin %f", s.cut_margin);
        CHECK(s.ranking.front().pinned && std::isinf(s.ranking.front().score),
              "the pin leads the ranking");
        // The pin is a ranking key only; the block row ranges stay the real geometry.
        CHECK(s.blocks[0].row_begin == 16 && s.blocks[0].row_end == 24, "row range");
    });
}

// Without the pin the top blocks are plain block maxima, ordered by score with the id as the tie
// key.
static void test_top_blocks_by_maximum() {
    run_case("top blocks by best row, ties by id", [] {
        // 48 rows, block size 8: six blocks. One representative value per block.
        const float per_block[6] = {0.5f, 0.9f, 0.1f, 0.9f, 0.7f, 0.2f};
        std::vector<float> src(48);
        for (std::uint32_t b = 0; b < 6; ++b) {
            for (std::uint32_t j = 0; j < 8; ++j) {
                src[b * 8 + j] = per_block[b] - 0.01f * static_cast<float>(j);
            }
        }
        const ds::BlockSelection s = ds::choose_candidate_blocks(src, 48, 8, 4);
        // Block maxima: b1 0.9, b3 0.9, b4 0.7, b0 0.5, b5 0.2, b2 0.1. The newest block (5) is
        // pinned and leads at infinite, then b1 before b3 on the ascending-id tie key.
        CHECK(fmt(block_ids(s)) == "1,3,4,5", "blocks %s", fmt(block_ids(s)).c_str());
        CHECK(std::abs(s.kept_score - 0.7f) < 1e-6f, "kept_score %f", s.kept_score);
        CHECK(std::abs(s.dropped_score - 0.5f) < 1e-6f, "dropped_score %f", s.dropped_score);
        CHECK(s.blocks_selected == 4 && s.blocks_total == 6, "selected %u total %u",
              s.blocks_selected, s.blocks_total);
        // Row ranges must tile the rows without overlapping.
        std::vector<std::uint32_t> lo, hi;
        for (const ds::SelectedBlock& b : s.blocks) {
            lo.push_back(b.row_begin);
            hi.push_back(b.row_end);
        }
        CHECK(fmt(lo) == "8,24,32,40" && fmt(hi) == "16,32,40,48", "ranges %s / %s",
              fmt(lo).c_str(), fmt(hi).c_str());
    });
}

// The reader scores the same rows differently, and only the chosen blocks are reachable. Two rows
// outside the chosen blocks hold the highest reader scores in the whole row; they must still lose.
static void test_rows_restricted_to_blocks() {
    run_case("row top-k restricted to chosen blocks", [] {
        // Blocks 0..5 score 0.10 .. 0.15, block 6 spikes to 0.99, block 7 holds the newest rows and
        // would lose on score alone, so stage one has to take the pin plus the spike.
        std::vector<float> src(64);
        for (std::uint32_t b = 0; b < 8; ++b) {
            const float v = b == 6 ? 0.99f : 0.10f + 0.01f * static_cast<float>(b);
            for (std::uint32_t j = 0; j < 8; ++j) src[b * 8 + j] = v;
        }
        std::vector<float> rd(64, 0.0f);
        const ds::BlockSelection blocks = ds::choose_candidate_blocks(src, 64, 8, 2);
        CHECK(fmt(block_ids(blocks)) == "6,7", "blocks %s", fmt(block_ids(blocks)).c_str());

        for (std::uint32_t r = 48; r < 56; ++r) rd[r] = 10.0f;   // block 6, the winner
        for (std::uint32_t r = 56; r < 64; ++r) rd[r] = 9.0f;    // block 7, the pinned newest
        rd[0] = 100.0f;                                          // dropped block: unreachable
        rd[16] = 99.0f;
        for (std::uint32_t r = 32; r < 40; ++r) rd[r] = 0.0f;    // below the four kept rows

        const ds::RowSelection s = ds::choose_top_rows(rd, blocks, 4);
        CHECK(s.candidates == 16, "candidates %u", s.candidates);
        CHECK(fmt(s.rows) == "48,49,50,51", "rows %s", fmt(s.rows).c_str());
        CHECK(ascending(s.rows), "rows must come back ascending");
        CHECK(s.rows_selected == 4 && s.cut_exact, "selected %u exact %d", s.rows_selected,
              int(s.cut_exact));
        CHECK(std::abs(s.row_scores.front() - 10.0f) < 1e-6f, "front score %f",
              s.row_scores.front());
        for (std::uint32_t r : s.rows) CHECK(in_blocks(blocks, r), "row %u outside the blocks", r);
    });
}

// Equal scores must resolve the same way every time, and the way is the oldest row first. A kernel
// that breaks ties by work-group order agrees here only by luck; the ascending-key rule is the
// contract it has to implement.
static void test_ties_break_toward_the_older_row() {
    run_case("ties resolve to the lowest row id", [] {
        // The tie is between block 0 and the pin, so ascending block id takes 0 even though the pin
        // is newer and outranks on every other rule.
        std::vector<float> src(40, 1.0f);
        for (std::uint32_t r = 8; r < 40; ++r) src[r] = 0.5f;
        const ds::BlockSelection s = ds::choose_candidate_blocks(src, 40, 8, 2);
        CHECK(fmt(block_ids(s)) == "0,4", "blocks %s", fmt(block_ids(s)).c_str());
        CHECK(std::abs(s.dropped_score - 0.5f) < 1e-6f, "the dropped block ties the pinned one %f",
              s.dropped_score);

        // The same rule on rows: every reachable row ties at 0.5, so the lowest ids come back.
        std::vector<float> rd(40, 0.5f);
        const ds::RowSelection rows = ds::choose_top_rows(rd, s, 3);
        CHECK(fmt(rows.rows) == "0,1,2", "rows %s", fmt(rows.rows).c_str());
        CHECK(rows.dropped_rank == 3 && rows.cut_exact, "the cut is decided, not the count");
        CHECK(ascending(rows.rows), "rows ascending");
    });
}

// A partial tail block is real work, not padding: it covers only the rows that exist, and when it
// is also the newest it is the pin. 83 visible rows at block size 8 leave a block of three.
static void test_partial_tail_block() {
    run_case("partial tail block is short and pinned", [] {
        // The tail block (10) holds the newest rows. Every block holds 0.05 and the tail also holds
        // 0.01 at its first row, so blocks 0 and 10 win on score and the tail is pinned besides.
        std::vector<float> src(83, 0.1f);
        for (std::uint32_t r = 0; r < 83; ++r) src[r] = 0.05f;
        src[81] = 0.01f;  // block 10, the partial tail block of three rows
        const ds::BlockSelection s = ds::choose_candidate_blocks(src, 83, 8, 3);
        CHECK(s.blocks_total == 11, "blocks_total %u", s.blocks_total);
        CHECK(s.pinned_block == 10, "pinned %u", s.pinned_block);
        // Blocks 0..9 all hold 0.05, so the tie key takes ids 0 and 1 into the two free slots.
        CHECK(fmt(block_ids(s)) == "0,1,10", "blocks %s", fmt(block_ids(s)).c_str());
        const ds::SelectedBlock& tail = s.blocks.back();
        CHECK(tail.id == 10 && tail.row_begin == 80 && tail.row_end == 83 && tail.rows == 3,
              "tail %u [%u,%u) rows %u", tail.id, tail.row_begin, tail.row_end, tail.rows);
        CHECK(tail.pinned, "the tail block is the pin");
        // Block 1 at 0.05 ties every other block that is not the tail, so the tie key takes id 0.
        CHECK(std::abs(s.kept_score - 0.05f) < 1e-6f, "kept_score %f", s.kept_score);
        CHECK(std::abs(s.dropped_score - 0.05f) < 1e-6f, "dropped_score %f", s.dropped_score);
        CHECK(s.blocks_selected == 3 && s.cut_exact, "selected %u exact %d", s.blocks_selected,
              int(s.cut_exact));
        std::vector<float> rd(83, 0.0f);
        rd[82] = 1.0f;
        const ds::RowSelection rows = ds::choose_top_rows(rd, s, 5);
        CHECK(fmt(rows.rows) == "0,1,2,3,82", "rows %s", fmt(rows.rows).c_str());
    });
}

// A step can end before the geometry asks for a full budget. Every case here returns fewer than
// asked, and each one says why through cut_exact, so a caller can tell a short answer from a cut.
static void test_fewer_blocks_than_requested() {
    run_case("fewer blocks than requested", [] {
        std::vector<float> src(20, 0.2f);
        const ds::BlockSelection s = ds::choose_candidate_blocks(src, 20, 8, 8);
        CHECK(s.blocks_total == 3, "blocks_total %u", s.blocks_total);
        CHECK(fmt(block_ids(s)) == "0,1,2", "blocks %s", fmt(block_ids(s)).c_str());
        CHECK(!s.cut_exact, "the block count, not the scores, ended the selection");
        CHECK(s.dropped_rank == s.blocks_total, "dropped_rank %u", s.dropped_rank);
        CHECK(s.cut_margin == -1.0f, "cut_margin %f", s.cut_margin);

        // Asking for no blocks at all is a legal, empty answer.
        const ds::BlockSelection none = ds::choose_candidate_blocks(src, 20, 8, 0);
        CHECK(none.blocks.empty() && !none.cut_exact, "blocks_wanted 0 selects nothing");
        CHECK(none.blocks_total == 3, "blocks_total %u", none.blocks_total);
    });
}

// A k larger than the candidate set returns every candidate, says the count ended it, and never
// invents a row. The same holds for k above the rows the blocks cover.
static void test_k_overflow_keeps_every_candidate() {
    run_case("k above the allowed rows", [] {
        // Every block ties at 0.5; the newest block (3) is the pin, so block 0 takes the free slot.
        std::vector<float> src(32, 0.5f);
        const ds::BlockSelection blocks = ds::choose_candidate_blocks(src, 32, 8, 2);
        CHECK(fmt(block_ids(blocks)) == "0,3", "blocks %s", fmt(block_ids(blocks)).c_str());
        std::vector<float> rd(32, 0.5f);

        const ds::RowSelection s = ds::choose_top_rows(rd, blocks, 512);
        CHECK(s.candidates == 16, "candidates %u", s.candidates);
        CHECK(s.rows_selected == 16 && s.rows.size() == 16, "selected %u", s.rows_selected);
        CHECK(!s.cut_exact, "the candidate count ended the selection, not the score");
        CHECK(ascending(s.rows), "rows ascending");
        CHECK(s.rows.front() == 0 && s.rows.back() == 31, "range %u..%u", s.rows.front(),
              s.rows.back());
        CHECK(s.row_scores.size() == s.rows.size(), "scores stay parallel to rows");

        // Rows outside the blocks are unreachable however high they score, and k=0 is empty.
        rd[0] = 1e30f;
        const ds::RowSelection again = ds::choose_top_rows(rd, blocks, 512);
        CHECK(again.rows == s.rows, "a masked-out row cannot enter through k");
        const ds::RowSelection zero = ds::choose_top_rows(rd, blocks, 0);
        // Candidates were still counted, so the caller can see that k, not the input, emptied it.
        CHECK(zero.rows.empty() && zero.candidates == 16 && !zero.cut_exact,
              "k=0 returns no rows (candidates %u exact %d)", zero.candidates, int(zero.cut_exact));
    });
}

// Nothing visible, nothing masked, nothing NaN, nothing surprising. Each degenerate input has to
// return a well-formed empty answer instead of reading out of bounds.
static void test_degenerate_inputs() {
    run_case("empty and masked inputs", [] {
        const std::vector<float> src(32, 0.5f);

        const ds::BlockSelection no_vis = ds::choose_candidate_blocks(src, 0, 8, 4);
        CHECK(no_vis.blocks.empty() && !no_vis.pinned_block_valid, "visible 0 selects nothing");
        CHECK(no_vis.blocks_total == 0 && no_vis.blocks_wanted == 4, "total %u want %u",
              no_vis.blocks_total, no_vis.blocks_wanted);
        const ds::RowSelection no_rows = ds::choose_top_rows(src, no_vis, 8);
        CHECK(no_rows.rows.empty() && no_rows.k == 8, "no blocks means no rows");

        const std::vector<float> nothing;
        const ds::BlockSelection no_scores = ds::choose_candidate_blocks(nothing, 16, 8, 4);
        CHECK(no_scores.blocks.empty() && no_scores.blocks_total == 0, "empty row selects nothing");
        CHECK(ds::choose_candidate_blocks(nothing, 0, 8, 4).blocks.empty(), "both empty");

        // visible beyond the scores clamps to the scores rather than reading past them.
        const ds::BlockSelection clamped = ds::choose_candidate_blocks(src, 9999, 8, 4);
        CHECK(clamped.row_count == 32 && clamped.visible_rows == 32, "row_count %u",
              clamped.row_count);

        // block_size 0 has no blocks to pool; nothing is selected and nothing is invented.
        const ds::BlockSelection no_blocks = ds::choose_candidate_blocks(src, 32, 0, 4);
        CHECK(no_blocks.blocks.empty() && no_blocks.blocks_total == 0, "block_size 0");

        // Every row masked (-inf): all four blocks tie at the mask value, so the ascending-id tie
        // key picks 0..2 and the pin 3 is the only thing that puts block 3 in the answer. This is
        // the documented behavior, not a fallback, so the test pins it down.
        std::vector<float> masked(32, ds::kMaskedScore);
        const ds::BlockSelection all_masked = ds::choose_candidate_blocks(masked, 32, 8, 4);
        CHECK(fmt(block_ids(all_masked)) == "0,1,2,3", "all-masked blocks %s",
              fmt(block_ids(all_masked)).c_str());
        CHECK(all_masked.blocks[3].pinned && all_masked.pinned_block_valid, "the pin survives");
        const ds::RowSelection masked_rows = ds::choose_top_rows(masked, all_masked, 4);
        CHECK(masked_rows.rows.empty() && masked_rows.candidates == 0, "masked rows unreachable");

        // A NaN row never becomes a block maximum, so its block still competes on its other rows.
        std::vector<float> nans(32, 0.5f);
        nans[0] = std::numeric_limits<float>::quiet_NaN();
        nans[9] = std::numeric_limits<float>::quiet_NaN();
        const ds::BlockSelection with_nan = ds::choose_candidate_blocks(nans, 32, 8, 4);
        CHECK(fmt(block_ids(with_nan)) == "0,1,2,3", "NaN does not sink its block %s",
              fmt(block_ids(with_nan)).c_str());
        // A block whose only row is NaN scores no better than an unreachable one, so the tie key
        // orders it against the masked floor by id and it still takes the last free slot.
        std::vector<float> nan_only(32, 0.5f);
        for (std::uint32_t r = 8; r < 16; ++r) {
            nan_only[r] = std::numeric_limits<float>::quiet_NaN();
        }
        const ds::BlockSelection sunk = ds::choose_candidate_blocks(nan_only, 32, 8, 4);
        CHECK(fmt(block_ids(sunk)) == "0,1,2,3", "a NaN-only block ranks last %s",
              fmt(block_ids(sunk)).c_str());
        // Prove it is really the last one to be considered: asking for three drops it.
        const ds::BlockSelection kept = ds::choose_candidate_blocks(nan_only, 32, 8, 3);
        CHECK(fmt(block_ids(kept)) == "0,2,3", "a NaN-only block is dropped first %s",
              fmt(block_ids(kept)).c_str());

        // A block size large enough to overflow the row range must still produce a readable range.
        const ds::BlockSelection huge = ds::choose_candidate_blocks(src, 32, 0xFFFFFFFFu, 2);
        CHECK(huge.blocks.size() == 1, "one block covers everything");
        CHECK(huge.blocks[0].row_begin == 0 && huge.blocks[0].row_end == 32, "range %u..%u",
              huge.blocks[0].row_begin, huge.blocks[0].row_end);
        const ds::RowSelection huge_rows = ds::choose_top_rows(src, huge, 3);
        CHECK(fmt(huge_rows.rows) == "0,1,2", "rows %s", fmt(huge_rows.rows).c_str());

        // A selection built from a longer row, presented to a shorter score row: only the rows this
        // score row has are candidates, and nothing is read past its end.
        ds::BlockSelection truncated = ds::choose_candidate_blocks(src, 32, 8, 4);
        truncated.row_count = 4;
        const std::vector<float> tiny(4, 0.5f);
        const ds::RowSelection short_rows = ds::choose_top_rows(tiny, truncated, 2);
        CHECK(fmt(short_rows.rows) == "0,1", "rows %s", fmt(short_rows.rows).c_str());
        CHECK(short_rows.candidates == 4, "candidates %u", short_rows.candidates);
        truncated.row_count = 0;
        CHECK(ds::choose_top_rows(tiny, truncated, 2).rows.empty(), "no rows, no candidates");
    });
}

namespace oracle {

/// A deliberately naive brute force for the long cases, written from the contract and sharing no
/// code with the header: explicit scanning, an O(n^2) rank search, and no sort-based partial order.
/// If the two agree on a long random case, the header is not merely self-consistent.
struct Answer {
    std::vector<std::uint32_t> blocks;
    std::vector<std::uint32_t> rows;
};

bool better(float score_a, std::uint32_t id_a, float score_b, std::uint32_t id_b) {
    if (score_a != score_b) return score_a > score_b;   // NaN never reaches here: masked below
    return id_a < id_b;
}

Answer solve(const std::vector<float>& src, const std::vector<float>& rd, std::uint32_t visible,
             std::uint32_t block_size, std::uint32_t blocks_wanted, std::uint32_t k) {
    Answer out;
    const std::uint32_t nb = (visible + block_size - 1) / block_size;
    std::vector<float> best(nb, -std::numeric_limits<float>::infinity());
    for (std::uint32_t r = 0; r < visible; ++r) {
        const float s = src[r];
        const std::uint32_t b = r / block_size;
        if (s > best[b]) best[b] = s;                    // strictly greater, so NaN never wins
    }
    best[(visible - 1) / block_size] = std::numeric_limits<float>::infinity();

    std::vector<bool> taken(nb, false);
    const std::uint32_t want = std::min(blocks_wanted, nb);
    for (std::uint32_t i = 0; i < want; ++i) {
        std::uint32_t best_id = nb;
        float best_score = -std::numeric_limits<float>::infinity();
        for (std::uint32_t b = 0; b < nb; ++b) {
            if (taken[b]) continue;
            if (best_id == nb || better(best[b], b, best_score, best_id)) {
                best_id = b;
                best_score = best[b];
            }
        }
        taken[best_id] = true;
        if (best_score > -std::numeric_limits<float>::infinity()) out.blocks.push_back(best_id);
    }
    std::sort(out.blocks.begin(), out.blocks.end());

    std::vector<std::uint32_t> order;
    for (std::uint32_t r = 0; r < visible; ++r) {
        if (!taken[r / block_size]) continue;
        if (!(rd[r] > -std::numeric_limits<float>::infinity())) continue;
        order.push_back(r);
    }
    for (std::size_t i = 0; i < order.size(); ++i) {
        for (std::size_t j = i + 1; j < order.size(); ++j) {
            if (better(rd[order[j]], order[j], rd[order[i]], order[i])) {
                std::swap(order[i], order[j]);
            }
        }
    }
    const std::uint32_t keep = std::min(k, static_cast<std::uint32_t>(order.size()));
    for (std::uint32_t i = 0; i < keep; ++i) out.rows.push_back(order[i]);
    std::sort(out.rows.begin(), out.rows.end());
    return out;
}

}  // namespace oracle

// Visibility past 16,384 rows, where a real indexer has thousands of blocks and the selection is
// decided among more candidates than fit in one launch. Two cases: V4.1's decode geometry over
// 40,000 rows, and a 32,768-row prompt with a partial tail block. Both are checked against the
// brute-force oracle above, so this is agreement between two implementations, not a tautology.
static void test_visibility_beyond_16384_rows() {
    run_case("visibility past 16,384 rows", [] {
        struct Case {
            const char* label;
            std::uint32_t visible;
            std::uint32_t block_size;
            std::uint32_t blocks_wanted;
            std::uint32_t k;
            std::uint64_t seed;
        };
        const Case cases[] = {
            {"40000 rows, block 8, 2048 blocks, k 512", 40000, 8, 2048, 512, 4141},
            {"32768 rows, block 16, 1024 blocks, k 2048", 32768, 16, 1024, 2048, 9001},
            {"17000 rows, block 32, 300 blocks, k 64", 17000, 32, 300, 64, 70003},
        };
        for (const Case& c : cases) {
            Lcg rng(c.seed);
            std::vector<float> src(c.visible);
            std::vector<float> rd(c.visible);
            for (std::uint32_t r = 0; r < c.visible; ++r) {
                // A repeating pattern plus jitter: ties do happen, and must still resolve.
                src[r] = 0.5f * std::sin(static_cast<float>(r % 977)) + 0.5f + 0.01f * rng.unit();
                rd[r] = 0.5f * std::cos(static_cast<float>(r % 613)) + 0.5f + 0.01f * rng.unit();
            }
            const ds::CandidateSelection got =
                ds::select_candidates(src, rd, c.visible, c.block_size, c.blocks_wanted, c.k);
            const oracle::Answer want =
                oracle::solve(src, rd, c.visible, c.block_size, c.blocks_wanted, c.k);
            char line[160];
            std::snprintf(line, sizeof(line), "%s: %u blocks of %u, %u kept of %u candidates",
                          c.label, c.visible / c.block_size, c.block_size,
                          got.rows.rows_selected, got.rows.candidates);
            detail(line);
            CHECK(block_ids(got.blocks) == want.blocks, "%s: blocks differ from the oracle",
                  c.label);
            CHECK(got.rows.rows == want.rows, "%s: rows differ from the oracle", c.label);
            CHECK(got.rows.rows.size() == c.k, "%s: kept %zu of %u", c.label, got.rows.rows.size(),
                  c.k);
            CHECK(ascending(block_ids(got.blocks)), "%s: block ids ascend", c.label);
            CHECK(ascending(got.rows.rows), "%s: row ids ascend", c.label);
            const bool pinned_chosen =
                std::find(block_ids(got.blocks).begin(), block_ids(got.blocks).end(),
                          got.blocks.pinned_block) != block_ids(got.blocks).end();
            CHECK(pinned_chosen, "%s: the newest block %u is not among the chosen blocks", c.label,
                  got.blocks.pinned_block);
            CHECK(got.blocks.pinned_block == (c.visible - 1) / c.block_size,
                  "%s: the pin is the newest block", c.label);
        }
    });
}

// The composition callers actually use, at a decode-step geometry: 2,048 rows of coarse score pick
// 64 blocks, a reader picks 512 rows inside them, and every row it returns lands in a chosen block.
static void test_composed_decode_step() {
    run_case("composed decode step", [] {
        Lcg rng(20261018);
        constexpr std::uint32_t visible = 2048;
        constexpr std::uint32_t block_size = 8;
        constexpr std::uint32_t blocks_wanted = 64;
        constexpr std::uint32_t k = 512;
        std::vector<float> src(visible);
        std::vector<float> rd(visible);
        for (std::uint32_t r = 0; r < visible; ++r) {
            src[r] = rng.unit();
            rd[r] = rng.unit();
        }
        const ds::CandidateSelection s =
            ds::select_candidates(src, rd, visible, block_size, blocks_wanted, k);
        CHECK(s.blocks.blocks_selected == blocks_wanted, "blocks %u",
              s.blocks.blocks_selected);
        CHECK(s.blocks.blocks_total == visible / block_size, "total %u", s.blocks.blocks_total);
        CHECK(s.rows.rows_selected == k && s.rows.candidates == blocks_wanted * block_size,
              "rows %u of %u candidates", s.rows.rows_selected, s.rows.candidates);
        // k equals the candidate count, so the count ended the row cut; the block cut was wider.
        CHECK(!s.rows.cut_exact && s.blocks.cut_exact, "row cut %d block cut %d",
              int(s.rows.cut_exact), int(s.blocks.cut_exact));
        for (std::uint32_t r : s.rows.rows) {
            CHECK(in_blocks(s.blocks, r), "row %u escaped the chosen blocks", r);
        }
        // A decided cut means the dropped competitor really is lower, which is what makes the
        // margin usable to decide whether a near tie invalidates a comparison. With no dropped row
        // there is no boundary and the margin says so.
        CHECK(s.rows.cut_margin == -1.0f, "row cut_margin %f", s.rows.cut_margin);
        CHECK(s.blocks.cut_margin > 0.0f, "block cut_margin %f", s.blocks.cut_margin);
        // Running twice returns the same thing: there is no iteration-order dependence.
        const ds::CandidateSelection again =
            ds::select_candidates(src, rd, visible, block_size, blocks_wanted, k);
        CHECK(block_ids(again.blocks) == block_ids(s.blocks), "block selection is not stable");
        CHECK(again.rows.rows == s.rows.rows, "row selection is not stable");
    });
}

int main() {
    std::printf("DeepSeek V4.1 candidate selection, CPU reference\n");
    test_pin_keeps_the_newest_block();
    test_top_blocks_by_maximum();
    test_rows_restricted_to_blocks();
    test_ties_break_toward_the_older_row();
    test_partial_tail_block();
    test_fewer_blocks_than_requested();
    test_k_overflow_keeps_every_candidate();
    test_degenerate_inputs();
    test_visibility_beyond_16384_rows();
    test_composed_decode_step();
    if (g_fail != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("all cases passed\n");
    return 0;
}
