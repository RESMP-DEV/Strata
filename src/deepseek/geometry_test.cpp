// src/deepseek/geometry_test.cpp - the DeepSeek V4.1 geometry contract test.
//
// Standalone and deterministic: no checkpoint, no network, no accelerator, no
// fixture files, no clock. It exercises the public contract of
// strata::deepseek_v41 (constants, derived helpers, both loaders, validation
// and its errors), not any internal implementation detail.
//
// Build and run:
//     clang++ -std=c++20 -Iinclude src/deepseek/geometry_test.cpp -o /tmp/deepseek_geometry_test
//     /tmp/deepseek_geometry_test
//
// NON-CLAIMS. A pass proves the pinned DeepSeek V4.1-Flash-1017 shape is
// encoded and validated exactly as specified. It proves nothing about any
// executable DeepSeek path, pack format, kernel, or hardware.

#include "strata/deepseek/geometry.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

namespace ds = strata::deepseek_v41;

namespace {

int checks = 0;

void check(bool value, const std::string& description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description.c_str());
        std::exit(1);
    }
}

/// Compares a constant span with an expected list without relying on a
/// vector/span comparison operator.
template <typename T>
bool spans_equal(std::span<const T> span, std::initializer_list<T> expected) {
    return std::size(span) == expected.size() &&
           std::equal(span.begin(), span.end(), expected.begin());
}

std::string join(const std::vector<ds::ValidationError>& errors) {
    std::string out;
    for (const auto& error : errors) {
        if (!out.empty()) {
            out += " | ";
        }
        out += error.field + ": " + error.detail;
    }
    return out;
}

const ds::ValidationError* find(const std::vector<ds::ValidationError>& errors,
                                const std::string& field) {
    for (const auto& error : errors) {
        if (error.field == field) {
            return &error;
        }
    }
    return nullptr;
}

/// Asserts that `errors` is empty and reports every entry when it is not.
void check_clean(const std::vector<ds::ValidationError>& errors, const std::string& what) {
    if (!errors.empty()) {
        std::fprintf(stderr, "FAIL: %s rejected: %s\n", what.c_str(), join(errors).c_str());
        std::exit(1);
    }
    ++checks;
}

/// Asserts that `field` is the first reported error and that its detail names
/// the given substrings, so an error can never degrade to "something is wrong".
/// Later entries are allowed: a wrong list can break several invariants at once,
/// and the extra findings are separately correct.
void check_first_error(const std::vector<ds::ValidationError>& errors, const std::string& field,
                       const std::string& needle_a, const std::string& needle_b,
                       const std::string& what) {
    if (errors.empty()) {
        std::fprintf(stderr, "FAIL: %s expected an error on '%s', got none\n", what.c_str(),
                     field.c_str());
        std::exit(1);
    }
    const ds::ValidationError& error = errors.front();
    if (error.field != field) {
        std::fprintf(stderr, "FAIL: %s expected field '%s', got '%s'\n", what.c_str(),
                     field.c_str(), error.field.c_str());
        std::exit(1);
    }
    const bool a = needle_a.empty() || error.detail.find(needle_a) != std::string::npos;
    const bool b = needle_b.empty() || error.detail.find(needle_b) != std::string::npos;
    if (!a || !b) {
        std::fprintf(stderr, "FAIL: %s detail '%s' lacks '%s' / '%s'\n", what.c_str(),
                     error.detail.c_str(), needle_a.c_str(), needle_b.c_str());
        std::exit(1);
    }
    ++checks;
}

// ------------------------------------------------------------- valid cases

void test_constants() {
    check(std::string(ds::MODEL_TYPE) == "deepseek_v41", "model type is deepseek_v41");
    check(ds::kTextLayers == 40, "40 text layers");
    check(ds::kDsparkTailLayers == 3, "3 DSpark tail layers");
    check(ds::kTotalTextLayers == 43, "43 text layer entries in total");
    check(ds::kHiddenSize == 5120, "hidden 5120");
    check(ds::kAttentionHeads == 64, "64 attention heads");
    check(ds::kKvHeads == 1, "1 KV head");
    check(ds::kHeadDim == 512, "head dim 512");
    check(ds::kQkRopeDim == 64, "qk rope dim 64");
    check(ds::kQLoraRank == 1280, "q lora 1280");
    check(ds::kOLoraRank == 1024, "o lora 1024");
    check(ds::kOutputGroups == 8, "8 output groups");
    check(ds::kRoutedExperts == 384, "384 routed experts");
    check(ds::kRoutedExpertsPerToken == 6, "top-6 routing");
    check(ds::kSharedExperts == 1, "one shared expert");
    check(ds::kMoeIntermediateSize == 2304, "moe intermediate 2304");
    check(ds::kSlidingWindow == 128, "SWA 128");
    check(ds::kIndexHeadCount == 32, "indexer 32 heads");
    check(ds::kIndexHeadDim == 128, "indexer head dim 128");
    check(ds::kIndexTopK == 512, "indexer top-k 512");
    check(ds::kCandidateSourceLayer == 20, "candidate source layer 20");
    check(ds::kCandidateBlocks == 2048, "2048 candidate blocks");
    check(ds::kCandidateBlockSize == 8, "candidate block size 8");
    check(ds::kCandidateRowSpan == 2048 * 8, "candidate span is blocks times block size");
    check(ds::kHcMultiplier == 4, "hc multiplier 4");
    check(ds::kHcSinkhornIterations == 20, "20 Sinkhorn iterations");
    check(ds::kDsparkLayers == 3, "3 DSpark layers");
    check(ds::kDsparkRoutedExperts == 128, "DSpark 128 routed experts");
    check(ds::kDsparkRoutedExpertsPerToken == 3, "DSpark top-3");
    check(ds::kDsparkBlockSize == 5, "DSpark block 5");
    check(ds::kVisionLayers == 32, "vision 32 layers");
    check(ds::kVisionHiddenSize == 1024, "vision hidden 1024");
    check(ds::kMaxContextTokens == 1048576, "1,048,576 max context");
    check(ds::kRopeType == "yarn", "yarn rope");
    check(ds::kRopeYarnFactor == 16.0f, "yarn factor 16");
    check(ds::kRopeOriginalContext == 65536, "yarn original context 65536");
    check(ds::kEngramLayers == 2, "2 Engram layers");
    check(ds::kEngramRows0 == 384006168 && ds::kEngramRows1 == 384016682, "Engram table rows");
    check(ds::kEngramRows0 + ds::kEngramRows1 == 768022850, "Engram rows total 768,022,850");
    check(ds::kTopKMethod == "noaux_tc" && ds::kScoringFunc == "sqrtsoftplus",
          "router method and scoring function");

    check(std::size(ds::kCompressRatios) == 43, "compress table has exactly 43 entries");
    check(std::size(ds::kKvSourceLayers) == 4, "4 kv source layers");
    check(std::size(ds::kIndexSourceLayers) == 8, "8 index source layers");
    check(std::size(ds::kEngramLayerIds) == 2, "2 Engram layer ids");
    check(ds::compress_ratios_view().size() == 43, "compress view covers 43 entries");
    check(ds::backbone_compress_ratios_view().size() == 40, "backbone view covers 40 entries");
    check(ds::backbone_compress_ratios_view().back() == 1, "backbone view stops before the DSpark tail");
}

/// The published pattern: [0, 0, eighteen 2s, twenty 1s, three 0s], checked
/// entry by entry against the stated positions, not against the same constant.
void test_compress_pattern() {
    const auto ratios = ds::compress_ratios_view();
    check(ratios.size() == 43, "43 compress entries");
    for (std::size_t layer = 0; layer < 43; ++layer) {
        const std::uint32_t expected =
            (layer < 2 || layer >= 40) ? 0u : (layer < 20 ? 2u : 1u);
        if (ratios[layer] != expected) {
            std::fprintf(stderr, "FAIL: compress_ratios[%zu] is %u, expected %u\n", layer,
                         ratios[layer], expected);
            std::exit(1);
        }
    }
    check(ratios[0] == 0 && ratios[1] == 0, "backbone layers 0 and 1 do not compress");
    check(ratios[2] == 2 && ratios[19] == 2, "backbone layers 2..19 compress by 2");
    check(ratios[20] == 1 && ratios[39] == 1, "backbone layers 20..39 compress by 1");
    check(ratios[40] == 0 && ratios[41] == 0 && ratios[42] == 0, "DSpark tail layers do not compress");

    std::size_t zeros = 0;
    std::size_t ones = 0;
    std::size_t twos = 0;
    for (std::uint32_t ratio : ratios) {
        zeros += (ratio == 0);
        ones += (ratio == 1);
        twos += (ratio == 2);
    }
    check(zeros == 5 && ones == 20 && twos == 18, "four leading and three trailing zeros, 18 twos, 20 ones");
}

void test_source_lists() {
    const auto kv = ds::kv_source_layers_view();
    check(spans_equal(kv, { 2, 8, 14, 20 }), "kv sources are [2,8,14,20]");
    const auto index = ds::index_source_layers_view();
    check(spans_equal(index, { 2, 8, 14, 20, 24, 28, 32, 36 }),
          "index sources are [2,8,14,20,24,28,32,36]");
    const auto engram = ds::engram_layer_ids_view();
    check(spans_equal(engram, { 1, 14 }), "Engram layers are [1,14]");

    const ds::Geometry g = ds::default_v41_flash();
    check(g.is_kv_source(2) && g.is_kv_source(20), "layers 2 and 20 own compressed rows");
    check(!g.is_kv_source(3) && !g.is_kv_source(19), "layers 3 and 19 read shared rows");
    check(g.is_index_source(24) && g.is_index_source(36), "layers 24 and 36 score shared keys");
    check(!g.is_index_source(23), "layer 23 does not score shared keys");
    check(g.is_engram_layer(1) && g.is_engram_layer(14), "layers 1 and 14 carry Engram tables");
    check(!g.is_engram_layer(2), "layer 2 carries no Engram table");
    check(g.total_source_count() == 12, "4 kv sources plus 8 index sources");
}

void test_derived_helpers() {
    const ds::Geometry g = ds::default_v41_flash();
    check(g.total_text_layers() == 43, "backbone plus the DSpark tail is 43 layers");
    check(g.is_backbone_layer(0) && g.is_backbone_layer(39), "layers 0 and 39 are backbone");
    check(!g.is_backbone_layer(40), "layer 40 is not a backbone layer");
    check(g.is_dspark_tail_layer(40) && g.is_dspark_tail_layer(42), "layers 40..42 are the DSpark tail");
    check(!g.is_dspark_tail_layer(39) && !g.is_dspark_tail_layer(43), "layers 39 and 43 are outside the DSpark tail");
    check(g.backbone_compress_ratio(0) == 0, "backbone layer 0 has ratio 0");
    check(g.backbone_compress_ratio(19) == 2, "backbone layer 19 has ratio 2");
    check(g.backbone_compress_ratio(39) == 1, "backbone layer 39 has ratio 1");
    check(g.backbone_compress_ratio(41) == 0, "an DSpark tail layer has ratio 0");
    check(g.backbone_compress_ratio(43) == 0, "an out-of-range layer id is ratio 0");
    check(g.backbone_compress_ratio(1000000) == 0, "a far out-of-range layer id is ratio 0");

    check(g.routed_experts_per_token_total() == 7, "text layer computes 6 routed plus 1 shared");
    check(g.text_experts_per_layer() == 385, "text layer stores 384 routed plus 1 shared");
    check(g.dspark_experts_per_token_total() == 3, "DSpark layer computes 3 routed experts");
    check(
        g.dspark_experts_per_layer() == 129,
        "DSpark layer stores 128 routed experts plus one shared expert");
    check(
        g.total_expert_count() == 40 * 385 + 3 * 129,
        "expert count covers 40 text layers and the routed plus shared experts in 3 DSpark layers");
    check(g.total_engram_rows() == 768022850, "Engram rows total");
    check(g.engram_rows_for(1) == 384006168, "Engram rows for layer 1");
    check(g.engram_rows_for(14) == 384016682, "Engram rows for layer 14");
    check(g.engram_rows_for(2) == 0, "a layer without a table reports 0 rows");

    // The derived invariants the contract depends on.
    check(64 % 1 == 0, "one KV head divides the query heads");
    check(64 % 8 == 0, "8 output groups divide the query heads");
    // The query bottleneck is wide: 64 heads x 512 head dim is 32768, which is
    // why the low-rank path factors it through q_lora 1280.
    check(ds::kHeadDim * ds::kAttentionHeads == 32768, "the query path expands to 32768");
    check(ds::kQLoraRank > 0 && ds::kQLoraRank < ds::kHeadDim * ds::kAttentionHeads,
          "q_lora 1280 is the bottleneck below the expanded query");
    check(ds::kOLoraRank > 0 && ds::kOLoraRank < ds::kHiddenSize,
          "o_lora 1024 is the bottleneck below hidden 5120");
    check(ds::kHeadDim * ds::kAttentionHeads / ds::kOutputGroups == 4096,
          "the grouped output projection sees 4096 rows per group");
    check(ds::kRoutedExpertsPerToken <= ds::kRoutedExperts, "top-6 fits inside 384 experts");
    check(ds::kRopeOriginalContext * static_cast<std::int64_t>(ds::kRopeYarnFactor) ==
              ds::kMaxContextTokens,
          "65536 times 16 is the 1M context");
}

void test_loaders_accept_the_default() {
    const ds::Geometry g = ds::default_v41_flash();
    check_clean(ds::validate(g), "the default geometry");

    const ds::Manifest m = ds::default_manifest_v41_flash();
    check_clean(ds::validate_manifest(m), "the default manifest");

    std::string error = "unset";
    const ds::Geometry built = ds::from_manifest(m, error);
    check(error.empty(), "from_manifest reports no error for the default manifest");
    check_clean(ds::validate(built), "the geometry built from the default manifest");

    // The built geometry carries every manifest value through unchanged.
    check(built.hidden_size == m.hidden_size && built.head_dim == m.head_dim &&
              built.routed_experts == m.routed_experts &&
              built.max_context_tokens == m.max_context_tokens,
          "from_manifest copies the scalars");
    check(built.compress_ratios == m.compress_ratios, "from_manifest copies the compress table");
    check(built.kv_source_layers == m.kv_source_layers, "from_manifest copies the kv sources");
    check(built.index_source_layers == m.index_source_layers,
          "from_manifest copies the index sources");
    check(built.engram_layer_ids == m.engram_layer_ids, "from_manifest copies the Engram layers");
    check(built.engram_rows == m.engram_rows, "from_manifest copies the Engram rows");
    check(built.top_k_method == m.top_k_method && built.scoring_func == m.scoring_func,
          "from_manifest copies the router method and scoring function");
    check(ds::validate_manifest(m).empty(), "validation is repeatable and side-effect free");
}

// ----------------------------------------------------------- invalid cases

/// Every scalar the contract pins, each mutated in turn through the manifest
/// loader, must be rejected on its own field with both values named.
void test_scalar_rejections() {
    struct Case {
        const char* name;
        void (*mutate)(ds::Manifest&);
    };
    const Case cases[] = {
        { "model_type", [](ds::Manifest& m) { m.model_type = "deepseek_v4"; } },
        { "text_layers", [](ds::Manifest& m) { m.text_layers = 48; } },
        { "hidden_size", [](ds::Manifest& m) { m.hidden_size = 2560; } },
        { "attention_heads", [](ds::Manifest& m) { m.attention_heads = 32; } },
        { "kv_heads", [](ds::Manifest& m) { m.kv_heads = 8; } },
        { "head_dim", [](ds::Manifest& m) { m.head_dim = 256; } },
        { "qk_rope_dim", [](ds::Manifest& m) { m.qk_rope_dim = 128; } },
        { "q_lora_rank", [](ds::Manifest& m) { m.q_lora_rank = 1024; } },
        { "o_lora_rank", [](ds::Manifest& m) { m.o_lora_rank = 512; } },
        { "output_groups", [](ds::Manifest& m) { m.output_groups = 4; } },
        { "sliding_window", [](ds::Manifest& m) { m.sliding_window = 4096; } },
        { "index_head_count", [](ds::Manifest& m) { m.index_head_count = 64; } },
        { "index_head_dim", [](ds::Manifest& m) { m.index_head_dim = 64; } },
        { "index_top_k", [](ds::Manifest& m) { m.index_top_k = 1024; } },
        { "candidate_blocks", [](ds::Manifest& m) { m.candidate_blocks = 1024; } },
        { "candidate_block_size", [](ds::Manifest& m) { m.candidate_block_size = 16; } },
        { "candidate_source_layer", [](ds::Manifest& m) { m.candidate_source_layer = 21; } },
        { "hc_multiplier", [](ds::Manifest& m) { m.hc_multiplier = 2; } },
        { "hc_sinkhorn_iterations", [](ds::Manifest& m) { m.hc_sinkhorn_iterations = 10; } },
        { "routed_experts", [](ds::Manifest& m) { m.routed_experts = 256; } },
        { "routed_experts_per_token", [](ds::Manifest& m) { m.routed_experts_per_token = 8; } },
        { "shared_experts", [](ds::Manifest& m) { m.shared_experts = 0; } },
        { "moe_intermediate_size", [](ds::Manifest& m) { m.moe_intermediate_size = 2048; } },
        { "routed_scaling_factor", [](ds::Manifest& m) { m.routed_scaling_factor = 2.0f; } },
        { "top_k_method", [](ds::Manifest& m) { m.top_k_method = "greedy"; } },
        { "scoring_func", [](ds::Manifest& m) { m.scoring_func = "softmax"; } },
        { "engram_max_ngram", [](ds::Manifest& m) { m.engram_max_ngram = 3; } },
        { "engram_head_dim", [](ds::Manifest& m) { m.engram_head_dim = 128; } },
        { "engram_heads", [](ds::Manifest& m) { m.engram_heads = 4; } },
        { "dspark_layers", [](ds::Manifest& m) { m.dspark_layers = 2; } },
        { "dspark_routed_experts", [](ds::Manifest& m) { m.dspark_routed_experts = 64; } },
        { "dspark_routed_experts_per_token",
          [](ds::Manifest& m) { m.dspark_routed_experts_per_token = 2; } },
        { "dspark_block_size", [](ds::Manifest& m) { m.dspark_block_size = 4; } },
        { "vision_layers", [](ds::Manifest& m) { m.vision_layers = 24; } },
        { "vision_hidden_size", [](ds::Manifest& m) { m.vision_hidden_size = 2048; } },
        { "vision_attention_heads", [](ds::Manifest& m) { m.vision_attention_heads = 32; } },
        { "vision_intermediate_size", [](ds::Manifest& m) { m.vision_intermediate_size = 4096; } },
        { "vision_patch_size", [](ds::Manifest& m) { m.vision_patch_size = 16; } },
        { "vision_downsample_ratio", [](ds::Manifest& m) { m.vision_downsample_ratio = 2; } },
        { "vision_max_image_tokens", [](ds::Manifest& m) { m.vision_max_image_tokens = 512; } },
        { "max_context_tokens", [](ds::Manifest& m) { m.max_context_tokens = 65536; } },
        { "rope_type", [](ds::Manifest& m) { m.rope_type = "linear"; } },
        { "rope_yarn_factor", [](ds::Manifest& m) { m.rope_yarn_factor = 8.0f; } },
        { "rope_yarn_beta_fast", [](ds::Manifest& m) { m.rope_yarn_beta_fast = 16.0f; } },
        { "rope_yarn_beta_slow", [](ds::Manifest& m) { m.rope_yarn_beta_slow = 2.0f; } },
        { "rope_original_context", [](ds::Manifest& m) { m.rope_original_context = 32768; } },
        { "compress_rope_theta", [](ds::Manifest& m) { m.compress_rope_theta = 10000.0f; } },
        { "rms_norm_eps", [](ds::Manifest& m) { m.rms_norm_eps = 1.0e-6f; } },
        { "swiglu_limit", [](ds::Manifest& m) { m.swiglu_limit = 7.0f; } },
        { "dspark_tail_layers", [](ds::Manifest& m) { m.dspark_tail_layers = 0; } },
    };

    for (const Case& c : cases) {
        ds::Manifest m = ds::default_manifest_v41_flash();
        c.mutate(m);
        const auto errors = ds::validate_manifest(m);
        if (errors.empty()) {
            std::fprintf(stderr, "FAIL: mutated %s was accepted\n", c.name);
            std::exit(1);
        }
        bool named = false;
        for (const auto& error : errors) {
            if (error.field == c.name) {
                named = true;
            }
        }
        if (!named) {
            std::fprintf(stderr, "FAIL: mutated %s was not reported on its own field: %s\n",
                         c.name, join(errors).c_str());
            std::exit(1);
        }
        ++checks;
        std::string error_text;
        (void) ds::from_manifest(m, error_text);
        check(!error_text.empty() && error_text.rfind(c.name, 0) == 0,
              std::string("from_manifest surfaces the first failure of ") + c.name);
    }
}

/// The compression table is checked per entry, so a single wrong layer names
/// that layer and the value it needed, in addition to the whole-table diff.
void test_compress_pattern_rejections() {
    struct Case {
        std::size_t layer;
        std::uint32_t value;
        const char* description;
    };
    const Case cases[] = {
        { 1, 2, "layer 1 must not compress" },
        { 2, 1, "layer 2 must compress by 2" },
        { 10, 1, "a middle ratio-2 layer must compress by 2" },
        { 18, 1, "the last ratio-2 layer must compress by 2" },
        { 19, 0, "the first ratio-2 layer must compress" },
        { 20, 2, "the first ratio-1 layer must compress by 1" },
        { 25, 2, "a middle ratio-1 layer must compress by 1" },
        { 39, 0, "the last backbone layer must compress" },
        { 40, 1, "an DSpark tail layer must not compress" },
    };
    for (const Case& c : cases) {
        ds::Manifest m = ds::default_manifest_v41_flash();
        m.compress_ratios[c.layer] = c.value;
        const auto errors = ds::validate_manifest(m);
        // The per-layer entry and the whole-table entry are always reported; a
        // mutation can also break the kv-source coverage invariant, which is a
        // third, independent finding.
        if (errors.size() < 2) {
            std::fprintf(stderr, "FAIL: %s expected a per-layer and a whole-table error, got: %s\n",
                         c.description, join(errors).c_str());
            std::exit(1);
        }
        const ds::ValidationError* whole = find(errors, "compress_ratios");
        const ds::ValidationError* layer =
            find(errors, "compress_ratios[" + std::to_string(c.layer) + "]");
        if (whole == nullptr || layer == nullptr) {
            std::fprintf(stderr, "FAIL: %s missing an error entry: %s\n", c.description,
                         join(errors).c_str());
            std::exit(1);
        }
        check(layer->detail.find(std::to_string(c.layer)) != std::string::npos,
              std::string("the per-layer error names layer ") + std::to_string(c.layer));
        check(layer->detail.find("must have compress ratio") != std::string::npos,
              std::string("the per-layer error names the required ratio for ") + c.description);
        check(whole->detail.find("expected [0,0,") != std::string::npos,
              std::string("the whole-table error restates the pattern for ") + c.description);
        check(whole->detail.find("got [") != std::string::npos,
              std::string("the whole-table error restates the supplied table for ") + c.description);
        ++checks;
    }

    // A 40-entry table (backbone only, the DSpark tail missing) is refused with the
    // count it needed.
    ds::Manifest short_table = ds::default_manifest_v41_flash();
    short_table.compress_ratios.resize(40);
    const auto short_errors = ds::validate_manifest(short_table);
    check_first_error(short_errors, "compress_ratios", "43", "40",
                    "a 40-entry compress table");

    // A 44-entry table is refused the same way.
    ds::Manifest long_table = ds::default_manifest_v41_flash();
    long_table.compress_ratios.push_back(0);
    check_first_error(ds::validate_manifest(long_table), "compress_ratios", "43", "44",
                    "a 44-entry compress table");

    // An empty table.
    ds::Manifest empty_table = ds::default_manifest_v41_flash();
    empty_table.compress_ratios.clear();
    check_first_error(ds::validate_manifest(empty_table), "compress_ratios", "43", "0",
                    "an empty compress table");
}

void test_source_list_rejections() {
    // One wrong entry inside an otherwise correct list is reported on that
    // entry, with the layer it needed, and still leaves the list-level diff.
    struct EntryCase {
        const char* field;
        std::string where;
        const char* needle;
        const char* what;
        void (*mutate)(ds::Manifest&);
    };
    const EntryCase entry_cases[] = {
        { "kv_source_layers", "kv_source_layers[21]", "must be 20, got 21",
          "the fourth kv source is 20, not 21",
          [](ds::Manifest& m) { m.kv_source_layers = { 2, 8, 14, 21 }; } },
        { "index_source_layers", "index_source_layers[37]", "must be 36, got 37",
          "the eighth index source is 36, not 37",
          [](ds::Manifest& m) { m.index_source_layers = { 2, 8, 14, 20, 24, 28, 32, 37 }; } },
        { "engram_layer_ids", "engram_layer_ids[15]", "must be 14, got 15",
          "the second Engram layer is 14, not 15",
          [](ds::Manifest& m) { m.engram_layer_ids = { 1, 15 }; } },
    };
    for (const EntryCase& c : entry_cases) {
        ds::Manifest m = ds::default_manifest_v41_flash();
        c.mutate(m);
        const auto errors = ds::validate_manifest(m);
        const ds::ValidationError* entry = find(errors, c.where);
        if (entry == nullptr) {
            std::fprintf(stderr, "FAIL: %s expected an error on '%s', got: %s\n", c.what,
                         c.where.c_str(), join(errors).c_str());
            std::exit(1);
        }
        check(entry->detail.find(c.needle) != std::string::npos,
              std::string("the entry error names the value it needed for ") + c.what);
        const ds::ValidationError* whole = find(errors, c.field);
        if (whole == nullptr) {
            std::fprintf(stderr, "FAIL: %s expected the list-level diff on '%s', got: %s\n",
                         c.what, c.field, join(errors).c_str());
            std::exit(1);
        }
        check(whole->detail.find("got [") != std::string::npos,
              std::string("the list-level error restates the supplied list for ") + c.what);
        ++checks;
    }

    // Wrong length.
    ds::Manifest kv_short = ds::default_manifest_v41_flash();
    kv_short.kv_source_layers = { 2, 8, 14 };
    check_first_error(ds::validate_manifest(kv_short), "kv_source_layers", "4", "3",
                    "three kv source layers");

    ds::Manifest index_long = ds::default_manifest_v41_flash();
    index_long.index_source_layers = { 2, 8, 14, 20, 24, 28, 32, 36, 38 };
    check_first_error(ds::validate_manifest(index_long), "index_source_layers", "8", "9",
                    "nine index source layers");

    // A reordered or duplicated list has the same length, so it is caught by
    // the ordering invariant; the entry pass still names the first id that
    // does not sit where it belongs.
    ds::Manifest unsorted = ds::default_manifest_v41_flash();
    unsorted.kv_source_layers = { 20, 14, 8, 2 };
    {
        const auto errors = ds::validate_manifest(unsorted);
        check_first_error(errors, "kv_source_layers[20]", "must be 2, got 20", "",
                          "descending kv sources");
        bool ordered = false;
        for (const auto& error : errors) {
            ordered = ordered ||
                      (error.field == "kv_source_layers" &&
                       error.detail.find("strictly ascending") != std::string::npos);
        }
        check(ordered, "descending kv sources are refused as not strictly ascending");
    }

    ds::Manifest duplicated = ds::default_manifest_v41_flash();
    duplicated.index_source_layers = { 2, 8, 14, 20, 24, 28, 32, 32 };
    {
        const auto errors = ds::validate_manifest(duplicated);
        bool ordered = false;
        for (const auto& error : errors) {
            ordered = ordered ||
                      (error.field == "index_source_layers" &&
                       error.detail.find("strictly ascending") != std::string::npos);
        }
        check(ordered, "a duplicated index source is refused as not strictly ascending");
    }

    // Out-of-range ids are refused by the range invariant and named by the
    // entry pass.
    ds::Manifest negative = ds::default_manifest_v41_flash();
    negative.engram_layer_ids = { -1, 14 };
    check_first_error(ds::validate_manifest(negative), "engram_layer_ids[-1]",
                      "outside the text layers 0..42", "", "a negative Engram layer id");

    ds::Manifest past_end = ds::default_manifest_v41_flash();
    past_end.index_source_layers = { 2, 8, 14, 20, 24, 28, 32, 43 };
    check_first_error(ds::validate_manifest(past_end), "index_source_layers[43]",
                      "outside the text layers 0..42", "", "an index source past the last layer");

    // An DSpark tail layer owns no compressed rows and scores no shared keys. It is
    // refused by the entry pass, which names the source it had to be; the
    // structural "owns rows with ratio 0" rule is checked on a geometry, below.
    ds::Manifest mtp_kv = ds::default_manifest_v41_flash();
    mtp_kv.kv_source_layers = { 2, 8, 14, 40 };
    check_first_error(ds::validate_manifest(mtp_kv), "kv_source_layers[40]", "must be 20, got 40", "",
                      "an DSpark tail layer as a kv source");

    ds::Manifest mtp_index = ds::default_manifest_v41_flash();
    mtp_index.index_source_layers = { 2, 8, 14, 20, 24, 28, 32, 40 };
    check_first_error(ds::validate_manifest(mtp_index), "index_source_layers[40]", "must be 36, got 40",
                      "", "an DSpark tail layer as an index source");
}

/// The Engram tables are per-layer counts, not a scalar pair.
void test_engram_row_rejections() {
    ds::Manifest wrong_first = ds::default_manifest_v41_flash();
    wrong_first.engram_rows[0] = 1;
    check_first_error(ds::validate_manifest(wrong_first), "engram_rows[0]", "384006168", "1",
                    "a wrong first Engram table size");

    ds::Manifest wrong_second = ds::default_manifest_v41_flash();
    wrong_second.engram_rows[1] += 1;
    check_first_error(ds::validate_manifest(wrong_second), "engram_rows[1]", "384016682", "384016683",
                    "an off-by-one second Engram table size");

    ds::Manifest missing_row = ds::default_manifest_v41_flash();
    missing_row.engram_rows.pop_back();
    check_first_error(ds::validate_manifest(missing_row), "engram_rows", "2", "1",
                    "one Engram table size for two Engram layers");
}

void test_from_manifest_error_ordering() {
    // Two problems: the scalar error is reported first and the loader refuses
    // to return a geometry.
    ds::Manifest m = ds::default_manifest_v41_flash();
    m.hidden_size = 1;
    m.compress_ratios[5] = 3;   // not a ratio any kv source can cover
    const auto errors = ds::validate_manifest(m);
    check(errors.size() == 4, "a wrong scalar and a wrong layer produce four errors");
    check(errors[0].field == "hidden_size", "the scalar error comes first");
    check(errors[1].field == "compress_ratios", "then the whole-table compress error");
    check(errors[2].field == "compress_ratios[5]", "then the per-layer compress error");
    check(errors[3].field == "kv_source_layers", "then the kv-source coverage finding");

    std::string error;
    const ds::Geometry built = ds::from_manifest(m, error);
    check(error.rfind("hidden_size", 0) == 0, "from_manifest reports the scalar failure first");
    check(built.compress_ratios.empty() && built.kv_source_layers.empty() &&
              built.index_source_layers.empty(),
          "a refused manifest yields a geometry with no per-layer lists");
    check(built.hidden_size == ds::kHiddenSize,
          "a refused manifest falls back to the member defaults, not to the manifest");
}


/// Invariants that hold for any coherent geometry, checked on a geometry
/// directly because the pinned-value pass rejects a different list first.
void test_geometry_invariants() {
    const ds::Geometry base = ds::default_v41_flash();

    // An index source whose kv source writes no index keys.
    ds::Geometry orphan_index = base;
    orphan_index.kv_source_layers = { 2, 8, 14, 20, 21 };
    orphan_index.compress_ratios[21] = 1;
    orphan_index.compress_ratios[22] = 1;
    {
        const auto errors = ds::validate(orphan_index);
        bool found = false;
        for (const auto& error : errors) {
            found = found || (error.field == "index_source_layers" &&
                              error.detail.find("is not an index source and writes none") !=
                                  std::string::npos);
        }
        check(found, "an index source reading a non-index kv source is refused");
    }

    // A candidate stage on a layer that selects no candidates.
    ds::Geometry misplaced_candidate = base;
    misplaced_candidate.candidate_source_layer = 3;
    {
        const auto errors = ds::validate(misplaced_candidate);
        bool found = false;
        for (const auto& error : errors) {
            found = found || (error.field == "candidate_source_layer" &&
                              error.detail.find("not an index source") != std::string::npos);
        }
        check(found, "candidate selection on a non-index-source layer is refused");
    }

    // A compressing layer with no kv source of its own ratio at or below it.
    ds::Geometry uncovered = base;
    uncovered.compress_ratios[9] = 1;
    {
        const auto errors = ds::validate(uncovered);
        bool found = false;
        for (const auto& error : errors) {
            found = found || (error.field == "kv_source_layers" &&
                              error.detail.find("has no kv source of the same ratio at or below it") !=
                                  std::string::npos);
        }
        check(found, "an uncovered compressing layer is refused");
    }

    // A structural break that is not a pinned value: the query heads no longer
    // divide into the output groups.
    ds::Geometry odd_groups = base;
    odd_groups.attention_heads = 6;
    odd_groups.head_dim = 5120 / 6;
    {
        const auto errors = ds::validate(odd_groups);
        bool found = false;
        for (const auto& error : errors) {
            found = found || (error.field == "output_groups" &&
                              error.detail.find("divisible") != std::string::npos);
        }
        check(found, "output groups that do not divide the query heads are refused");
    }

    // The YaRN context must follow from the base context and the factor.
    ds::Geometry bad_context = base;
    bad_context.max_context_tokens = 262144;
    {
        const auto errors = ds::validate(bad_context);
        bool found = false;
        for (const auto& error : errors) {
            found = found || (error.field == "max_context_tokens" &&
                              error.detail.find("does not reach") != std::string::npos);
        }
        check(found, "a context length the rope scaling cannot reach is refused");
    }

    // Engram rows must be one per Engram layer and positive.
    ds::Geometry mismatched_rows = base;
    mismatched_rows.engram_rows.pop_back();
    {
        const auto errors = ds::validate(mismatched_rows);
        bool found = false;
        for (const auto& error : errors) {
            found = found || (error.field == "engram_rows" &&
                              error.detail.find("for 2 engram layers") != std::string::npos);
        }
        check(found, "a missing Engram table size is refused");
    }

    ds::Geometry empty_rows = base;
    empty_rows.engram_rows[0] = 0;
    {
        const auto errors = ds::validate(empty_rows);
        bool found = false;
        for (const auto& error : errors) {
            found = found || (error.field == "engram_rows" &&
                              error.detail.find("at least one row") != std::string::npos);
        }
        check(found, "an empty Engram table is refused");
    }

    // A top-k wider than every row the indexer could score.
    ds::Geometry huge_topk = base;
    huge_topk.index_top_k = 1000000;
    {
        const auto errors = ds::validate(huge_topk);
        bool found = false;
        for (const auto& error : errors) {
            found = found || (error.field == "index_top_k" &&
                              error.detail.find("must be in 1..5120") != std::string::npos);
        }
        check(found, "an indexer top-k wider than the context is refused");
    }

    // A candidate source outside the text layers.
    ds::Geometry far_candidate = base;
    far_candidate.candidate_source_layer = 43;
    {
        const auto errors = ds::validate(far_candidate);
        bool found = false;
        for (const auto& error : errors) {
            found = found || (error.field == "candidate_source_layer" &&
                              error.detail.find("outside the text layers") != std::string::npos);
        }
        check(found, "a candidate source past the last layer is refused");
    }
}

}  // namespace

int main() {
    test_constants();
    test_compress_pattern();
    test_source_lists();
    test_derived_helpers();
    test_loaders_accept_the_default();
    test_scalar_rejections();
    test_compress_pattern_rejections();
    test_source_list_rejections();
    test_engram_row_rejections();
    test_from_manifest_error_ordering();
    test_geometry_invariants();

    std::printf("deepseek_geometry_test: %d checks passed\n", checks);
    return 0;
}
