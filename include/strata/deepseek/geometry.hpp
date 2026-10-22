// include/strata/deepseek/geometry.hpp - the Strata-owned DeepSeek V4.1
// (model type `deepseek_v41`) geometry and configuration contract.
//
// SCOPE. This header is a host-safe, header-only model contract: what the
// DeepSeek V4.1-Flash-1017 shape is, expressed in Strata's own names, with
// strict validation and precise errors. It is not an executable engine path.
// It reads no checkpoint, allocates no expert or KV memory, and proves nothing
// about any backend. It does not define a GGUF dialect, a pack format, or a
// tensor naming scheme; those belong to the planned RESMP-owned converter and
// quantizer (see docs/architecture.md, "DeepSeek V4.1 own-quantization and
// backend order"). Any on-disk format reader built later must transcode its own
// metadata into the compact manifest below.
//
// PROVENANCE of every number in `default_v41_flash()`: the pinned checkpoint
// configuration at deepseek-ai/DeepSeek-V4.1-Flash-1017 revision
// 314113f8a7dadb388215396ba5fbee2c610a04d6, recorded in
// evidence/deepseek-v41-flash-portability-audit.json (`target_model`), and
// dated in docs/architecture.md ("DeepSeek-V4.1-Flash portability audit").
// Lucebox commit 5ada2a5 was read for semantics only (compressed-cache
// ownership, index-source hand-off, candidate-block pre-selection, Engram
// placement); no Lucebox code, metadata dialect, or ROCMFP format is adopted
// here.
//
// LAYERS. The text backbone has 40 layers, ids 0..39. Three multi-token
// prediction layers follow it, ids 40..42, so an artifact lists 43 layer
// entries in total: the published per-layer compression table is the backbone's
// 40 entries (a leading 0 and 0, then eighteen 2s, twenty 1s, and three 0s for
// the DSpark tail). DSpark is a separate semi-autoregressive block with its own
// three layers, and the vision tower has its own 32 layers.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata::deepseek_v41 {

// ---------------------------------------------------------------- constants

inline constexpr char MODEL_TYPE[] = "deepseek_v41";

/// Text backbone layers, ids 0..39. Three DSpark tail layers follow (ids 40..42).
inline constexpr std::size_t kTextLayers = 40;
inline constexpr std::size_t kDsparkTailLayers = 3;
/// Text backbone plus the DSpark tail: the count a per-layer artifact table carries.
inline constexpr std::size_t kTotalTextLayers = kTextLayers + kDsparkTailLayers;

inline constexpr std::size_t kHiddenSize = 5120;
inline constexpr std::size_t kAttentionHeads = 64;
inline constexpr std::size_t kKvHeads = 1;
inline constexpr std::size_t kHeadDim = 512;
inline constexpr std::size_t kQkRopeDim = 64;
inline constexpr std::size_t kQLoraRank = 1280;
inline constexpr std::size_t kOLoraRank = 1024;
inline constexpr std::size_t kOutputGroups = 8;

inline constexpr std::size_t kRoutedExperts = 384;
inline constexpr std::size_t kRoutedExpertsPerToken = 6;
inline constexpr std::size_t kSharedExperts = 1;
inline constexpr std::size_t kMoeIntermediateSize = 2304;
inline constexpr float kRoutedScalingFactor = 1.5f;
inline constexpr std::string_view kTopKMethod = "noaux_tc";
inline constexpr std::string_view kScoringFunc = "sqrtsoftplus";

inline constexpr std::size_t kSlidingWindow = 128;
inline constexpr std::size_t kIndexHeadCount = 32;
inline constexpr std::size_t kIndexHeadDim = 128;
inline constexpr std::size_t kIndexTopK = 512;
inline constexpr std::size_t kCandidateBlocks = 2048;
inline constexpr std::size_t kCandidateBlockSize = 8;
inline constexpr std::size_t kCandidateSourceLayer = 20;
/// Candidate blocks touch at most this many compressed rows.
inline constexpr std::size_t kCandidateRowSpan =
    kCandidateBlocks * kCandidateBlockSize;

inline constexpr std::size_t kHcMultiplier = 4;
inline constexpr std::size_t kHcSinkhornIterations = 20;

inline constexpr std::size_t kEngramMaxNgram = 4;
inline constexpr std::size_t kEngramHeadDim = 256;
inline constexpr std::size_t kEngramHeads = 8;
inline constexpr std::size_t kEngramLayers = 2;

inline constexpr std::size_t kDsparkLayers = 3;
inline constexpr std::size_t kDsparkRoutedExperts = 128;
inline constexpr std::size_t kDsparkRoutedExpertsPerToken = 3;
inline constexpr std::size_t kDsparkBlockSize = 5;

inline constexpr std::size_t kVisionLayers = 32;
inline constexpr std::size_t kVisionHiddenSize = 1024;
inline constexpr std::size_t kVisionAttentionHeads = 16;
inline constexpr std::size_t kVisionIntermediateSize = 2816;
inline constexpr std::size_t kVisionPatchSize = 14;
inline constexpr std::size_t kVisionDownsampleRatio = 3;
inline constexpr std::size_t kVisionMaxImageTokens = 1024;

inline constexpr std::int64_t kMaxContextTokens = 1048576;
inline constexpr std::string_view kRopeType = "yarn";
inline constexpr float kRopeYarnFactor = 16.0f;
inline constexpr float kRopeYarnBetaFast = 32.0f;
inline constexpr float kRopeYarnBetaSlow = 1.0f;
inline constexpr std::int64_t kRopeOriginalContext = 65536;
inline constexpr float kCompressRopeTheta = 160000.0f;
inline constexpr float kRmsNormEps = 1.0e-20f;
inline constexpr float kSwigluLimit = 10.0f;

/// Engram table rows, per layer, in layer-id order.
inline constexpr std::int64_t kEngramRows0 = 384006168;
inline constexpr std::int64_t kEngramRows1 = 384016682;

/// Published per-layer compression table: 43 entries, the backbone's 40 plus
/// the three DSpark tail entries.
inline constexpr std::uint32_t kCompressRatios[kTotalTextLayers] = {
    0, 0,                                        // backbone layers 0 and 1
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  // backbone layers 2..19
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // backbone layers 20..39
    0, 0, 0,                                      // DSpark tail layers 40..42
};

/// Layers that run a compressor and own the compressed rows read later.
inline constexpr std::int32_t kKvSourceLayers[] = {2, 8, 14, 20};
/// Layers that score the shared index keys and hand their top-k onwards.
inline constexpr std::int32_t kIndexSourceLayers[] = {2, 8, 14, 20, 24, 28, 32, 36};
/// Layers carrying an Engram n-gram table.
inline constexpr std::int32_t kEngramLayerIds[] = {1, 14};

// ------------------------------------------------------------ derived views

/// Whether `ids` contains `id` (also used by the layer helpers below).
constexpr bool contains_id(const std::vector<std::int32_t>& ids, std::int32_t id) noexcept {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

constexpr std::span<const std::uint32_t> compress_ratios_view() noexcept {
    return std::span<const std::uint32_t>(kCompressRatios, kTotalTextLayers);
}
constexpr std::span<const std::int32_t> kv_source_layers_view() noexcept {
    return std::span<const std::int32_t>(kKvSourceLayers, std::size(kKvSourceLayers));
}
constexpr std::span<const std::int32_t> index_source_layers_view() noexcept {
    return std::span<const std::int32_t>(kIndexSourceLayers, std::size(kIndexSourceLayers));
}
constexpr std::span<const std::int32_t> engram_layer_ids_view() noexcept {
    return std::span<const std::int32_t>(kEngramLayerIds, std::size(kEngramLayerIds));
}
/// The backbone's own 40 compression entries (the DSpark tail is excluded).
constexpr std::span<const std::uint32_t> backbone_compress_ratios_view() noexcept {
    return compress_ratios_view().first(kTextLayers);
}

// ----------------------------------------------------------------- geometry

/// The model contract. Every field is a model constant, not a runtime knob.
struct Geometry {
    std::string model_type = MODEL_TYPE;

    // Text backbone.
    std::size_t text_layers = kTextLayers;
    std::size_t dspark_tail_layers = kDsparkTailLayers;
    std::size_t hidden_size = kHiddenSize;
    std::size_t attention_heads = kAttentionHeads;
    std::size_t kv_heads = kKvHeads;
    std::size_t head_dim = kHeadDim;
    std::size_t qk_rope_dim = kQkRopeDim;
    std::size_t q_lora_rank = kQLoraRank;
    std::size_t o_lora_rank = kOLoraRank;
    std::size_t output_groups = kOutputGroups;
    std::size_t sliding_window = kSlidingWindow;

    // Sparse-attention indexer and the hierarchical candidate-block stage.
    std::size_t index_head_count = kIndexHeadCount;
    std::size_t index_head_dim = kIndexHeadDim;
    std::size_t index_top_k = kIndexTopK;
    std::size_t candidate_blocks = kCandidateBlocks;
    std::size_t candidate_block_size = kCandidateBlockSize;
    std::size_t candidate_source_layer = kCandidateSourceLayer;

    // Hyper-connection (mHC).
    std::size_t hc_multiplier = kHcMultiplier;
    std::size_t hc_sinkhorn_iterations = kHcSinkhornIterations;

    // Mixture of experts.
    std::size_t routed_experts = kRoutedExperts;
    std::size_t routed_experts_per_token = kRoutedExpertsPerToken;
    std::size_t shared_experts = kSharedExperts;
    std::size_t moe_intermediate_size = kMoeIntermediateSize;
    float routed_scaling_factor = kRoutedScalingFactor;
    std::string top_k_method{ kTopKMethod };
    std::string scoring_func{ kScoringFunc };

    // Engram n-gram memory.
    std::vector<std::int32_t> engram_layer_ids;
    std::vector<std::int64_t> engram_rows;
    std::size_t engram_max_ngram = kEngramMaxNgram;
    std::size_t engram_head_dim = kEngramHeadDim;
    std::size_t engram_heads = kEngramHeads;

    // DSpark speculative block.
    std::size_t dspark_layers = kDsparkLayers;
    std::size_t dspark_routed_experts = kDsparkRoutedExperts;
    std::size_t dspark_routed_experts_per_token = kDsparkRoutedExpertsPerToken;
    std::size_t dspark_block_size = kDsparkBlockSize;

    // Vision tower.
    std::size_t vision_layers = kVisionLayers;
    std::size_t vision_hidden_size = kVisionHiddenSize;
    std::size_t vision_attention_heads = kVisionAttentionHeads;
    std::size_t vision_intermediate_size = kVisionIntermediateSize;
    std::size_t vision_patch_size = kVisionPatchSize;
    std::size_t vision_downsample_ratio = kVisionDownsampleRatio;
    std::size_t vision_max_image_tokens = kVisionMaxImageTokens;

    // Position encoding and norms.
    std::int64_t max_context_tokens = kMaxContextTokens;
    std::string rope_type{ kRopeType };
    float rope_yarn_factor = kRopeYarnFactor;
    float rope_yarn_beta_fast = kRopeYarnBetaFast;
    float rope_yarn_beta_slow = kRopeYarnBetaSlow;
    std::int64_t rope_original_context = kRopeOriginalContext;
    float compress_rope_theta = kCompressRopeTheta;
    float rms_norm_eps = kRmsNormEps;
    float swiglu_limit = kSwigluLimit;

    // Per-layer lists (43, 4, 8, and 2 entries respectively).
    std::vector<std::uint32_t> compress_ratios;
    std::vector<std::int32_t> kv_source_layers;
    std::vector<std::int32_t> index_source_layers;

    // --- derived helpers, all bounds-safe -------------------------------

    /// Text layers the backbone holds.
    std::size_t total_text_layers() const noexcept {
        return text_layers + dspark_tail_layers;
    }
    /// True when `layer` is inside the backbone.
    bool is_backbone_layer(std::size_t layer) const noexcept {
        return layer < text_layers;
    }
    /// True when `layer` is an DSpark tail layer (ids text_layers .. total_text_layers()-1).
    bool is_dspark_tail_layer(std::size_t layer) const noexcept {
        return layer >= text_layers && layer < total_text_layers();
    }
    /// The compression ratio of a backbone layer; 0 for DSpark tail or out-of-range ids.
    std::uint32_t backbone_compress_ratio(std::size_t layer) const noexcept {
        if (!is_backbone_layer(layer) || layer >= compress_ratios.size()) {
            return 0;
        }
        return compress_ratios[layer];
    }
    /// True when the layer writes and owns compressed rows.
    bool is_kv_source(std::size_t layer) const noexcept {
        return contains_id(kv_source_layers, static_cast<std::int32_t>(layer));
    }
    /// True when the layer scores shared index keys for the layers after it.
    bool is_index_source(std::size_t layer) const noexcept {
        return contains_id(index_source_layers, static_cast<std::int32_t>(layer));
    }
    bool is_engram_layer(std::size_t layer) const noexcept {
        return contains_id(engram_layer_ids, static_cast<std::int32_t>(layer));
    }
    /// Expert columns computed per token in a backbone layer.
    std::size_t routed_experts_per_token_total() const noexcept {
        return routed_experts_per_token + shared_experts;
    }
    /// Expert tensors stored for one backbone layer.
    std::size_t text_experts_per_layer() const noexcept {
        return routed_experts + shared_experts;
    }
    /// Expert columns computed per token in a DSpark layer.
    std::size_t dspark_experts_per_token_total() const noexcept {
        return dspark_routed_experts_per_token;
    }
    /// Expert tensors stored for one DSpark layer.
    std::size_t dspark_experts_per_layer() const noexcept {
        // The published index has layers.* for the 40 backbone layers and
        // mtp.* for the three DSpark stages.  Every DSpark stage also stores
        // the same one shared-expert stack as a backbone layer.  Counting the
        // tail here, rather than as a text layer below, prevents double-counting
        // the three stages in total_expert_count().
        return dspark_routed_experts + shared_experts;
    }
    /// Every expert tensor across the model: backbone and text and DSpark tail layers plus
    /// the DSpark layers.
    std::size_t total_expert_count() const noexcept {
        return text_layers * text_experts_per_layer() +
               dspark_layers * dspark_experts_per_layer();
    }
    std::size_t total_source_count() const noexcept {
        return kv_source_layers.size() + index_source_layers.size();
    }
    std::size_t total_engram_rows() const noexcept;
    /// The Engram table size of one layer; 0 when the id is unknown.
    std::int64_t engram_rows_for(std::size_t layer) const noexcept;
};

// ----------------------------------------------------------------- manifest

/// A compact, file-format-free description of a DeepSeek V4.1 artifact: the
/// scalars plus the per-layer lists. Unknown fields are absent rather than
/// zero-valued, so a caller can tell "not reported" from "reported as zero".
struct Manifest {
    std::string model_type = MODEL_TYPE;

    std::size_t text_layers = 0;
    std::size_t dspark_tail_layers = 0;
    std::size_t hidden_size = 0;
    std::size_t attention_heads = 0;
    std::size_t kv_heads = 0;
    std::size_t head_dim = 0;
    std::size_t qk_rope_dim = 0;
    std::size_t q_lora_rank = 0;
    std::size_t o_lora_rank = 0;
    std::size_t output_groups = 0;
    std::size_t sliding_window = 0;

    std::size_t index_head_count = 0;
    std::size_t index_head_dim = 0;
    std::size_t index_top_k = 0;
    std::size_t candidate_blocks = 0;
    std::size_t candidate_block_size = 0;
    std::size_t candidate_source_layer = 0;

    std::size_t hc_multiplier = 0;
    std::size_t hc_sinkhorn_iterations = 0;

    std::size_t routed_experts = 0;
    std::size_t routed_experts_per_token = 0;
    std::size_t shared_experts = 0;
    std::size_t moe_intermediate_size = 0;
    float routed_scaling_factor = 0.0f;
    std::string top_k_method;
    std::string scoring_func;

    std::vector<std::int32_t> engram_layer_ids;
    std::vector<std::int64_t> engram_rows;
    std::size_t engram_max_ngram = 0;
    std::size_t engram_head_dim = 0;
    std::size_t engram_heads = 0;

    std::size_t dspark_layers = 0;
    std::size_t dspark_routed_experts = 0;
    std::size_t dspark_routed_experts_per_token = 0;
    std::size_t dspark_block_size = 0;

    std::size_t vision_layers = 0;
    std::size_t vision_hidden_size = 0;
    std::size_t vision_attention_heads = 0;
    std::size_t vision_intermediate_size = 0;
    std::size_t vision_patch_size = 0;
    std::size_t vision_downsample_ratio = 0;
    std::size_t vision_max_image_tokens = 0;

    std::int64_t max_context_tokens = 0;
    std::string rope_type;
    float rope_yarn_factor = 0.0f;
    float rope_yarn_beta_fast = 0.0f;
    float rope_yarn_beta_slow = 0.0f;
    std::int64_t rope_original_context = 0;
    float compress_rope_theta = 0.0f;
    float rms_norm_eps = 0.0f;
    float swiglu_limit = 0.0f;

    std::vector<std::uint32_t> compress_ratios;
    std::vector<std::int32_t> kv_source_layers;
    std::vector<std::int32_t> index_source_layers;
};

// ------------------------------------------------------- loaders and checks

/// The pinned DeepSeek V4.1-Flash-1017 contract (revision
/// 314113f8a7dadb388215396ba5fbee2c610a04d6).
Geometry default_v41_flash();

/// A manifest holding exactly the default values.
Manifest default_manifest_v41_flash();

/// Fields the manifest omits (everything a default fills in).
struct ValidationError {
    std::string field;   ///< dotted field path, e.g. "compress_ratios[19]"
    std::string detail;  ///< why it was rejected
};

/// Validates a geometry against the V4.1-Flash contract and every internal
/// invariant. Errors are ordered (scalars, then per-layer lists, then
/// invariants); they are facts about the argument, not about any artifact.
std::vector<ValidationError> validate(const Geometry& geometry);

/// Validates a manifest as a standalone description: every field it reports
/// must match the pinned contract, and the values it carries must satisfy every
/// internal invariant. The result is exactly what `validate` reports for the
/// geometry that `from_manifest` would build from it.
std::vector<ValidationError> validate_manifest(const Manifest& manifest);

/// Builds a geometry from a validated manifest; `error` receives the first
/// failure in the same order `validate_manifest` reports.
Geometry from_manifest(const Manifest& manifest, std::string& error);

inline std::size_t Geometry::total_engram_rows() const noexcept {
    std::size_t total = 0;
    for (std::int64_t rows : engram_rows) {
        total += static_cast<std::size_t>(rows);
    }
    return total;
}

inline std::int64_t Geometry::engram_rows_for(std::size_t layer) const noexcept {
    for (std::size_t i = 0; i < engram_layer_ids.size(); ++i) {
        if (engram_layer_ids[i] == static_cast<std::int64_t>(layer) && i < engram_rows.size()) {
            return engram_rows[i];
        }
    }
    return 0;
}

namespace detail {

// The pinned pattern as an owning copy, so the check does not read the same
// storage it is validating and an error can quote it.
inline const std::vector<std::uint32_t>& expected_compress_ratios() {
    static const std::vector<std::uint32_t> expected(kCompressRatios,
                                                     kCompressRatios + kTotalTextLayers);
    return expected;
}

struct ErrorSink {
    std::vector<ValidationError> errors;
    void add(std::string field, std::string detail) {
        errors.push_back(ValidationError{ std::move(field), std::move(detail) });
    }
};

template <typename T>
void want_size(ErrorSink& sink, const char* field, T got, T want) {
    if (got != want) {
        sink.add(field, "expected " + std::to_string(want) + ", got " + std::to_string(got));
    }
}

inline void want_float(ErrorSink& sink, const char* field, float got, float want) {
    if (got != want) {
        char detail[128];
        std::snprintf(detail, sizeof detail, "expected %g, got %g",
                      static_cast<double>(want), static_cast<double>(got));
        sink.add(field, detail);
    }
}

inline void want_i64(ErrorSink& sink, const char* field, std::int64_t got, std::int64_t want) {
    if (got != want) {
        sink.add(field, "expected " + std::to_string(want) + ", got " + std::to_string(got));
    }
}

bool is_ascending_in_range(const std::vector<std::int32_t>& ids, std::size_t layer_count) {
    std::int64_t previous = -1;
    for (std::int32_t id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= layer_count) {
            return false;
        }
        if (id <= previous) {
            return false;
        }
        previous = id;
    }
    return true;
}

std::string describe(std::span<const std::int32_t> ids) {
    std::string out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        out += std::to_string(ids[i]);
    }
    return out;
}

inline std::string describe(const std::vector<std::int32_t>& ids) {
    return describe(std::span<const std::int32_t>(ids.data(), ids.size()));
}

inline std::string describe(const std::vector<std::uint32_t>& ratios) {
    std::string out;
    for (std::size_t i = 0; i < ratios.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        out += std::to_string(ratios[i]);
    }
    return out;
}

/// One wrong id is named on its own entry, with the value the contract wants,
/// and the list as a whole is still diffed so a reader sees both.
void check_kv_sources(const std::vector<std::int32_t>& ids, ErrorSink& sink) {
    want_size(sink, "kv_source_layers", ids.size(), std::size(kKvSourceLayers));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::string where = "kv_source_layers[" + std::to_string(ids[i]) + "]";
        if (ids[i] < 0 || static_cast<std::size_t>(ids[i]) >= kTotalTextLayers) {
            sink.add(where, "layer " + std::to_string(ids[i]) + " is outside the text layers 0.." +
                                std::to_string(kTotalTextLayers - 1));
        } else if (i >= std::size(kKvSourceLayers) ||
                   ids[i] != kKvSourceLayers[i]) {
            sink.add(where, std::string("kv source layer ") + std::to_string(i) + " must be " +
                                std::to_string(kKvSourceLayers[std::min(i, std::size(kKvSourceLayers) - 1)]) +
                                ", got " + std::to_string(ids[i]));
        }
    }
    if (ids.size() == std::size(kKvSourceLayers) &&
        !std::equal(ids.begin(), ids.end(), kKvSourceLayers)) {
        sink.add("kv_source_layers", "expected [" + describe(kv_source_layers_view()) +
                                         "], got [" + describe(ids) + "]");
    }
}

void check_index_sources(const std::vector<std::int32_t>& ids, ErrorSink& sink) {
    want_size(sink, "index_source_layers", ids.size(), std::size(kIndexSourceLayers));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::string where = "index_source_layers[" + std::to_string(ids[i]) + "]";
        if (ids[i] < 0 || static_cast<std::size_t>(ids[i]) >= kTotalTextLayers) {
            sink.add(where, "layer " + std::to_string(ids[i]) + " is outside the text layers 0.." +
                                std::to_string(kTotalTextLayers - 1));
        } else if (i >= std::size(kIndexSourceLayers) || ids[i] != kIndexSourceLayers[i]) {
            sink.add(where, "index source layer " + std::to_string(i) + " must be " +
                                std::to_string(kIndexSourceLayers[std::min(i, std::size(kIndexSourceLayers) - 1)]) +
                                ", got " + std::to_string(ids[i]));
        }
    }
    if (ids.size() == std::size(kIndexSourceLayers) &&
        !std::equal(ids.begin(), ids.end(), kIndexSourceLayers)) {
        sink.add("index_source_layers", "expected [" + describe(index_source_layers_view()) +
                                           "], got [" + describe(ids) + "]");
    }
}

void check_engram_layers(const std::vector<std::int32_t>& ids,
                         const std::vector<std::int64_t>& rows, ErrorSink& sink) {
    want_size(sink, "engram_layer_ids", ids.size(), std::size(kEngramLayerIds));
    want_size(sink, "engram_rows", rows.size(), std::size(kEngramLayerIds));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::string where = "engram_layer_ids[" + std::to_string(ids[i]) + "]";
        if (ids[i] < 0 || static_cast<std::size_t>(ids[i]) >= kTotalTextLayers) {
            sink.add(where, "layer " + std::to_string(ids[i]) + " is outside the text layers 0.." +
                                std::to_string(kTotalTextLayers - 1));
        } else if (i >= std::size(kEngramLayerIds) || ids[i] != kEngramLayerIds[i]) {
            sink.add(where, "engram layer " + std::to_string(i) + " must be " +
                                std::to_string(kEngramLayerIds[std::min(i, std::size(kEngramLayerIds) - 1)]) +
                                ", got " + std::to_string(ids[i]));
        }
    }
    if (ids.size() == std::size(kEngramLayerIds) &&
        !std::equal(ids.begin(), ids.end(), kEngramLayerIds)) {
        sink.add("engram_layer_ids", "expected [" + describe(engram_layer_ids_view()) +
                                         "], got [" + describe(ids) + "]");
    }
    if (rows.size() == std::size(kEngramLayerIds)) {
        want_i64(sink, "engram_rows[0]", rows[0], kEngramRows0);
        want_i64(sink, "engram_rows[1]", rows[1], kEngramRows1);
    }
}

/// Structural checks over the derived relationships. Every one of these is a
/// property of a coherent model, independent of the pinned numbers, so they
/// hold for any geometry a future converter hands this contract.
void check_invariants(const Geometry& g, ErrorSink& sink) {
    const std::size_t total = g.total_text_layers();

    if (g.attention_heads % g.kv_heads != 0) {
        sink.add("kv_heads", "attention_heads " + std::to_string(g.attention_heads) +
                                " must be divisible by kv_heads " + std::to_string(g.kv_heads));
    }
    if (g.attention_heads % g.output_groups != 0) {
        sink.add("output_groups", "attention_heads " + std::to_string(g.attention_heads) +
                                      " must be divisible by output_groups " +
                                      std::to_string(g.output_groups));
    }
    if (g.qk_rope_dim == 0 || g.qk_rope_dim > g.head_dim) {
        sink.add("qk_rope_dim", "must be in 1.." + std::to_string(g.head_dim));
    }
    if (g.routed_experts_per_token == 0 || g.routed_experts_per_token > g.routed_experts) {
        sink.add("routed_experts_per_token",
                 "must be in 1.." + std::to_string(g.routed_experts));
    }
    if (g.index_top_k == 0 || g.index_top_k > g.sliding_window * g.text_layers) {
        sink.add("index_top_k", "must be in 1.." +
                                    std::to_string(g.sliding_window * g.text_layers));
    }
    if (g.candidate_source_layer >= total) {
        sink.add("candidate_source_layer",
                 "layer " + std::to_string(g.candidate_source_layer) +
                     " is outside the text layers 0.." + std::to_string(total - 1));
    }
    if (g.candidate_block_size == 0 || g.candidate_blocks == 0) {
        sink.add("candidate_blocks", "candidate_blocks and candidate_block_size must both be > 0");
    }
    if (g.vision_attention_heads == 0 ||
        g.vision_hidden_size % g.vision_attention_heads != 0) {
        sink.add("vision_attention_heads",
                 "vision_hidden_size " + std::to_string(g.vision_hidden_size) +
                     " must be divisible by vision_attention_heads " +
                     std::to_string(g.vision_attention_heads));
    }
    if (!(g.rope_yarn_factor >= 1.0f)) {
        sink.add("rope_yarn_factor", "must be at least 1, got " +
                                         std::to_string(static_cast<double>(g.rope_yarn_factor)));
    }
    if (g.rope_yarn_factor >= 1.0f && g.max_context_tokens > 0 &&
        static_cast<std::int64_t>(g.rope_original_context * g.rope_yarn_factor) !=
            g.max_context_tokens) {
        sink.add("max_context_tokens", "rope_original_context " +
                                          std::to_string(g.rope_original_context) +
                                          " times rope_yarn_factor " +
                                          std::to_string(static_cast<double>(g.rope_yarn_factor)) +
                                          " does not reach " + std::to_string(g.max_context_tokens));
    }
    if (g.engram_layer_ids.size() != g.engram_rows.size()) {
        sink.add("engram_rows", "engram_rows holds " + std::to_string(g.engram_rows.size()) +
                                    " entries for " + std::to_string(g.engram_layer_ids.size()) +
                                    " engram layers");
    }
    for (std::int64_t rows : g.engram_rows) {
        if (rows <= 0) {
            sink.add("engram_rows", "every engram table must hold at least one row");
            break;
        }
    }

    // Per-layer lists. From here on the compress table is indexed, so a table
    // of the wrong length stops the pass.
    if (g.compress_ratios.size() != total) {
        // The table length is already reported by validate's strict pass; stop
        // here rather than restating it and indexing a table of another length.
        return;
    }
    const std::vector<std::uint32_t>& expected = expected_compress_ratios();
    for (std::size_t layer = 0; layer < total; ++layer) {
        if (g.compress_ratios[layer] == expected[layer]) {
            continue;
        }
        const char* where = g.is_dspark_tail_layer(layer) ? "DSpark tail layer " : "text layer ";
        sink.add("compress_ratios[" + std::to_string(layer) + "]",
                 std::string(where) + std::to_string(layer) + " must have compress ratio " +
                     std::to_string(expected[layer]) + ", got " +
                     std::to_string(g.compress_ratios[layer]));
    }
    if (!is_ascending_in_range(g.kv_source_layers, total)) {
        sink.add("kv_source_layers", "layer ids must be strictly ascending and inside 0.." +
                                         std::to_string(total - 1) + ", got " +
                                         describe(g.kv_source_layers));
    }
    if (!is_ascending_in_range(g.index_source_layers, total)) {
        sink.add("index_source_layers", "layer ids must be strictly ascending and inside 0.." +
                                           std::to_string(total - 1) + ", got " +
                                           describe(g.index_source_layers));
    }
    if (!is_ascending_in_range(g.engram_layer_ids, total)) {
        sink.add("engram_layer_ids", "layer ids must be strictly ascending and inside 0.." +
                                         std::to_string(total - 1) + ", got " +
                                         describe(g.engram_layer_ids));
    }

    // A source layer must compress rows, and every compressing layer must find
    // a kv source of its own ratio at or below it; the layers after that
    // source share its compressed rows.
    for (std::int32_t layer : g.kv_source_layers) {
        if (static_cast<std::size_t>(layer) < total &&
            g.compress_ratios[static_cast<std::size_t>(layer)] == 0) {
            sink.add("kv_source_layers[" + std::to_string(layer) + "]",
                     "layer " + std::to_string(layer) + " owns compressed rows but its compress ratio is 0");
        }
    }
    for (std::int32_t layer : g.index_source_layers) {
        if (static_cast<std::size_t>(layer) < total &&
            g.compress_ratios[static_cast<std::size_t>(layer)] == 0) {
            sink.add("index_source_layers[" + std::to_string(layer) + "]",
                     "layer " + std::to_string(layer) + " scores shared index keys but its compress ratio is 0");
        }
    }
    for (std::size_t layer = 0; layer < g.text_layers; ++layer) {
        const std::uint32_t ratio = g.compress_ratios[layer];
        if (ratio == 0) {
            continue;
        }
        std::size_t source = 0;
        bool found = false;
        for (std::size_t candidate = layer + 1; candidate > 0; --candidate) {
            if (contains_id(g.kv_source_layers, static_cast<std::int32_t>(candidate - 1))) {
                source = candidate - 1;
                found = true;
                break;
            }
        }
        if (!found || g.compress_ratios[source] != ratio) {
            sink.add("kv_source_layers", "text layer " + std::to_string(layer) + " (ratio " +
                                             std::to_string(ratio) +
                                             ") has no kv source of the same ratio at or below it");
        }
    }
    // Index keys are written where the compressed rows are, so an index source
    // must read a kv source that is also an index source.
    for (std::int32_t layer : g.index_source_layers) {
        if (static_cast<std::size_t>(layer) >= total) {
            continue;
        }
        const std::size_t layer_id = static_cast<std::size_t>(layer);
        std::size_t source = 0;
        bool found = false;
        for (std::size_t candidate = layer_id + 1; candidate > 0; --candidate) {
            if (contains_id(g.kv_source_layers, static_cast<std::int32_t>(candidate - 1))) {
                source = candidate - 1;
                found = true;
                break;
            }
        }
        if (found && !contains_id(g.index_source_layers, static_cast<std::int32_t>(source))) {
            sink.add("index_source_layers", "index source layer " + std::to_string(layer) +
                                               " reads index keys from kv source layer " +
                                               std::to_string(source) +
                                               ", which is not an index source and writes none");
        }
    }
    // The candidate stage only exists where index keys are scored.
    if (g.candidate_source_layer < total && g.candidate_blocks > 0 &&
        !contains_id(g.index_source_layers, static_cast<std::int32_t>(g.candidate_source_layer))) {
        sink.add("candidate_source_layer",
                 "candidate selection runs at layer " +
                     std::to_string(g.candidate_source_layer) +
                     ", which is not an index source and selects no candidate blocks");
    }
}

}  // namespace detail

inline Geometry default_v41_flash() {
    Geometry g;
    g.engram_layer_ids.assign(kEngramLayerIds, kEngramLayerIds + std::size(kEngramLayerIds));
    g.engram_rows = { kEngramRows0, kEngramRows1 };
    g.compress_ratios.assign(kCompressRatios, kCompressRatios + kTotalTextLayers);
    g.kv_source_layers.assign(kKvSourceLayers, kKvSourceLayers + std::size(kKvSourceLayers));
    g.index_source_layers.assign(kIndexSourceLayers, kIndexSourceLayers + std::size(kIndexSourceLayers));
    return g;
}

inline Manifest default_manifest_v41_flash() {
    const Geometry g = default_v41_flash();
    Manifest m;
    m.model_type = g.model_type;
    m.text_layers = g.text_layers;
    m.dspark_tail_layers = g.dspark_tail_layers;
    m.hidden_size = g.hidden_size;
    m.attention_heads = g.attention_heads;
    m.kv_heads = g.kv_heads;
    m.head_dim = g.head_dim;
    m.qk_rope_dim = g.qk_rope_dim;
    m.q_lora_rank = g.q_lora_rank;
    m.o_lora_rank = g.o_lora_rank;
    m.output_groups = g.output_groups;
    m.sliding_window = g.sliding_window;
    m.index_head_count = g.index_head_count;
    m.index_head_dim = g.index_head_dim;
    m.index_top_k = g.index_top_k;
    m.candidate_blocks = g.candidate_blocks;
    m.candidate_block_size = g.candidate_block_size;
    m.candidate_source_layer = g.candidate_source_layer;
    m.hc_multiplier = g.hc_multiplier;
    m.hc_sinkhorn_iterations = g.hc_sinkhorn_iterations;
    m.routed_experts = g.routed_experts;
    m.routed_experts_per_token = g.routed_experts_per_token;
    m.shared_experts = g.shared_experts;
    m.moe_intermediate_size = g.moe_intermediate_size;
    m.routed_scaling_factor = g.routed_scaling_factor;
    m.top_k_method = g.top_k_method;
    m.scoring_func = g.scoring_func;
    m.engram_layer_ids = g.engram_layer_ids;
    m.engram_rows = g.engram_rows;
    m.engram_max_ngram = g.engram_max_ngram;
    m.engram_head_dim = g.engram_head_dim;
    m.engram_heads = g.engram_heads;
    m.dspark_layers = g.dspark_layers;
    m.dspark_routed_experts = g.dspark_routed_experts;
    m.dspark_routed_experts_per_token = g.dspark_routed_experts_per_token;
    m.dspark_block_size = g.dspark_block_size;
    m.vision_layers = g.vision_layers;
    m.vision_hidden_size = g.vision_hidden_size;
    m.vision_attention_heads = g.vision_attention_heads;
    m.vision_intermediate_size = g.vision_intermediate_size;
    m.vision_patch_size = g.vision_patch_size;
    m.vision_downsample_ratio = g.vision_downsample_ratio;
    m.vision_max_image_tokens = g.vision_max_image_tokens;
    m.max_context_tokens = g.max_context_tokens;
    m.rope_type = g.rope_type;
    m.rope_yarn_factor = g.rope_yarn_factor;
    m.rope_yarn_beta_fast = g.rope_yarn_beta_fast;
    m.rope_yarn_beta_slow = g.rope_yarn_beta_slow;
    m.rope_original_context = g.rope_original_context;
    m.compress_rope_theta = g.compress_rope_theta;
    m.rms_norm_eps = g.rms_norm_eps;
    m.swiglu_limit = g.swiglu_limit;
    m.compress_ratios = g.compress_ratios;
    m.kv_source_layers = g.kv_source_layers;
    m.index_source_layers = g.index_source_layers;
    return m;
}

inline std::vector<ValidationError> validate(const Geometry& g) {
    detail::ErrorSink sink;
    if (g.model_type != MODEL_TYPE) {
        sink.add("model_type", "expected \"" + std::string(MODEL_TYPE) + "\", got \"" +
                                   g.model_type + "\"");
    }

    // Scalars.
    detail::want_size(sink, "text_layers", g.text_layers, kTextLayers);
    detail::want_size(sink, "dspark_tail_layers", g.dspark_tail_layers, kDsparkTailLayers);
    detail::want_size(sink, "hidden_size", g.hidden_size, kHiddenSize);
    detail::want_size(sink, "attention_heads", g.attention_heads, kAttentionHeads);
    detail::want_size(sink, "kv_heads", g.kv_heads, kKvHeads);
    detail::want_size(sink, "head_dim", g.head_dim, kHeadDim);
    detail::want_size(sink, "qk_rope_dim", g.qk_rope_dim, kQkRopeDim);
    detail::want_size(sink, "q_lora_rank", g.q_lora_rank, kQLoraRank);
    detail::want_size(sink, "o_lora_rank", g.o_lora_rank, kOLoraRank);
    detail::want_size(sink, "output_groups", g.output_groups, kOutputGroups);
    detail::want_size(sink, "sliding_window", g.sliding_window, kSlidingWindow);
    detail::want_size(sink, "index_head_count", g.index_head_count, kIndexHeadCount);
    detail::want_size(sink, "index_head_dim", g.index_head_dim, kIndexHeadDim);
    detail::want_size(sink, "index_top_k", g.index_top_k, kIndexTopK);
    detail::want_size(sink, "candidate_blocks", g.candidate_blocks, kCandidateBlocks);
    detail::want_size(sink, "candidate_block_size", g.candidate_block_size, kCandidateBlockSize);
    detail::want_size(sink, "candidate_source_layer", g.candidate_source_layer, kCandidateSourceLayer);
    detail::want_size(sink, "hc_multiplier", g.hc_multiplier, kHcMultiplier);
    detail::want_size(sink, "hc_sinkhorn_iterations", g.hc_sinkhorn_iterations, kHcSinkhornIterations);
    detail::want_size(sink, "routed_experts", g.routed_experts, kRoutedExperts);
    detail::want_size(sink, "routed_experts_per_token", g.routed_experts_per_token, kRoutedExpertsPerToken);
    detail::want_size(sink, "shared_experts", g.shared_experts, kSharedExperts);
    detail::want_size(sink, "moe_intermediate_size", g.moe_intermediate_size, kMoeIntermediateSize);
    detail::want_float(sink, "routed_scaling_factor", g.routed_scaling_factor, kRoutedScalingFactor);
    if (g.top_k_method != kTopKMethod) {
        sink.add("top_k_method", "expected \"" + std::string(kTopKMethod) + "\", got \"" +
                                     g.top_k_method + "\"");
    }
    if (g.scoring_func != kScoringFunc) {
        sink.add("scoring_func", "expected \"" + std::string(kScoringFunc) + "\", got \"" +
                                    g.scoring_func + "\"");
    }
    detail::want_size(sink, "engram_max_ngram", g.engram_max_ngram, kEngramMaxNgram);
    detail::want_size(sink, "engram_head_dim", g.engram_head_dim, kEngramHeadDim);
    detail::want_size(sink, "engram_heads", g.engram_heads, kEngramHeads);
    detail::want_size(sink, "dspark_layers", g.dspark_layers, kDsparkLayers);
    detail::want_size(sink, "dspark_routed_experts", g.dspark_routed_experts, kDsparkRoutedExperts);
    detail::want_size(sink, "dspark_routed_experts_per_token", g.dspark_routed_experts_per_token, kDsparkRoutedExpertsPerToken);
    detail::want_size(sink, "dspark_block_size", g.dspark_block_size, kDsparkBlockSize);
    detail::want_size(sink, "vision_layers", g.vision_layers, kVisionLayers);
    detail::want_size(sink, "vision_hidden_size", g.vision_hidden_size, kVisionHiddenSize);
    detail::want_size(sink, "vision_attention_heads", g.vision_attention_heads, kVisionAttentionHeads);
    detail::want_size(sink, "vision_intermediate_size", g.vision_intermediate_size, kVisionIntermediateSize);
    detail::want_size(sink, "vision_patch_size", g.vision_patch_size, kVisionPatchSize);
    detail::want_size(sink, "vision_downsample_ratio", g.vision_downsample_ratio, kVisionDownsampleRatio);
    detail::want_size(sink, "vision_max_image_tokens", g.vision_max_image_tokens, kVisionMaxImageTokens);
    detail::want_i64(sink, "max_context_tokens", g.max_context_tokens, kMaxContextTokens);
    if (g.rope_type != kRopeType) {
        sink.add("rope_type", "expected \"" + std::string(kRopeType) + "\", got \"" +
                                  g.rope_type + "\"");
    }
    detail::want_float(sink, "rope_yarn_factor", g.rope_yarn_factor, kRopeYarnFactor);
    detail::want_float(sink, "rope_yarn_beta_fast", g.rope_yarn_beta_fast, kRopeYarnBetaFast);
    detail::want_float(sink, "rope_yarn_beta_slow", g.rope_yarn_beta_slow, kRopeYarnBetaSlow);
    detail::want_i64(sink, "rope_original_context", g.rope_original_context, kRopeOriginalContext);
    detail::want_float(sink, "compress_rope_theta", g.compress_rope_theta, kCompressRopeTheta);
    detail::want_float(sink, "rms_norm_eps", g.rms_norm_eps, kRmsNormEps);
    detail::want_float(sink, "swiglu_limit", g.swiglu_limit, kSwigluLimit);

    // Per-layer lists and invariants.
    // Each list is checked entry by entry before its contents are compared, so a
    // single wrong id is reported on that id rather than only as a whole-list
    // diff, and a wrong length never masks a wrong entry.
    detail::check_kv_sources(g.kv_source_layers, sink);
    detail::check_index_sources(g.index_source_layers, sink);
    detail::check_engram_layers(g.engram_layer_ids, g.engram_rows, sink);
    if (g.compress_ratios.size() != kTotalTextLayers) {
        sink.add("compress_ratios", "expected " + std::to_string(kTotalTextLayers) +
                                        " entries, got " + std::to_string(g.compress_ratios.size()));
    } else if (!std::equal(g.compress_ratios.begin(), g.compress_ratios.end(), kCompressRatios)) {
        sink.add("compress_ratios", "expected [" + detail::describe(detail::expected_compress_ratios()) +
                                        "], got [" + detail::describe(g.compress_ratios) + "]");
    }

    detail::check_invariants(g, sink);
    return std::move(sink.errors);
}

inline Geometry from_manifest(const Manifest& m, std::string& error) {
    const std::vector<ValidationError> errors = validate_manifest(m);
    if (!errors.empty()) {
        // Refused: return a default-constructed geometry rather than the pinned
        // one, so a caller that ignores `error` cannot mistake a refusal for a
        // loaded model.
        error = errors.front().field + ": " + errors.front().detail;
        return Geometry{};
    }
    error.clear();
    Geometry g = default_v41_flash();
    g.model_type = m.model_type;
    g.text_layers = m.text_layers;
    g.dspark_tail_layers = m.dspark_tail_layers;
    g.hidden_size = m.hidden_size;
    g.attention_heads = m.attention_heads;
    g.kv_heads = m.kv_heads;
    g.head_dim = m.head_dim;
    g.qk_rope_dim = m.qk_rope_dim;
    g.q_lora_rank = m.q_lora_rank;
    g.o_lora_rank = m.o_lora_rank;
    g.output_groups = m.output_groups;
    g.sliding_window = m.sliding_window;
    g.index_head_count = m.index_head_count;
    g.index_head_dim = m.index_head_dim;
    g.index_top_k = m.index_top_k;
    g.candidate_blocks = m.candidate_blocks;
    g.candidate_block_size = m.candidate_block_size;
    g.candidate_source_layer = m.candidate_source_layer;
    g.hc_multiplier = m.hc_multiplier;
    g.hc_sinkhorn_iterations = m.hc_sinkhorn_iterations;
    g.routed_experts = m.routed_experts;
    g.routed_experts_per_token = m.routed_experts_per_token;
    g.shared_experts = m.shared_experts;
    g.moe_intermediate_size = m.moe_intermediate_size;
    g.routed_scaling_factor = m.routed_scaling_factor;
    g.top_k_method = m.top_k_method;
    g.scoring_func = m.scoring_func;
    g.engram_layer_ids = m.engram_layer_ids;
    g.engram_rows = m.engram_rows;
    g.engram_max_ngram = m.engram_max_ngram;
    g.engram_head_dim = m.engram_head_dim;
    g.engram_heads = m.engram_heads;
    g.dspark_layers = m.dspark_layers;
    g.dspark_routed_experts = m.dspark_routed_experts;
    g.dspark_routed_experts_per_token = m.dspark_routed_experts_per_token;
    g.dspark_block_size = m.dspark_block_size;
    g.vision_layers = m.vision_layers;
    g.vision_hidden_size = m.vision_hidden_size;
    g.vision_attention_heads = m.vision_attention_heads;
    g.vision_intermediate_size = m.vision_intermediate_size;
    g.vision_patch_size = m.vision_patch_size;
    g.vision_downsample_ratio = m.vision_downsample_ratio;
    g.vision_max_image_tokens = m.vision_max_image_tokens;
    g.max_context_tokens = m.max_context_tokens;
    g.rope_type = m.rope_type;
    g.rope_yarn_factor = m.rope_yarn_factor;
    g.rope_yarn_beta_fast = m.rope_yarn_beta_fast;
    g.rope_yarn_beta_slow = m.rope_yarn_beta_slow;
    g.rope_original_context = m.rope_original_context;
    g.compress_rope_theta = m.compress_rope_theta;
    g.rms_norm_eps = m.rms_norm_eps;
    g.swiglu_limit = m.swiglu_limit;
    g.compress_ratios = m.compress_ratios;
    g.kv_source_layers = m.kv_source_layers;
    g.index_source_layers = m.index_source_layers;
    return g;
}

inline std::vector<ValidationError> validate_manifest(const Manifest& m) {
    // Deliberately not from_manifest(): the loader would call this function
    // back, and a header-only contract must not depend on a declaration order.
    Geometry g;
    g.model_type = m.model_type;
    g.text_layers = m.text_layers;
    g.dspark_tail_layers = m.dspark_tail_layers;
    g.hidden_size = m.hidden_size;
    g.attention_heads = m.attention_heads;
    g.kv_heads = m.kv_heads;
    g.head_dim = m.head_dim;
    g.qk_rope_dim = m.qk_rope_dim;
    g.q_lora_rank = m.q_lora_rank;
    g.o_lora_rank = m.o_lora_rank;
    g.output_groups = m.output_groups;
    g.sliding_window = m.sliding_window;
    g.index_head_count = m.index_head_count;
    g.index_head_dim = m.index_head_dim;
    g.index_top_k = m.index_top_k;
    g.candidate_blocks = m.candidate_blocks;
    g.candidate_block_size = m.candidate_block_size;
    g.candidate_source_layer = m.candidate_source_layer;
    g.hc_multiplier = m.hc_multiplier;
    g.hc_sinkhorn_iterations = m.hc_sinkhorn_iterations;
    g.routed_experts = m.routed_experts;
    g.routed_experts_per_token = m.routed_experts_per_token;
    g.shared_experts = m.shared_experts;
    g.moe_intermediate_size = m.moe_intermediate_size;
    g.routed_scaling_factor = m.routed_scaling_factor;
    g.top_k_method = m.top_k_method;
    g.scoring_func = m.scoring_func;
    g.engram_layer_ids = m.engram_layer_ids;
    g.engram_rows = m.engram_rows;
    g.engram_max_ngram = m.engram_max_ngram;
    g.engram_head_dim = m.engram_head_dim;
    g.engram_heads = m.engram_heads;
    g.dspark_layers = m.dspark_layers;
    g.dspark_routed_experts = m.dspark_routed_experts;
    g.dspark_routed_experts_per_token = m.dspark_routed_experts_per_token;
    g.dspark_block_size = m.dspark_block_size;
    g.vision_layers = m.vision_layers;
    g.vision_hidden_size = m.vision_hidden_size;
    g.vision_hidden_size = m.vision_hidden_size;
    g.vision_attention_heads = m.vision_attention_heads;
    g.vision_intermediate_size = m.vision_intermediate_size;
    g.vision_patch_size = m.vision_patch_size;
    g.vision_downsample_ratio = m.vision_downsample_ratio;
    g.vision_max_image_tokens = m.vision_max_image_tokens;
    g.max_context_tokens = m.max_context_tokens;
    g.rope_type = m.rope_type;
    g.rope_yarn_factor = m.rope_yarn_factor;
    g.rope_yarn_beta_fast = m.rope_yarn_beta_fast;
    g.rope_yarn_beta_slow = m.rope_yarn_beta_slow;
    g.rope_original_context = m.rope_original_context;
    g.compress_rope_theta = m.compress_rope_theta;
    g.rms_norm_eps = m.rms_norm_eps;
    g.swiglu_limit = m.swiglu_limit;
    g.compress_ratios = m.compress_ratios;
    g.kv_source_layers = m.kv_source_layers;
    g.index_source_layers = m.index_source_layers;
    return validate(g);
}

}  // namespace strata::deepseek_v41
