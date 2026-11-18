#pragma once
// STFH v1: the Stage 1 MTP feature trace.  It records the target model's WIDE (multi-stream) residual rows at the
// two seams a draft layer consumes them - the end of every prompt chunk and the end of every verify window - with
// the request ordinal and the absolute position that aligns row 0.  This file owns the container only: layout,
// validation and buffered writes.  Nothing here touches a model, a GPU or a kernel; a caller copies device rows to
// host memory and hands them over.
//
// LAYOUT (little-endian; a big-endian host is refused at compile time)
//
//   header (64 bytes, MtpFeaturesHeader)
//   record + (48-byte MtpFeatureRecordHeader) + payload_bytes of f32 payload, repeated
//
// A residual record's payload is `rows * width` f32 values, row-major: row r is the residual at absolute position
// `position + r`, spanning all `width = hc * n_embd` streams.  `layers` and `vocab` describe the model the rows came
// from.  `record_kinds` advertises which kinds a reader must understand; v1 writes residual records only (bit 0).
// The logits bit is reserved and never set by this version, so a reader can reject a newer file that needs it.
//
// The header is self-describing: magic, version, both structure sizes, the kind bitmap, byte order and element size
// come before any geometry, so a reader can validate the container without the model.  `record_header_bytes` lets a
// v2 grow the per-record header without changing v1 readers' framing.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata::trace {

inline constexpr uint32_t kMtpFeaturesMagic = 0x48465453u;   // "STFH" in little-endian file order: S T F H
inline constexpr uint16_t kMtpFeaturesVersion = 1;
inline constexpr uint16_t kMtpFeaturesHeaderBytes = 64;
inline constexpr uint32_t kMtpFeaturesRecordHeaderBytes = 48;
inline constexpr uint32_t kMtpFeaturesEndianLittle = 1;
inline constexpr uint32_t kMtpFeaturesElementBytes = 4;       // f32
inline constexpr uint32_t kMtpFeaturesKindResidual = 1u << 0;
inline constexpr uint32_t kMtpFeaturesKindLogits = 1u << 1;   // reserved; v1 never writes it
inline constexpr uint32_t kMtpFeaturesSeamPromptChunk = 1;
inline constexpr uint32_t kMtpFeaturesSeamPostVerify = 2;

static_assert(std::endian::native == std::endian::little,
              "the STFH feature trace is defined little-endian; this build changes every payload's byte order");

#pragma pack(push, 1)
struct MtpFeaturesHeader {
    uint32_t magic = kMtpFeaturesMagic;                             // @0
    uint16_t version = kMtpFeaturesVersion;                         // @4
    uint16_t header_bytes = kMtpFeaturesHeaderBytes;                // @6
    uint32_t flags = 0;                                             // @8 (no flag is defined in v1)
    uint32_t record_header_bytes = kMtpFeaturesRecordHeaderBytes;   // @12
    uint32_t record_kinds = kMtpFeaturesKindResidual;               // @16
    uint32_t endian = kMtpFeaturesEndianLittle;                     // @20
    uint32_t element_bytes = kMtpFeaturesElementBytes;              // @24
    uint32_t reserved0 = 0;                                         // @28
    uint64_t width = 0;                                             // @32: hc * n_embd floats per position
    uint64_t layers = 0;                                            // @40
    uint64_t vocab = 0;                                             // @48
    uint64_t reserved1 = 0;                                         // @56
};

struct MtpFeatureRecordHeader {
    uint32_t kind = kMtpFeaturesKindResidual;   // @0
    uint32_t seam = 0;                          // @4: kMtpFeaturesSeamPromptChunk or kMtpFeaturesSeamPostVerify
    uint64_t request = 0;                       // @8: 1-based ordinal of the request that produced the rows
    uint64_t position = 0;                      // @16: absolute position of row 0
    uint32_t rows = 0;                          // @24
    uint32_t width = 0;                         // @28: must equal the file header's width
    uint64_t sequence = 0;                      // @32: 0-based ordinal of this seam within this request
    uint64_t payload_bytes = 0;                 // @40: rows * width * 4
};
#pragma pack(pop)

static_assert(sizeof(MtpFeaturesHeader) == kMtpFeaturesHeaderBytes, "the STFH header must stay 64 bytes");
static_assert(sizeof(MtpFeatureRecordHeader) == kMtpFeaturesRecordHeaderBytes,
              "the STFH record header must stay 48 bytes");
static_assert(offsetof(MtpFeaturesHeader, width) == 32, "STFH header layout changed");
static_assert(offsetof(MtpFeaturesHeader, layers) == 40, "STFH header layout changed");
static_assert(offsetof(MtpFeatureRecordHeader, position) == 16, "STFH record layout changed");
static_assert(offsetof(MtpFeatureRecordHeader, sequence) == 32, "STFH record layout changed");
static_assert(offsetof(MtpFeatureRecordHeader, payload_bytes) == 40, "STFH record layout changed");

inline bool valid_mtp_features_header(const MtpFeaturesHeader& h, std::string* error) {
    auto fail = [&](const char* why) {
        if (error != nullptr) *error = why;
        return false;
    };
    if (h.magic != kMtpFeaturesMagic) return fail("the STFH magic is not \"STFH\"");
    if (h.version != kMtpFeaturesVersion) return fail("unsupported STFH version");
    if (h.header_bytes != kMtpFeaturesHeaderBytes) return fail("the STFH header size is not 64");
    if (h.record_header_bytes != kMtpFeaturesRecordHeaderBytes) return fail("the STFH record-header size is not 48");
    if (h.endian != kMtpFeaturesEndianLittle) return fail("the STFH trace is not little-endian");
    if (h.element_bytes != kMtpFeaturesElementBytes) return fail("the STFH elements are not 4-byte floats");
    if ((h.record_kinds & kMtpFeaturesKindResidual) == 0)
        return fail("the STFH trace does not announce residual records");
    if (h.width == 0) return fail("the STFH width is zero");
    if (h.width > UINT32_MAX) return fail("the STFH width does not fit the record header");
    return true;
}

inline bool valid_mtp_feature_record(const MtpFeatureRecordHeader& r, uint64_t file_width, std::string* error) {
    auto fail = [&](const char* why) {
        if (error != nullptr) *error = why;
        return false;
    };
    if (r.kind != kMtpFeaturesKindResidual) return fail("unsupported STFH record kind");
    if (r.seam != kMtpFeaturesSeamPromptChunk && r.seam != kMtpFeaturesSeamPostVerify)
        return fail("unknown STFH seam");
    if (r.rows == 0) return fail("an STFH record holds no rows");
    if (r.width == 0) return fail("an STFH record has zero width");
    if (r.width != file_width) return fail("the STFH record width does not match the file header");
    const uint64_t cells = (uint64_t) r.rows * (uint64_t) r.width;
    if (cells > UINT64_MAX / kMtpFeaturesElementBytes) return fail("the STFH payload size overflows");
    if (r.payload_bytes != cells * kMtpFeaturesElementBytes) return fail("the STFH payload size is wrong");
    return true;
}

/// Write a header after validating it.  A rejected header writes nothing, so the file position is unchanged.
inline bool write_mtp_features_header(std::FILE* f, const MtpFeaturesHeader& h, std::string* error) {
    if (f == nullptr) {
        if (error != nullptr) *error = "the STFH trace file is not open";
        return false;
    }
    std::string why;
    if (!valid_mtp_features_header(h, &why)) {
        if (error != nullptr) *error = why;
        return false;
    }
    if (std::fwrite(&h, sizeof h, 1, f) != 1) {
        if (error != nullptr) *error = "cannot write the STFH header";
        return false;
    }
    return true;
}

/// One complete residual record: its header, then `rows * width` floats.  A rejected record writes nothing.
inline bool write_mtp_feature_record(std::FILE* f, const MtpFeatureRecordHeader& r, const float* rows,
                                     std::string* error) {
    if (f == nullptr) {
        if (error != nullptr) *error = "the STFH trace file is not open";
        return false;
    }
    std::string why;
    if (!valid_mtp_feature_record(r, r.width, &why)) {
        if (error != nullptr) *error = why;
        return false;
    }
    const size_t floats = (size_t) r.rows * (size_t) r.width;
    if (std::fwrite(&r, sizeof r, 1, f) != 1 ||
        (floats > 0 && (rows == nullptr || std::fwrite(rows, sizeof(float), floats, f) != floats))) {
        if (error != nullptr) *error = "cannot write the STFH residual record";
        return false;
    }
    return true;
}

/// The trace's file, its request/sequence bookkeeping and its buffers.  One record may be written whole
/// (`residual`) or in host slices (`begin_residual` / `append_rows` / `end_residual`); a device row set larger
/// than a host staging buffer is copied and appended slice by slice, so the host allocation stays bounded.
/// Calls are serialized; the engine can capture from a prompt-reading thread while the main thread is not writing.
class MtpFeatureTrace {
public:
    MtpFeatureTrace() = default;
    ~MtpFeatureTrace() {
        std::lock_guard<std::mutex> lock(mutex_);
        close_locked(nullptr);
    }
    MtpFeatureTrace(const MtpFeatureTrace&) = delete;
    MtpFeatureTrace& operator=(const MtpFeatureTrace&) = delete;

    bool open(const std::string& path, uint64_t width, uint64_t layers, uint64_t vocab, std::string& err) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_ != nullptr) {
            err = "the STFH feature trace is already open";
            return false;
        }
        if (width == 0 || width > UINT32_MAX) {
            err = "the STFH width must be 1..4294967295 floats (" + std::to_string(width) + ")";
            return false;
        }
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) {
            err = "cannot open " + path;
            return false;
        }
        buffer_.resize(kBufferBytes);
        if (std::setvbuf(f, buffer_.data(), _IOFBF, buffer_.size()) != 0) buffer_.clear();
        MtpFeaturesHeader h;
        h.width = width;
        h.layers = layers;
        h.vocab = vocab;
        std::string why;
        if (!valid_mtp_features_header(h, &why)) {
            std::fclose(f);
            err = "the STFH header is invalid: " + why;
            return false;
        }
        if (std::fwrite(&h, sizeof h, 1, f) != 1) {
            std::fclose(f);
            err = "cannot write the STFH header to " + path;
            return false;
        }
        file_ = f;
        width_ = width;
        failed_.clear();
        records_ = 0;
        payload_floats_ = 0;
        last_request_ = 0;
        current_request_ = 0;
        request_pending_ = true;
        sequence_[0] = sequence_[1] = 0;
        open_record_ = false;
        open_seam_ = 0;
        open_rows_left_ = 0;
        open_floats_left_ = 0;
        open_floats_total_ = 0;
        return true;
    }

    bool enabled() const { return file_ != nullptr; }
    uint64_t width() const { return width_; }
    uint64_t records() const { return records_; }
    uint64_t payload_floats() const { return payload_floats_; }
    uint64_t last_request_id() const { return last_request_; }

    /// Start a new request: its ordinal is assigned by the first record that follows, so ordinals count requests
    /// with captured rows, and `sequence` restarts at 0 for each seam.
    bool begin_request(std::string& err) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_ == nullptr) return true;
        if (!failed_.empty()) {
            err = failed_;
            return false;
        }
        if (open_record_) {
            failed_ = "an STFH residual record was still open at a request boundary";
            err = failed_;
            return false;
        }
        request_pending_ = true;
        current_request_ = 0;
        sequence_[0] = sequence_[1] = 0;
        if (std::fflush(file_) != 0) return fail_locked("cannot flush the STFH trace at a request boundary", err);
        return true;
    }

    /// One whole residual record from host rows (`rows * width` floats).
    bool residual(uint32_t seam, int64_t position, const float* host_rows, uint32_t rows, std::string& err) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (host_rows == nullptr && rows > 0) {
            err = "the STFH residual rows are null";
            return false;
        }
        MtpFeatureRecordHeader r;
        if (!prepare_locked(seam, position, rows, r, err)) return false;
        const size_t floats = (size_t) rows * (size_t) width_;
        if (std::fwrite(&r, sizeof r, 1, file_) != 1 ||
            (floats > 0 && std::fwrite(host_rows, sizeof(float), floats, file_) != floats))
            return fail_locked("cannot write the STFH residual record", err);
        if (std::fflush(file_) != 0) return fail_locked("cannot flush the STFH residual record", err);
        records_ += 1;
        payload_floats_ += (uint64_t) floats;
        ++sequence_[seam - 1];
        return true;
    }

    /// Begin a residual record whose payload is appended in slices.  `rows` is the record's total.
    bool begin_residual(uint32_t seam, int64_t position, uint32_t rows, std::string& err) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (open_record_) {
            err = "an STFH residual record is already open";
            return false;
        }
        MtpFeatureRecordHeader r;
        if (!prepare_locked(seam, position, rows, r, err)) return false;
        if (std::fwrite(&r, sizeof r, 1, file_) != 1)
            return fail_locked("cannot write the STFH residual record header", err);
        open_record_ = true;
        open_seam_ = seam;
        open_rows_left_ = rows;
        open_floats_left_ = (uint64_t) rows * width_;
        open_floats_total_ = open_floats_left_;
        return true;
    }

    /// Append `rows` host rows to the open record (at most what the record announced).
    bool append_rows(const float* host_rows, uint32_t rows, std::string& err) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_ == nullptr) {
            err = "the STFH feature trace is not open";
            return false;
        }
        if (!failed_.empty()) {
            err = failed_;
            return false;
        }
        if (!open_record_) {
            err = "no STFH residual record is open";
            return false;
        }
        if (rows > open_rows_left_) {
            err = "an STFH residual append exceeds the record's announced rows";
            return false;
        }
        if (host_rows == nullptr && rows > 0) {
            err = "the STFH residual rows are null";
            return false;
        }
        const size_t floats = (size_t) rows * (size_t) width_;
        if (floats > 0 && std::fwrite(host_rows, sizeof(float), floats, file_) != floats)
            return fail_locked("cannot write the STFH residual payload", err);
        open_rows_left_ -= rows;
        open_floats_left_ -= (uint64_t) floats;
        return true;
    }

    /// Finish the open record; it must have received every announced row.
    bool end_residual(std::string& err) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_ == nullptr) {
            err = "the STFH feature trace is not open";
            return false;
        }
        if (!failed_.empty()) {
            err = failed_;
            return false;
        }
        if (!open_record_) {
            err = "no STFH residual record is open";
            return false;
        }
        if (open_rows_left_ != 0 || open_floats_left_ != 0) {
            err = "the STFH residual record is short of its announced rows";
            return false;
        }
        if (std::fflush(file_) != 0) return fail_locked("cannot flush the STFH residual record", err);
        records_ += 1;
        payload_floats_ += open_floats_total_;
        open_record_ = false;
        ++sequence_[open_seam_ - 1];
        return true;
    }

    /// Mark the trace unusable after a caller-side failure (for example a device-to-host copy), so a truncated
    /// record cannot be mistaken for a complete one and every later capture fails instead of appending.
    void abort(std::string why) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_.empty()) failed_ = std::move(why);
        request_pending_ = false;
        open_record_ = false;
    }

    bool flush(std::string& err) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_ == nullptr) return true;
        if (std::fflush(file_) != 0) return fail_locked("cannot flush the STFH trace", err);
        return true;
    }

    /// Flush and close.  False when any write failed or a record was left open; the message is sticky.
    bool close(std::string& err) {
        std::lock_guard<std::mutex> lock(mutex_);
        return close_locked(&err);
    }

    static constexpr size_t kBufferBytes = 1u << 20;

private:
    bool prepare_locked(uint32_t seam, int64_t position, uint32_t rows, MtpFeatureRecordHeader& r,
                        std::string& err) {
        if (file_ == nullptr) {
            err = "the STFH feature trace is not open";
            return false;
        }
        if (!failed_.empty()) {
            err = failed_;
            return false;
        }
        if (open_record_) {
            err = "an STFH residual record is already open";
            return false;
        }
        if (seam != kMtpFeaturesSeamPromptChunk && seam != kMtpFeaturesSeamPostVerify) {
            err = "the STFH seam must be prompt chunk (1) or post verify (2)";
            return false;
        }
        if (position < 0) {
            err = "the STFH position is negative";
            return false;
        }
        if (rows == 0) {
            err = "an STFH residual record needs at least one row";
            return false;
        }
        if (request_pending_) {
            ++last_request_;
            current_request_ = last_request_;
            request_pending_ = false;
            sequence_[0] = sequence_[1] = 0;
        }
        r.kind = kMtpFeaturesKindResidual;
        r.seam = seam;
        r.request = current_request_;
        r.position = (uint64_t) position;
        r.rows = rows;
        r.width = (uint32_t) width_;
        r.sequence = sequence_[seam - 1];
        r.payload_bytes = (uint64_t) rows * width_ * kMtpFeaturesElementBytes;
        std::string why;
        if (!valid_mtp_feature_record(r, width_, &why)) {
            err = "the STFH record is invalid: " + why;
            return false;
        }
        return true;
    }

    bool fail_locked(const char* why, std::string& err) {
        if (failed_.empty()) failed_ = why;
        err = failed_;
        open_record_ = false;
        return false;
    }

    bool close_locked(std::string* err) {
        if (file_ == nullptr) {
            if (err != nullptr) *err = failed_;
            return failed_.empty();
        }
        if (open_record_ && failed_.empty()) failed_ = "an STFH residual record was left open";
        open_record_ = false;
        std::FILE* f = file_;
        file_ = nullptr;
        const bool flushed = std::fflush(f) == 0;
        const bool closed = std::fclose(f) == 0;
        if ((!flushed || !closed) && failed_.empty()) failed_ = "cannot flush and close the STFH trace";
        if (err != nullptr) *err = failed_;
        return failed_.empty();
    }

    mutable std::mutex mutex_;
    std::FILE* file_ = nullptr;
    std::vector<char> buffer_;
    std::string failed_;
    uint64_t width_ = 0;
    uint64_t records_ = 0;
    uint64_t payload_floats_ = 0;
    uint64_t last_request_ = 0;
    uint64_t current_request_ = 0;
    uint64_t sequence_[2] = {0, 0};
    bool request_pending_ = true;
    bool open_record_ = false;
    uint32_t open_seam_ = 0;
    uint32_t open_rows_left_ = 0;
    uint64_t open_floats_left_ = 0;
    uint64_t open_floats_total_ = 0;
};

}  // namespace strata::trace
