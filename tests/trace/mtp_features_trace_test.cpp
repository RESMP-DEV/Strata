// The STFH v1 feature trace: header contract, residual round trips, request/sequence bookkeeping and every refusal
// that can leave a reader with a record it cannot frame.  Synthetic rows only: no model, no GPU, no CUDA.

#include "strata/trace/mtp_features.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace {

using strata::trace::MtpFeatureRecordHeader;
using strata::trace::MtpFeaturesHeader;
using strata::trace::MtpFeatureTrace;

bool expect(bool ok, const char* what) {
    if (!ok) std::fprintf(stderr, "mtp_features_trace_test: failed: %s\n", what);
    return ok;
}

bool read_exact(std::FILE* f, void* p, size_t n) {
    return n == 0 || std::fread(p, 1, n, f) == n;
}

std::string temp_path(const char* tag) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    const std::string name = std::string("strata_mtp_features_") + tag + "_" + std::to_string(now) + ".bin";
    return (ec ? std::filesystem::path(name) : dir / name).string();
}

struct ParsedFile {
    MtpFeaturesHeader header{};
    std::vector<MtpFeatureRecordHeader> records;
    std::vector<std::vector<float>> payloads;
};

bool parse(const std::string& path, ParsedFile& out, std::string& error) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        error = "cannot reopen " + path;
        return false;
    }
    bool ok = read_exact(f, &out.header, sizeof out.header);
    if (!ok) error = "short header read";
    std::string why;
    if (ok && !strata::trace::valid_mtp_features_header(out.header, &why)) {
        ok = false;
        error = "invalid header: " + why;
    }
    while (ok) {
        MtpFeatureRecordHeader h{};
        const size_t got = std::fread(&h, 1, sizeof h, f);
        if (got == 0 && std::feof(f)) break;
        if (got != sizeof h) {
            ok = false;
            error = "short record-header read";
            break;
        }
        if (!strata::trace::valid_mtp_feature_record(h, out.header.width, &why)) {
            ok = false;
            error = "invalid record: " + why;
            break;
        }
        std::vector<float> rows((size_t) h.rows * (size_t) h.width);
        if (!read_exact(f, rows.data(), rows.size() * sizeof(float))) {
            ok = false;
            error = "short payload read";
            break;
        }
        out.records.push_back(h);
        out.payloads.push_back(std::move(rows));
    }
    if (ok) {   // a framed record must end exactly at EOF
        const int extra = std::fgetc(f);
        if (extra != EOF) {
            ok = false;
            error = "trailing bytes after the last record";
        }
    }
    std::fclose(f);
    return ok;
}

bool header_contract() {
    bool ok = true;
    ok = expect(sizeof(MtpFeaturesHeader) == 64, "the STFH header is not 64 bytes") && ok;
    ok = expect(sizeof(MtpFeatureRecordHeader) == 48, "the STFH record header is not 48 bytes") && ok;
    ok = expect(offsetof(MtpFeaturesHeader, width) == 32, "the STFH width is not at offset 32") && ok;
    ok = expect(offsetof(MtpFeatureRecordHeader, payload_bytes) == 40,
                "the STFH payload size is not at offset 40") &&
         ok;

    std::FILE* f = std::tmpfile();
    ok = expect(f != nullptr, "cannot make a temporary file") && ok;
    if (f == nullptr) return false;

    MtpFeaturesHeader h;
    h.width = 6;
    h.layers = 3;
    h.vocab = 11;
    std::string error;
    ok = expect(strata::trace::write_mtp_features_header(f, h, &error), "the valid header was refused") && ok;
    ok = expect(std::ftell(f) == 64, "the header does not occupy 64 bytes") && ok;

    // The first bytes are the complete self-description a reader trusts before it reads any geometry.
    std::rewind(f);
    std::array<unsigned char, 8> first{};
    ok = expect(read_exact(f, first.data(), first.size()), "short first-header-bytes read") && ok;
    const std::array<unsigned char, 8> expected{0x53, 0x54, 0x46, 0x48, 0x01, 0x00, 0x40, 0x00};
    ok = expect(first == expected, "the STFH magic/version/header-size bytes do not match the layout") && ok;

    std::rewind(f);
    MtpFeaturesHeader back{};
    ok = expect(read_exact(f, &back, sizeof back), "short header read") && ok;
    ok = expect(back.magic == strata::trace::kMtpFeaturesMagic, "the header magic did not round-trip") && ok;
    ok = expect(back.version == strata::trace::kMtpFeaturesVersion, "the header version did not round-trip") && ok;
    ok = expect(back.header_bytes == 64, "the header size field did not round-trip") && ok;
    ok = expect(back.record_header_bytes == 48, "the record-header size field did not round-trip") && ok;
    ok = expect(back.record_kinds == strata::trace::kMtpFeaturesKindResidual,
                "the record-kind bitmap does not announce residuals") &&
         ok;
    ok = expect(back.endian == strata::trace::kMtpFeaturesEndianLittle, "the endian field did not round-trip") && ok;
    ok = expect(back.element_bytes == 4, "the element size field did not round-trip") && ok;
    ok = expect(back.width == 6 && back.layers == 3 && back.vocab == 11,
                "the geometry fields did not round-trip") &&
         ok;

    std::array<float, 12> rows{};
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = (float) i + 0.5f;
    MtpFeatureRecordHeader r;
    r.seam = strata::trace::kMtpFeaturesSeamPromptChunk;
    r.request = 4;
    r.position = 100;
    r.rows = 2;
    r.width = 6;
    r.sequence = 7;
    r.payload_bytes = 2 * 6 * 4;
    error.clear();
    ok = expect(strata::trace::write_mtp_feature_record(f, r, rows.data(), &error),
                "the valid residual record was refused") &&
         ok;

    std::rewind(f);
    ok = expect(std::fseek(f, 64, SEEK_SET) == 0, "cannot seek past the header") && ok;
    MtpFeatureRecordHeader rb{};
    ok = expect(read_exact(f, &rb, sizeof rb), "short record-header read") && ok;
    ok = expect(rb.kind == strata::trace::kMtpFeaturesKindResidual, "the record kind did not round-trip") && ok;
    ok = expect(rb.seam == strata::trace::kMtpFeaturesSeamPromptChunk, "the record seam did not round-trip") && ok;
    ok = expect(rb.request == 4, "the record request did not round-trip") && ok;
    ok = expect(rb.position == 100, "the record position did not round-trip") && ok;
    ok = expect(rb.rows == 2 && rb.width == 6, "the record geometry did not round-trip") && ok;
    ok = expect(rb.sequence == 7, "the record sequence did not round-trip") && ok;
    ok = expect(rb.payload_bytes == 48, "the record payload size did not round-trip") && ok;
    std::array<float, 12> decoded{};
    ok = expect(read_exact(f, decoded.data(), decoded.size() * sizeof(float)), "short residual read") && ok;
    ok = expect(decoded == rows, "the residual floats did not round-trip") && ok;
    std::fclose(f);

    // Every invalid header is refused before a byte is written.
    auto refused_header = [&](MtpFeaturesHeader bad, const char* what) {
        std::FILE* t = std::tmpfile();
        bool good = expect(t != nullptr, "cannot make a temporary header file");
        if (t != nullptr) {
            std::string why;
            good = expect(!strata::trace::write_mtp_features_header(t, bad, &why), what) && good;
            good = expect(std::ftell(t) == 0, "a refused header wrote bytes") && good;
            std::fclose(t);
        }
        return good;
    };
    MtpFeaturesHeader bad = h;
    bad.magic = 0;
    ok = refused_header(bad, "a bad magic was accepted") && ok;
    bad = h;
    bad.version = 2;
    ok = refused_header(bad, "an unknown version was accepted") && ok;
    bad = h;
    bad.header_bytes = 63;
    ok = refused_header(bad, "a short header size was accepted") && ok;
    bad = h;
    bad.record_header_bytes = 47;
    ok = refused_header(bad, "a short record-header size was accepted") && ok;
    bad = h;
    bad.endian = 2;
    ok = refused_header(bad, "a big-endian trace was accepted") && ok;
    bad = h;
    bad.element_bytes = 8;
    ok = refused_header(bad, "an eight-byte element was accepted") && ok;
    bad = h;
    bad.width = 0;
    ok = refused_header(bad, "a zero width was accepted") && ok;
    bad = h;
    bad.record_kinds = strata::trace::kMtpFeaturesKindLogits;
    ok = refused_header(bad, "a logits-only trace was accepted") && ok;

    // Record refusals, including the payload-size overflow that a naive rows * width * 4 check misses.
    auto bad_record = [&](MtpFeatureRecordHeader br, uint64_t file_width, const char* what) {
        std::string why;
        return expect(!strata::trace::valid_mtp_feature_record(br, file_width, &why), what);
    };
    MtpFeatureRecordHeader br = r;
    br.width = 7;
    br.payload_bytes = 2 * 7 * 4;
    ok = bad_record(br, 6, "a record whose width differs from the header was accepted") && ok;
    br = r;
    br.rows = 0;
    br.payload_bytes = 0;
    ok = bad_record(br, 6, "a zero-row record was accepted") && ok;
    br = r;
    br.seam = 0;
    ok = bad_record(br, 6, "an unknown seam was accepted") && ok;
    br = r;
    br.kind = strata::trace::kMtpFeaturesKindLogits;
    ok = bad_record(br, 6, "a logits record was accepted by a v1 reader") && ok;
    br = r;
    br.payload_bytes = 47;
    ok = bad_record(br, 6, "a record with a wrong payload size was accepted") && ok;
    br = r;
    br.rows = UINT32_MAX;
    br.width = UINT32_MAX;
    br.payload_bytes = 0;
    ok = bad_record(br, UINT32_MAX, "an overflowing payload size was accepted") && ok;
    return ok;
}

bool class_round_trip() {
    bool ok = true;
    const std::string path = temp_path("roundtrip");
    MtpFeatureTrace trace;
    std::string error;
    ok = expect(trace.open(path, 3, 2, 7, error), "the trace did not open") && ok;
    ok = expect(trace.enabled(), "the trace does not report itself enabled") && ok;
    ok = expect(trace.width() == 3, "the trace width is wrong") && ok;

    const std::array<float, 6> a{1, 2, 3, 4, 5, 6};
    const std::array<float, 3> b{7, 8, 9};
    const std::array<float, 3> c{10, 11, 12};
    const std::array<float, 3> d{13, 14, 15};
    const std::array<float, 3> e{16, 17, 18};
    error.clear();
    ok = expect(trace.residual(strata::trace::kMtpFeaturesSeamPromptChunk, 0, a.data(), 2, error),
                "the first prompt-chunk record failed") &&
         ok;
    ok = expect(trace.residual(strata::trace::kMtpFeaturesSeamPostVerify, 2, b.data(), 1, error),
                "the first post-verify record failed") &&
         ok;
    ok = expect(trace.begin_request(error), "the second request did not begin") && ok;
    ok = expect(trace.residual(strata::trace::kMtpFeaturesSeamPromptChunk, 10, c.data(), 1, error),
                "the second request's prompt-chunk record failed") &&
         ok;
    ok = expect(trace.residual(strata::trace::kMtpFeaturesSeamPostVerify, 11, d.data(), 1, error),
                "the second request's first post-verify record failed") &&
         ok;
    ok = expect(trace.residual(strata::trace::kMtpFeaturesSeamPostVerify, 12, e.data(), 1, error),
                "the second request's second post-verify record failed") &&
         ok;
    ok = expect(trace.close(error), "the trace did not close cleanly") && ok;
    ok = expect(!trace.enabled(), "a closed trace still reports itself enabled") && ok;
    ok = expect(trace.records() == 5, "the record count is wrong") && ok;
    ok = expect(trace.payload_floats() == 18, "the payload float count is wrong") && ok;
    ok = expect(trace.last_request_id() == 2, "the captured-request ordinal is wrong") && ok;

    ParsedFile parsed;
    error.clear();
    ok = expect(parse(path, parsed, error), error.c_str()) && ok;
    if (ok) {
        ok = expect(parsed.header.width == 3 && parsed.header.layers == 2 && parsed.header.vocab == 7,
                    "the file header's geometry is wrong") &&
             ok;
        ok = expect(parsed.records.size() == 5, "the parsed record count is wrong") && ok;
        const uint64_t requests[5] = {1, 1, 2, 2, 2};
        const uint64_t positions[5] = {0, 2, 10, 11, 12};
        const uint64_t sequences[5] = {0, 0, 0, 0, 1};
        const uint32_t seams[5] = {strata::trace::kMtpFeaturesSeamPromptChunk,
                                   strata::trace::kMtpFeaturesSeamPostVerify,
                                   strata::trace::kMtpFeaturesSeamPromptChunk,
                                   strata::trace::kMtpFeaturesSeamPostVerify,
                                   strata::trace::kMtpFeaturesSeamPostVerify};
        const uint32_t row_counts[5] = {2, 1, 1, 1, 1};
        for (size_t i = 0; i < parsed.records.size() && i < 5; ++i) {
            ok = expect(parsed.records[i].request == requests[i], "the request ordinal is wrong") && ok;
            ok = expect(parsed.records[i].position == positions[i], "the position alignment is wrong") && ok;
            ok = expect(parsed.records[i].sequence == sequences[i], "the per-seam sequence is wrong") && ok;
            ok = expect(parsed.records[i].seam == seams[i], "the seam is wrong") && ok;
            ok = expect(parsed.records[i].rows == row_counts[i], "the row count is wrong") && ok;
        }
        ok = expect(parsed.payloads[0] == std::vector<float>(a.begin(), a.end()),
                    "the first record's rows changed") &&
             ok;
        ok = expect(parsed.payloads[1] == std::vector<float>(b.begin(), b.end()),
                    "the post-verify record's rows changed") &&
             ok;
        ok = expect(parsed.payloads[4] == std::vector<float>(e.begin(), e.end()),
                    "the last record's rows changed") &&
             ok;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return ok;
}

bool class_streaming() {
    bool ok = true;
    const std::string path = temp_path("streaming");
    MtpFeatureTrace trace;
    std::string error;
    ok = expect(trace.open(path, 3, 2, 5, error), "the streaming trace did not open") && ok;
    ok = expect(trace.begin_request(error), "the streaming request did not begin") && ok;
    ok = expect(trace.begin_residual(strata::trace::kMtpFeaturesSeamPostVerify, 20, 3, error),
                "the streaming record did not begin") &&
         ok;
    const std::array<float, 3> first{1, 2, 3};
    const std::array<float, 6> rest{4, 5, 6, 7, 8, 9};
    ok = expect(trace.append_rows(first.data(), 1, error), "the first streaming slice failed") && ok;
    ok = expect(trace.append_rows(rest.data(), 2, error), "the second streaming slice failed") && ok;
    ok = expect(trace.end_residual(error), "the streaming record did not end") && ok;
    ok = expect(trace.close(error), "the streaming trace did not close cleanly") && ok;
    ok = expect(trace.records() == 1, "the streaming record was not counted") && ok;
    ok = expect(trace.payload_floats() == 9, "the streaming payload float count is wrong") && ok;

    ParsedFile parsed;
    error.clear();
    ok = expect(parse(path, parsed, error), error.c_str()) && ok;
    if (ok) {
        ok = expect(parsed.records.size() == 1, "the streaming record count is wrong") && ok;
        if (!parsed.records.empty()) {
            const MtpFeatureRecordHeader& r = parsed.records[0];
            ok = expect(r.request == 1 && r.position == 20 && r.rows == 3 && r.width == 3,
                        "the streaming record's geometry is wrong") &&
                 ok;
            const std::vector<float> all{1, 2, 3, 4, 5, 6, 7, 8, 9};
            ok = expect(parsed.payloads[0] == all, "the streamed rows were not concatenated") && ok;
        }
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return ok;
}

bool class_refusals() {
    bool ok = true;
    const std::string path = temp_path("refusals");
    MtpFeatureTrace trace;
    std::string error;
    ok = expect(trace.open(path, 2, 1, 3, error), "the refusal trace did not open") && ok;
    const std::array<float, 4> rows{1, 2, 3, 4};
    error.clear();
    ok = expect(!trace.residual(0, 0, rows.data(), 1, error), "an unknown seam was accepted") && ok;
    error.clear();
    ok = expect(!trace.residual(strata::trace::kMtpFeaturesSeamPromptChunk, 0, rows.data(), 0, error),
                "a zero-row record was accepted") &&
         ok;
    error.clear();
    ok = expect(!trace.residual(strata::trace::kMtpFeaturesSeamPromptChunk, -1, rows.data(), 1, error),
                "a negative position was accepted") &&
         ok;
    error.clear();
    ok = expect(!trace.append_rows(rows.data(), 1, error), "an append without an open record was accepted") && ok;
    error.clear();
    ok = expect(trace.begin_residual(strata::trace::kMtpFeaturesSeamPostVerify, 0, 2, error),
                "the guarded record did not begin") &&
         ok;
    ok = expect(!trace.begin_residual(strata::trace::kMtpFeaturesSeamPostVerify, 0, 2, error),
                "a second open record was accepted") &&
         ok;
    error.clear();
    ok = expect(!trace.append_rows(rows.data(), 3, error), "a slice past the announced rows was accepted") && ok;
    ok = expect(trace.append_rows(rows.data(), 2, error), "the guarded record's legal append failed") && ok;
    ok = expect(trace.end_residual(error), "the guarded record did not finish") && ok;
    error.clear();
    ok = expect(trace.residual(strata::trace::kMtpFeaturesSeamPromptChunk, 4, rows.data(), 2, error),
                "the trace was unusable after the refused calls") &&
         ok;
    ok = expect(trace.close(error), "the refusal trace did not close cleanly") && ok;

    ParsedFile parsed;
    error.clear();
    ok = expect(parse(path, parsed, error), error.c_str()) && ok;
    ok = expect(parsed.records.size() == 2, "a refused call wrote a record") && ok;
    if (parsed.records.size() == 2)
        ok = expect(parsed.records[0].sequence == 0 && parsed.records[1].sequence == 0 &&
                        parsed.records[0].request == 1 && parsed.records[1].request == 1,
                    "the refused calls disturbed the ordinals") &&
             ok;
    std::error_code ec;
    std::filesystem::remove(path, ec);

    // A request boundary inside an open record poisons the trace: the record can never be framed, so every later
    // capture (and the close) must fail rather than append a plausible continuation.
    const std::string poisoned_path = temp_path("poisoned");
    MtpFeatureTrace poisoned;
    error.clear();
    ok = expect(poisoned.open(poisoned_path, 2, 1, 3, error), "the poisoned trace did not open") && ok;
    ok = expect(poisoned.begin_residual(strata::trace::kMtpFeaturesSeamPostVerify, 0, 2, error),
                "the poisoned record did not begin") &&
         ok;
    error.clear();
    ok = expect(!poisoned.begin_request(error), "a request boundary inside an open record was accepted") && ok;
    error.clear();
    ok = expect(!poisoned.append_rows(rows.data(), 2, error), "a poisoned trace accepted an append") && ok;
    error.clear();
    ok = expect(!poisoned.close(error) && !error.empty(), "a poisoned trace closed cleanly") && ok;
    std::filesystem::remove(poisoned_path, ec);

    // A close with an unannounced payload must fail, and it must say so instead of pretending the file is whole.
    const std::string short_path = temp_path("short");
    MtpFeatureTrace short_trace;
    error.clear();
    ok = expect(short_trace.open(short_path, 2, 1, 3, error), "the short-record trace did not open") && ok;
    ok = expect(short_trace.begin_residual(strata::trace::kMtpFeaturesSeamPromptChunk, 0, 2, error),
                "the short record did not begin") &&
         ok;
    error.clear();
    ok = expect(!short_trace.close(error) && !error.empty(), "closing a short record succeeded silently") && ok;
    std::filesystem::remove(short_path, ec);
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    ok = header_contract() && ok;
    ok = class_round_trip() && ok;
    ok = class_streaming() && ok;
    ok = class_refusals() && ok;
    if (!ok) std::fprintf(stderr, "mtp_features_trace_test: FAILED\n");
    return ok ? 0 : 1;
}
