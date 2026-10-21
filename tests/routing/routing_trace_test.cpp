#include "strata/routing/trace.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

bool read_exact(std::FILE* file, void* destination, size_t bytes) {
    return std::fread(destination, 1, bytes, file) == bytes;
}

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "routing_trace_test: %s\n", message);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    bool ok = true;
    std::FILE* file = std::tmpfile();
    ok = expect(file != nullptr, "cannot create a temporary trace") && ok;
    if (!ok) return 1;

    std::string error;
    ok = expect(
             strata::routing::write_trace_header(file, 48, 10, &error),
             "the valid header was refused"
         ) &&
         ok;

    const std::array<int32_t, 4> ids{{3, 511, 0, 10}};
    const std::array<float, 4> weights{{0.5f, 0.25f, 0.125f, 0.0625f}};
    ok = expect(
             strata::routing::write_batched_prefill_routing(
                 file, 1, 47, 13963, 2, 2, ids.data(), weights.data(), &error
             ),
             "the valid routing record was refused"
         ) &&
         ok;
    ok = expect(
             strata::routing::write_batched_prefill_routing(
                 file, 2, 47, 13963, 2, 2, ids.data(), weights.data(), &error
             ),
             "the duplicate-position routing record was refused"
         ) &&
         ok;

    const long before_invalid = std::ftell(file);
    ok = expect(before_invalid >= 0, "cannot tell the temporary trace position") && ok;
    ok = expect(
             !strata::routing::write_batched_prefill_routing(
                 file, 1, -1, 0, 1, 1, ids.data(), weights.data(), &error
             ),
             "a negative layer was accepted"
         ) &&
         ok;
    ok = expect(std::ftell(file) == before_invalid, "a rejected record changed the file") && ok;

    std::rewind(file);
    std::array<unsigned char, 40> header_bytes{};
    ok = expect(read_exact(file, header_bytes.data(), header_bytes.size()), "short header read") && ok;
    const std::array<unsigned char, 40> expected_header{
        0x53, 0x54, 0x52, 0x50, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    ok = expect(header_bytes == expected_header, "the header bytes do not match the documented layout") && ok;

    strata::routing::BatchedPrefillRecordHeader record{};
    ok = expect(read_exact(file, &record, sizeof record), "short record-header read") && ok;
    ok = expect(record.layer == 47, "the record layer did not round-trip") && ok;
    ok = expect(record.k == 2, "the record top-k did not round-trip") && ok;
    ok = expect(record.position == 13963, "the record position did not round-trip") && ok;
    ok = expect(record.token_count == 2, "the record token count did not round-trip") && ok;
    ok = expect(record.reserved == 0, "the record reserved field is not zero") && ok;
    ok = expect(record.request_id == 1, "the first request id did not round-trip") && ok;

    std::array<int32_t, 4> decoded_ids{};
    std::array<float, 4> decoded_weights{};
    ok = expect(read_exact(file, decoded_ids.data(), decoded_ids.size() * sizeof(int32_t)), "short ids read") &&
         ok;
    ok = expect(
             read_exact(file, decoded_weights.data(), decoded_weights.size() * sizeof(float)),
             "short weights read"
         ) &&
         ok;
    ok = expect(decoded_ids == ids, "the routed ids did not round-trip") && ok;
    ok = expect(decoded_weights == weights, "the router weights did not round-trip") && ok;

    strata::routing::BatchedPrefillRecordHeader duplicate{};
    ok = expect(read_exact(file, &duplicate, sizeof duplicate), "short duplicate record-header read") && ok;
    ok = expect(duplicate.layer == record.layer, "the duplicate record layer changed") && ok;
    ok = expect(duplicate.position == record.position, "the duplicate record position changed") && ok;
    ok = expect(duplicate.request_id == 2, "the second request id did not round-trip") && ok;
    std::array<int32_t, 4> duplicate_ids{};
    std::array<float, 4> duplicate_weights{};
    ok = expect(
             read_exact(file, duplicate_ids.data(), duplicate_ids.size() * sizeof(int32_t)) &&
                 read_exact(file, duplicate_weights.data(), duplicate_weights.size() * sizeof(float)),
             "short duplicate payload read"
         ) &&
         ok;
    ok = expect(duplicate_ids == ids, "the duplicate request ids payload changed") && ok;
    ok = expect(duplicate_weights == weights, "the duplicate request weights payload changed") && ok;

    std::fclose(file);
    if (!ok) std::fprintf(stderr, "routing_trace_test: %s\n", error.c_str());
    return ok ? 0 : 1;
}
