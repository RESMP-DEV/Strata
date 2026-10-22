#include "strata/core/mtp_format.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {
namespace mf = strata::core::mtp_format;

bool expect(bool condition, const char* what) {
    if (!condition) {
        std::printf("mtp_format_test: FAILED %s\n", what);
        return false;
    }
    return true;
}

mf::Tensor make(const char* name, const char* kind, int64_t rows, int64_t cols, int64_t off, int64_t bytes) {
    return {name, kind, rows, cols, off, bytes};
}

}  // namespace

int main() {
    bool ok = true;
    std::string err;
    std::vector<mf::Tensor> valid{
        make("fc_embedding.weight", "iq4_xs", 2560, 2560, 0, 2560 * 10 * 136),
        make("shared.weight", "iq4_nl", 2560, 640, 2560 * 10 * 136, 2560 * 20 * 18),
        make("norm.weight", "f32", 1, 256, 2560 * 10 * 136 + 2560 * 20 * 18, 1024),
    };
    ok = expect(mf::validate(valid, 2560 * 10 * 136 + 2560 * 20 * 18 + 1024, err), "valid mixed manifest") && ok;

    const mf::Tensor unaligned = make("down.weight", "iq4_xs", 2560, 640, 0, 2560 * 3 * 136);
    int64_t ignored = 0;
    ok = expect(!mf::expected_bytes(unaligned, ignored), "IQ4_XS rejects a 640-column tensor") && ok;

    std::vector<mf::Tensor> duplicate{valid};
    duplicate.push_back(valid[0]);
    err.clear();
    ok = expect(!mf::validate(duplicate, 1 << 30, err), "duplicate tensor names are rejected") && ok;

    err.clear();
    std::vector<mf::Tensor> out_of_bounds{make("late.weight", "iq4_nl", 1, 32, 256, 18)};
    ok = expect(!mf::validate(out_of_bounds, 256, err), "out-of-bounds tensor is rejected") && ok;

    // The MMVQ contract asks for a 4-byte aligned pointer, not the 256-byte stride tools/mtp_rt.py writes.
    err.clear();
    std::vector<mf::Tensor> four_byte_aligned{make("odd.weight", "iq4_nl", 1, 32, 4, 18)};
    ok = expect(mf::validate(four_byte_aligned, 64, err), "a 4-byte aligned offset is accepted") && ok;

    err.clear();
    std::vector<mf::Tensor> misaligned{make("odd.weight", "iq4_nl", 1, 32, 2, 18)};
    ok = expect(!mf::validate(misaligned, 64, err), "a 2-byte aligned offset is rejected") && ok;

    std::printf("mtp_format_test: all ok\n");
    return ok ? 0 : 1;
}
