#include "strata/program/native_dump_guard.hpp"

#include <cassert>
#include <cstdint>
#include <string_view>
#include <vector>

int main() {
    using strata::program::native_dump_guard::accepted_rows;
    using strata::program::native_dump_guard::collides;

    assert(!collides("/tmp/logits.bin", "/tmp/residual.bin"));
    assert(collides("/tmp/logits.bin", "/tmp/logits.bin"));
    assert(collides("/tmp/logits.bin", "/tmp/logits.bin.windows.tsv"));
    assert(collides("./tmp/logits.bin", "tmp/logits.bin"));

    const std::vector<int64_t> eos_ids = {2, 3};
    const std::vector<int32_t> tokens = {7, 2, 9, 11};
    assert(accepted_rows(tokens, 4, false, eos_ids) == 4);
    assert(accepted_rows(tokens, 4, true, eos_ids) == 2);
    assert(accepted_rows(tokens, 4, true, {5}) == 4);
    assert(accepted_rows(tokens, -1, true, eos_ids) == 0);
    return 0;
}
