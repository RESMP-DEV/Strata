#include "strata/runtime/accelerator_plugin.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

float bf16(uint16_t bits) {
    uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: plugin_loader_test <reference-plugin-library>\n";
        return 2;
    }
    try {
        bool missing_library_rejected = false;
        try {
            (void)strata::runtime::AcceleratorPlugin::load("/definitely/not/a/strata-plugin.so");
        } catch (const std::runtime_error&) {
            missing_library_rejected = true;
        }
        if (!missing_library_rejected) {
            std::cerr << "loader did not reject a missing library\n";
            return 1;
        }
        auto plugin = strata::runtime::AcceleratorPlugin::load(argv[1], "case=loader");
        if (plugin->abi() != 1 || plugin->device_name() != "reference-cpu" ||
            plugin->description().find("router_logits_bf16") == std::string::npos) {
            std::cerr << "plugin metadata mismatch\n";
            return 1;
        }
        const int64_t n_in = 3;
        const int64_t n_out = 2;
        const std::vector<uint16_t> weights{0x3f80, 0x4000, 0x4040, 0xc000, 0x4100, 0x4200};
        const std::vector<uint16_t> input{0x3f80, 0x4000, 0x4040, 0xc000, 0x4100, 0x4200};
        auto router = plugin->create_router(7, n_in, n_out, weights.data());
        std::vector<float> logits(static_cast<size_t>(n_out * 2), -1.0f);
        router.execute(2, input.data(), logits.data());

        const float expected[4] = {
            bf16(0x3f80) * bf16(0x3f80) + bf16(0x4000) * bf16(0x4000) + bf16(0x4040) * bf16(0x4040),
            bf16(0xc000) * bf16(0x3f80) + bf16(0x4100) * bf16(0x4000) + bf16(0x4200) * bf16(0x4040),
            bf16(0x3f80) * bf16(0xc000) + bf16(0x4000) * bf16(0x4100) + bf16(0x4040) * bf16(0x4200),
            bf16(0xc000) * bf16(0xc000) + bf16(0x4100) * bf16(0x4100) + bf16(0x4200) * bf16(0x4200),
        };
        for (size_t index = 0; index < logits.size(); ++index) {
            if (std::fabs(logits[index] - expected[index]) > 1e-6f) {
                std::cerr << "logit " << index << ": got " << logits[index] << ", expected " << expected[index] << "\n";
                return 1;
            }
        }
        bool invalid_dimensions_rejected = false;
        try {
            (void)plugin->create_router(8, 0, n_out, weights.data());
        } catch (const std::invalid_argument&) {
            invalid_dimensions_rejected = true;
        }
        if (!invalid_dimensions_rejected) {
            std::cerr << "router creation did not reject invalid dimensions\n";
            return 1;
        }
        std::cout << "plugin loader and BF16 router reference: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "plugin loader test failed: " << error.what() << "\n";
        return 1;
    }
}
