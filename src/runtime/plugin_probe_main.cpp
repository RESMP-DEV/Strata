#include "strata/runtime/accelerator_plugin.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

uint16_t value_bits(size_t index) {
    return static_cast<uint16_t>(0x3800 + ((index * 37u) & 0xffu));
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\b': escaped += "\\b"; break;
        case '\f': escaped += "\\f"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (static_cast<unsigned char>(character) < 0x20) {
                char code[7];
                std::snprintf(code, sizeof(code), "\\u%04x", static_cast<unsigned char>(character));
                escaped += code;
            } else {
                escaped += character;
            }
        }
    }
    return escaped;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::cerr << "usage: strata-plugin-probe <plugin> [n_in=2560] [n_out=512] [tokens=1] [tolerance=0.05]\n";
        return 2;
    }
    const int64_t n_in = argc > 2 ? std::strtoll(argv[2], nullptr, 10) : 2560;
    const int64_t n_out = argc > 3 ? std::strtoll(argv[3], nullptr, 10) : 512;
    const int64_t n_tokens = argc > 4 ? std::strtoll(argv[4], nullptr, 10) : 1;
    const float tolerance = argc > 5 ? std::strtof(argv[5], nullptr) : 0.05f;
    if (n_in <= 0 || n_out <= 0 || n_tokens <= 0 || tolerance < 0.0f) {
        std::cerr << "dimensions, token count, and tolerance must be nonnegative and dimensions positive\n";
        return 2;
    }

    try {
        const auto load_started = std::chrono::steady_clock::now();
        auto plugin = strata::runtime::AcceleratorPlugin::load(argv[1]);
        std::vector<uint16_t> weights(static_cast<size_t>(n_in * n_out));
        std::vector<uint16_t> input(static_cast<size_t>(n_in * n_tokens));
        std::vector<float> actual(static_cast<size_t>(n_out * n_tokens));
        std::vector<float> expected(static_cast<size_t>(n_out * n_tokens));
        for (size_t index = 0; index < weights.size(); ++index) weights[index] = value_bits(index);
        for (size_t index = 0; index < input.size(); ++index) input[index] = value_bits(index + 17);
        auto router = plugin->create_router(0, n_in, n_out, weights.data());
        const auto load_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_started).count();
        const auto execute_started = std::chrono::steady_clock::now();
        router.execute(n_tokens, input.data(), actual.data());
        const double execute_elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - execute_started).count();

        float max_error = 0.0f;
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t output = 0; output < n_out; ++output) {
                float sum = 0.0f;
                for (int64_t input_index = 0; input_index < n_in; ++input_index) {
                    const size_t weight_index = static_cast<size_t>(output * n_in + input_index);
                    const size_t input_index_flat = static_cast<size_t>(token * n_in + input_index);
                    sum += bf16(weights[weight_index]) * bf16(input[input_index_flat]);
                }
                const size_t result_index = static_cast<size_t>(token * n_out + output);
                max_error = std::max(max_error, std::fabs(sum - actual[result_index]));
                expected[result_index] = sum;
            }
        }
        const bool passed = max_error <= tolerance;
        std::cout << "{"
                  << "\"abi\":" << plugin->abi()
                  << ",\"device\":\"" << plugin->device_name() << "\""
                  << ",\"description\":\"" << json_escape(plugin->description()) << "\""
                  << ",\"n_in\":" << n_in
                  << ",\"n_out\":" << n_out
                  << ",\"tokens\":" << n_tokens
                  << ",\"max_abs_error\":" << max_error
                  << ",\"tolerance\":" << tolerance
                  << ",\"load_seconds\":" << load_elapsed
                  << ",\"execute_seconds\":" << execute_elapsed
                  << ",\"passed\":" << (passed ? "true" : "false")
                  << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "plugin probe failed: " << error.what() << "\n";
        return 1;
    }
}
