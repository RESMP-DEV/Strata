#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

thread_local std::string g_error;

float bf16(uint16_t bits) {
    uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

struct ReferenceRouter {
    int64_t n_in = 0;
    int64_t n_out = 0;
    std::vector<uint16_t> weights;
};

void clear_error() { g_error.clear(); }

}  // namespace

extern "C" {

uint32_t strata_plugin_abi() { return 1; }

const char* strata_plugin_device_name() { return "reference-cpu"; }

const char* strata_plugin_description() {
    return "{\"abi\":1,\"operations\":[\"router_logits_bf16\"]}";
}

void* strata_plugin_create(const char*) {
    clear_error();
    return new (std::nothrow) int{1};
}

void strata_plugin_destroy(void* instance) { delete static_cast<int*>(instance); }

void* strata_plugin_create_router(void*, int64_t, int64_t n_in, int64_t n_out,
                                  const uint16_t* weights, const char*) {
    clear_error();
    if (n_in <= 0 || n_out <= 0 || weights == nullptr) {
        g_error = "reference router requires positive dimensions and nonnull weights";
        return nullptr;
    }
    auto* router = new (std::nothrow) ReferenceRouter{};
    if (router == nullptr) {
        g_error = "reference router allocation failed";
        return nullptr;
    }
    router->n_in = n_in;
    router->n_out = n_out;
    router->weights.assign(weights, weights + static_cast<size_t>(n_in * n_out));
    return router;
}

void strata_plugin_destroy_router(void* router) { delete static_cast<ReferenceRouter*>(router); }

int32_t strata_plugin_execute_router(void* opaque, int64_t n_tokens, const uint16_t* input, float* logits) {
    clear_error();
    auto* router = static_cast<ReferenceRouter*>(opaque);
    if (router == nullptr || n_tokens <= 0 || input == nullptr || logits == nullptr) {
        g_error = "reference router arguments are invalid";
        return -1;
    }
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t output = 0; output < router->n_out; ++output) {
            float sum = 0.0f;
            for (int64_t input_index = 0; input_index < router->n_in; ++input_index) {
                const size_t weight_index = static_cast<size_t>(output * router->n_in + input_index);
                const size_t input_index_flat = static_cast<size_t>(token * router->n_in + input_index);
                sum += bf16(router->weights[weight_index]) * bf16(input[input_index_flat]);
            }
            logits[static_cast<size_t>(token * router->n_out + output)] = sum;
        }
    }
    return 0;
}

const char* strata_plugin_last_error() { return g_error.c_str(); }

}  // extern "C"
