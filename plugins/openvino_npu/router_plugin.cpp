// OpenVINO implementation of Strata accelerator plugin ABI v1.
//
// Build and run explicitly on a host with the OpenVINO runtime; it is not built
// by Strata's default configure and the engine does not load it automatically.

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

thread_local std::string g_error;

using namespace ov::opset13;

struct OpenVinoPlugin {
    ov::Core core;
    std::string device;
};

struct RouterModel {
    ov::CompiledModel model;
    ov::InferRequest request;
    int64_t n_in = 0;
    int64_t n_out = 0;
};

uint16_t bf16_to_f16_bits(uint16_t bf16) {
    const uint32_t float_bits = static_cast<uint32_t>(bf16) << 16;
    float value = 0.0f;
    std::memcpy(&value, &float_bits, sizeof(value));
    const ov::float16 rounded(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &rounded, sizeof(bits));
    return bits;
}

std::string device_from_options(const char* options) {
    const std::string text = options != nullptr ? options : "";
    const std::string key = "device=";
    const size_t start = text.find(key);
    if (start == std::string::npos) return "NPU";
    const size_t value = start + key.size();
    const size_t end = text.find(',', value);
    std::string device = text.substr(value, end == std::string::npos ? end : end - value);
    return device.empty() ? "NPU" : device;
}

void set_error(const std::exception& error) { g_error = error.what(); }

}  // namespace

extern "C" {

uint32_t strata_plugin_abi() { return 1; }

const char* strata_plugin_device_name() {
    thread_local std::string name;
    try {
        ov::Core core;
        name = core.get_property("NPU", "FULL_DEVICE_NAME").as<std::string>();
    } catch (const std::exception&) {
        name = "OpenVINO(NPU)";
    }
    return name.c_str();
}

const char* strata_plugin_description() {
    return "{\"abi\":1,\"runtime\":\"OpenVINO\",\"operations\":[\"router_logits_bf16\"],"
           "\"runtime_precision\":\"f16\",\"input_abi_precision\":\"bf16\",\"max_tokens\":1}";
}

void convert_bf16_to_f16(const uint16_t* source, size_t count, std::vector<uint16_t>& destination) {
    destination.resize(count);
    for (size_t index = 0; index < count; ++index) destination[index] = bf16_to_f16_bits(source[index]);
}

void* strata_plugin_create(const char* options) {
    g_error.clear();
    try {
        auto* plugin = new (std::nothrow) OpenVinoPlugin{};
        if (plugin == nullptr) {
            g_error = "OpenVINO plugin allocation failed";
            return nullptr;
        }
        plugin->device = device_from_options(options);
        return plugin;
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    }
}

void strata_plugin_destroy(void* instance) { delete static_cast<OpenVinoPlugin*>(instance); }

void* strata_plugin_create_router(void* opaque, int64_t, int64_t n_in, int64_t n_out,
                                  const uint16_t* weights_bf16, const char* options) {
    g_error.clear();
    auto* plugin = static_cast<OpenVinoPlugin*>(opaque);
    if (plugin == nullptr || n_in <= 0 || n_out <= 0 || weights_bf16 == nullptr) {
        g_error = "OpenVINO router requires a plugin instance, positive dimensions, and weights";
        return nullptr;
    }
    try {
        std::vector<uint16_t> weights(weights_bf16, weights_bf16 + static_cast<size_t>(n_in * n_out));
        std::vector<uint16_t> weights_f16;
        convert_bf16_to_f16(weights.data(), weights.size(), weights_f16);
        auto input = std::make_shared<Parameter>(ov::element::f16, ov::Shape{1, static_cast<size_t>(n_in)});
        auto constant = std::make_shared<Constant>(ov::element::f16, ov::Shape{
            static_cast<size_t>(n_out), static_cast<size_t>(n_in)
        }, weights_f16.data());
        auto logits = std::make_shared<MatMul>(input, constant, false, true);
        auto converted = std::make_shared<Convert>(logits, ov::element::f32);
        auto result = std::make_shared<Result>(converted);
        ov::ParameterVector parameters{input};
        ov::ResultVector results{result};
        auto model = std::make_shared<ov::Model>(results, parameters, "StrataRouterLogits");
        ov::CompiledModel compiled = plugin->core.compile_model(
            model, plugin->device, ov::AnyMap{{ov::hint::inference_precision.name(), ov::element::f16}});
        RouterModel router{std::move(compiled), {}, n_in, n_out};
        router.request = router.model.create_infer_request();
        (void)options;
        return new (std::nothrow) RouterModel(std::move(router));
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    }
}

void strata_plugin_destroy_router(void* router) { delete static_cast<RouterModel*>(router); }

int32_t strata_plugin_execute_router(void* opaque, int64_t n_tokens, const uint16_t* input_bf16,
                                     float* logits_f32) {
    g_error.clear();
    auto* router = static_cast<RouterModel*>(opaque);
    if (router == nullptr || n_tokens != 1 || input_bf16 == nullptr || logits_f32 == nullptr) {
        g_error = "OpenVINO router execution arguments are invalid";
        return -1;
    }
    try {
        std::vector<uint16_t> input_f16;
        convert_bf16_to_f16(input_bf16, static_cast<size_t>(n_tokens * router->n_in), input_f16);
        ov::Tensor input(ov::element::f16, ov::Shape{
            static_cast<size_t>(n_tokens), static_cast<size_t>(router->n_in)
        }, input_f16.data());
        ov::Tensor output(ov::element::f32, ov::Shape{
            static_cast<size_t>(n_tokens), static_cast<size_t>(router->n_out)
        }, logits_f32);
        router->request.set_input_tensor(input);
        router->request.set_output_tensor(output);
        router->request.infer();
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
        return -1;
    }
}

const char* strata_plugin_last_error() { return g_error.c_str(); }

}  // extern "C"
