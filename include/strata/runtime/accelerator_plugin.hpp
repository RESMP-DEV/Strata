#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

/// Version 1 of the stable C ABI exposed by a Strata accelerator plugin.
///
/// The boundary is deliberately operation-specific rather than a generic tensor
/// graph. The first accelerator candidate is the MoE router projection: BF16
/// inputs and weights enter the plugin, synchronous F32 logits leave it, and
/// Strata retains top-k selection. A plugin must copy router weights during
/// create_router and treat execute_router as synchronous and stateless after
/// initialization. The owner serializes calls to one router.
namespace strata::runtime {

inline constexpr uint32_t kAcceleratorPluginAbi = 1;

struct AcceleratorPluginApi {
    void* library = nullptr;
    uint32_t (*abi)() = nullptr;
    const char* (*device_name)() = nullptr;
    const char* (*description)() = nullptr;
    void* (*create)(const char* options) = nullptr;
    void (*destroy)(void* instance) = nullptr;
    void* (*create_router)(void* instance, int64_t router_id, int64_t n_in, int64_t n_out,
                           const uint16_t* weights_bf16, const char* options) = nullptr;
    void (*destroy_router)(void* router) = nullptr;
    int32_t (*execute_router)(void* router, int64_t n_tokens, const uint16_t* input_bf16,
                              float* logits_f32) = nullptr;
    const char* (*last_error)() = nullptr;
};

class AcceleratorRouter {
public:
    AcceleratorRouter(const AcceleratorPluginApi* api, void* router) noexcept;
    ~AcceleratorRouter();
    AcceleratorRouter(const AcceleratorRouter&) = delete;
    AcceleratorRouter& operator=(const AcceleratorRouter&) = delete;
    AcceleratorRouter(AcceleratorRouter&& other) noexcept;
    AcceleratorRouter& operator=(AcceleratorRouter&& other) noexcept;

    void execute(int64_t n_tokens, const uint16_t* input_bf16, float* logits_f32) const;

private:
    const AcceleratorPluginApi* api_ = nullptr;
    void* router_ = nullptr;
};

class AcceleratorPlugin {
public:
    static std::unique_ptr<AcceleratorPlugin> load(const std::string& path, const std::string& options = {});
    ~AcceleratorPlugin();
    AcceleratorPlugin(const AcceleratorPlugin&) = delete;
    AcceleratorPlugin& operator=(const AcceleratorPlugin&) = delete;
    AcceleratorPlugin(AcceleratorPlugin&&) = delete;
    AcceleratorPlugin& operator=(AcceleratorPlugin&&) = delete;

    uint32_t abi() const { return api_.abi(); }
    const std::string& device_name() const { return device_; }
    const std::string& description() const { return description_; }
    AcceleratorRouter create_router(int64_t router_id, int64_t n_in, int64_t n_out,
                                    const uint16_t* weights_bf16, const std::string& options = {}) const;

private:
    AcceleratorPlugin(AcceleratorPluginApi api, void* instance, std::string device, std::string description);

    AcceleratorPluginApi api_{};
    void* instance_ = nullptr;
    std::string device_;
    std::string description_;
};

}  // namespace strata::runtime
