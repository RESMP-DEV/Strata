#include "strata/runtime/accelerator_plugin.hpp"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace strata::runtime {
namespace {

void close_library(void* library) {
    if (library == nullptr) return;
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(library));
#else
    dlclose(library);
#endif
}

void* open_library(const std::filesystem::path& path, std::string& error) {
#if defined(_WIN32)
    const std::wstring wide = path.wstring();
    HMODULE library = LoadLibraryW(wide.c_str());
    if (library == nullptr) {
        error = "LoadLibraryW failed with Windows error " + std::to_string(GetLastError());
        return nullptr;
    }
    return static_cast<void*>(library);
#else
    void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        const char* reason = dlerror();
        error = reason != nullptr ? reason : "dlopen failed without an error";
        return nullptr;
    }
    return library;
#endif
}

void* resolve_symbol(void* library, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

template <typename Signature>
Signature resolve_required(void* library, const char* name) {
    void* symbol = resolve_symbol(library, name);
    if (symbol == nullptr) {
        throw std::runtime_error(std::string("plugin is missing ") + name);
    }
    return reinterpret_cast<Signature>(symbol);
}

std::string nonempty(const char* value, const char* what) {
    if (value == nullptr || *value == '\0') {
        throw std::runtime_error(std::string("plugin returned an empty ") + what);
    }
    return value;
}

}  // namespace

AcceleratorRouter::AcceleratorRouter(const AcceleratorPluginApi* api, void* router) noexcept
    : api_(api), router_(router) {}

AcceleratorRouter::~AcceleratorRouter() {
    if (api_ != nullptr && router_ != nullptr) api_->destroy_router(router_);
}

AcceleratorRouter::AcceleratorRouter(AcceleratorRouter&& other) noexcept
    : api_(std::exchange(other.api_, nullptr)), router_(std::exchange(other.router_, nullptr)) {}

AcceleratorRouter& AcceleratorRouter::operator=(AcceleratorRouter&& other) noexcept {
    if (this != &other) {
        if (api_ != nullptr && router_ != nullptr) api_->destroy_router(router_);
        api_ = std::exchange(other.api_, nullptr);
        router_ = std::exchange(other.router_, nullptr);
    }
    return *this;
}

void AcceleratorRouter::execute(int64_t n_tokens, const uint16_t* input_bf16, float* logits_f32) const {
    if (api_ == nullptr || router_ == nullptr) throw std::logic_error("router is not loaded");
    if (n_tokens <= 0 || input_bf16 == nullptr || logits_f32 == nullptr) {
        throw std::invalid_argument("router execute requires positive n_tokens and nonnull tensors");
    }
    if (api_->execute_router(router_, n_tokens, input_bf16, logits_f32) != 0) {
        const char* reason = api_->last_error();
        throw std::runtime_error(reason != nullptr && *reason != '\0' ? reason : "router execution failed");
    }
}

AcceleratorPlugin::AcceleratorPlugin(AcceleratorPluginApi api, void* instance, std::string device,
                                     std::string description)
    : api_(api), instance_(instance), device_(std::move(device)), description_(std::move(description)) {}

AcceleratorPlugin::~AcceleratorPlugin() {
    if (instance_ != nullptr) api_.destroy(instance_);
    close_library(api_.library);
}

std::unique_ptr<AcceleratorPlugin> AcceleratorPlugin::load(const std::string& path, const std::string& options) {
    if (path.empty()) throw std::invalid_argument("accelerator plugin path is empty");
    std::filesystem::path library_path(path);
    if (!library_path.is_absolute()) library_path = std::filesystem::absolute(library_path);
    std::error_code filesystem_error;
    if (!std::filesystem::exists(library_path, filesystem_error) || filesystem_error) {
        throw std::runtime_error("accelerator plugin does not exist: " + library_path.string());
    }

    std::string open_error;
    void* library = open_library(library_path, open_error);
    if (library == nullptr) throw std::runtime_error(open_error);
    AcceleratorPluginApi api{};
    api.library = library;
    try {
        api.abi = resolve_required<uint32_t (*)()>(library, "strata_plugin_abi");
        api.device_name = resolve_required<const char* (*)()>(library, "strata_plugin_device_name");
        api.description = resolve_required<const char* (*)()>(library, "strata_plugin_description");
        api.create = resolve_required<void* (*)(const char*)>(library, "strata_plugin_create");
        api.destroy = resolve_required<void (*)(void*)>(library, "strata_plugin_destroy");
        api.create_router = resolve_required<void* (*)(void*, int64_t, int64_t, int64_t, const uint16_t*, const char*)>(
            library, "strata_plugin_create_router");
        api.destroy_router = resolve_required<void (*)(void*)>(library, "strata_plugin_destroy_router");
        api.execute_router = resolve_required<int32_t (*)(void*, int64_t, const uint16_t*, float*)>(
            library, "strata_plugin_execute_router");
        api.last_error = resolve_required<const char* (*)()>(library, "strata_plugin_last_error");
        if (api.abi() != kAcceleratorPluginAbi) {
            throw std::runtime_error("plugin ABI " + std::to_string(api.abi()) + " is not " +
                                     std::to_string(kAcceleratorPluginAbi));
        }
        const std::string device = nonempty(api.device_name(), "device name");
        const std::string description = nonempty(api.description(), "description");
        void* instance = api.create(options.c_str());
        if (instance == nullptr) {
            const char* reason = api.last_error();
            throw std::runtime_error(reason != nullptr && *reason != '\0' ? reason : "plugin create failed");
        }
        return std::unique_ptr<AcceleratorPlugin>(
            new AcceleratorPlugin(api, instance, std::move(device), std::move(description)));
    } catch (...) {
        close_library(library);
        throw;
    }
}

AcceleratorRouter AcceleratorPlugin::create_router(int64_t router_id, int64_t n_in, int64_t n_out,
                                                   const uint16_t* weights_bf16,
                                                   const std::string& options) const {
    if (n_in <= 0 || n_out <= 0 || weights_bf16 == nullptr) {
        throw std::invalid_argument("router creation requires positive dimensions and nonnull weights");
    }
    void* router = api_.create_router(instance_, router_id, n_in, n_out, weights_bf16, options.c_str());
    if (router == nullptr) {
        const char* reason = api_.last_error();
        throw std::runtime_error(reason != nullptr && *reason != '\0' ? reason : "router creation failed");
    }
    return AcceleratorRouter(&api_, router);
}

}  // namespace strata::runtime
