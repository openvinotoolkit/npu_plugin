//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstdint>

#include <openvino/runtime/properties.hpp>

#include "sync_infer_request.hpp"
#include "vpux/utils/IE/config.hpp"

namespace vpux {

using Uuid = ov::device::UUID;

//------------------------------------------------------------------------------
class IDevice;
class Device;

class IEngineBackend : public std::enable_shared_from_this<IEngineBackend> {
public:
    /** @brief Get device, which can be used for inference. Backend responsible for selection. */
    virtual const std::shared_ptr<IDevice> getDevice() const;
    /** @brief Search for a specific device by name */
    virtual const std::shared_ptr<IDevice> getDevice(const std::string& specificDeviceName) const;
    /** @brief Get device, which is configured/suitable for provided params */
    virtual const std::shared_ptr<IDevice> getDevice(const ov::AnyMap& paramMap) const;
    /** @brief Provide a list of names of all devices, with which user can work directly */
    virtual const std::vector<std::string> getDeviceNames() const;
    /** @brief Get name of backend */
    virtual const std::string getName() const = 0;
    /** @brief Register backend-specific options */
    virtual void registerOptions(OptionsDesc& options) const;

#ifndef OPENVINO_STATIC_LIBRARY
protected:
#endif
    ~IEngineBackend() = default;
};

class EngineBackend final {
public:
    EngineBackend() = default;
    EngineBackend(const EngineBackend&) = default;
    EngineBackend& operator=(const EngineBackend&) = default;

#ifdef OPENVINO_STATIC_LIBRARY
    EngineBackend(std::shared_ptr<IEngineBackend> impl);
#endif

#ifndef OPENVINO_STATIC_LIBRARY
    EngineBackend(const std::string& pathToLib, const Config& config);
#endif

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~EngineBackend() {
        _impl = {};
    }

    const std::shared_ptr<Device> getDevice() const;
    const std::shared_ptr<Device> getDevice(const std::string& specificDeviceName) const;
    const std::shared_ptr<Device> getDevice(const ov::AnyMap& paramMap) const;
    const std::vector<std::string> getDeviceNames() const {
        return _impl->getDeviceNames();
    }
    const std::string getName() const {
        return _impl->getName();
    }
    void registerOptions(OptionsDesc& options) const {
        _impl->registerOptions(options);
        options.addSharedObject(_so);
    }

private:
    std::shared_ptr<IEngineBackend> _impl;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::shared_ptr<void> _so;
};

//------------------------------------------------------------------------------

class Executor {
public:
    using Ptr = std::shared_ptr<Executor>;
    using CPtr = std::shared_ptr<const Executor>;

    virtual ~Executor() = default;
};

//------------------------------------------------------------------------------

class IDevice : public std::enable_shared_from_this<IDevice> {
public:
    virtual std::shared_ptr<Executor> createExecutor(const NetworkDescription::CPtr networkDescription,
                                                     const Config& config) = 0;

    virtual std::string getName() const = 0;
    virtual std::string getFullDeviceName() const = 0;
    virtual Uuid getUuid() const;
    virtual uint64_t getAllocMemSize() const;
    virtual uint64_t getTotalMemSize() const;
    virtual uint32_t getDriverVersion() const;

    virtual std::shared_ptr<SyncInferRequest> createInferRequest(
            const std::shared_ptr<const ov::ICompiledModel> compiledModel,
            const std::shared_ptr<const NetworkDescription> networkDescription, const Executor::Ptr executor,
            const Config& config) = 0;

protected:
    virtual ~IDevice() = default;
};

class Device final {
public:
    using Ptr = std::shared_ptr<Device>;
    using CPtr = std::shared_ptr<const Device>;

    Device(const std::shared_ptr<IDevice>& device, const std::shared_ptr<void>& so): _impl(device), _so(so) {
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~Device() {
        _impl = {};
    }

    std::shared_ptr<Executor> createExecutor(const NetworkDescription::CPtr networkDescription, const Config& config) {
        return _impl->createExecutor(networkDescription, config);
    }

    std::string getName() const {
        return _impl->getName();
    }

    std::string getFullDeviceName() const {
        return _impl->getFullDeviceName();
    }

    Uuid getUuid() const {
        return _impl->getUuid();
    }

    uint64_t getAllocMemSize() const {
        return _impl->getAllocMemSize();
    }

    uint64_t getTotalMemSize() const {
        return _impl->getTotalMemSize();
    }

    uint32_t getDriverVersion() const {
        return _impl->getDriverVersion();
    }

    std::shared_ptr<SyncInferRequest> createInferRequest(
            const std::shared_ptr<const ov::ICompiledModel> compiledModel,
            const std::shared_ptr<const NetworkDescription> networkDescription, const Executor::Ptr executor,
            const Config& config) {
        return _impl->createInferRequest(compiledModel, networkDescription, executor, config);
    }

private:
    std::shared_ptr<IDevice> _impl;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::shared_ptr<void> _so;
};

}  // namespace vpux
