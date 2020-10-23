//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <vector>
// clang-format off
#include <cstdint>
// clang-format on
#include <memory>
#include <map>
#include <set>

#include <details/ie_so_pointer.hpp>
#include <ie_blob.h>
#include <ie_common.h>
#include <ie_remote_context.hpp>
#include <ie_icnn_network.hpp>

#include "vpux_compiler.hpp"
#include "vpux_config.hpp"

namespace vpux {

//------------------------------------------------------------------------------
class IDevice;
class Device;

class IEngineBackend : public InferenceEngine::details::IRelease {
public:
    // TODO Remove this method and default implementation after merging all to one plugin.
    /** @deprecated Will be replaced with methods bellow */
    virtual const std::map<std::string, std::shared_ptr<IDevice>>& getDevices() const;

    /** @brief Get device, which can be used for inference. Backend responsible for selection. */
    virtual const std::shared_ptr<IDevice> getDevice() const;
    /** @brief Search for a specific device by name */
    virtual const std::shared_ptr<IDevice> getDevice(const std::string& /*specificDeviceName*/) const;
    /** @brief Get device, which is configured/suitable for provided params */
    virtual const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& /*paramMap*/) const;
    /** @brief Provide a list of names of all devices, with which user can work directly */
    virtual const std::vector<std::string> getDeviceNames() const;
    /** @brief Get name of backend */
    virtual const std::string getName() const = 0;

    void Release() noexcept override { delete this; }
};

class EngineBackendConfigurator;

class EngineBackend final {
public:
    /** @deprecated Will be replaced with methods below */
    const std::map<std::string, std::shared_ptr<Device>>& getDevices() const { return _devices; }
    virtual const std::shared_ptr<Device> getDevice() const;
    virtual const std::shared_ptr<Device> getDevice(const std::string& specificDeviceName) const;
    virtual const std::shared_ptr<Device> getDevice(const InferenceEngine::ParamMap& paramMap) const;
    virtual const std::vector<std::string> getDeviceNames() const { return _impl->getDeviceNames(); }
    virtual const std::string getName() const { return _impl->getName(); }

private:
    friend class EngineBackendConfigurator;

    using IEngineBackendPtr = InferenceEngine::details::SOPointer<IEngineBackend>;
    IEngineBackendPtr _impl = {};
    /** @deprecated Force storing will be removed after plugins merging */
    const std::map<std::string, std::shared_ptr<Device>> _devices = {};

private:
    EngineBackend(std::string pathToLib);
    EngineBackend() = default;
    const std::map<std::string, std::shared_ptr<Device>> createDeviceMap();
};

class EngineBackendConfigurator {
public:
    static std::shared_ptr<EngineBackend> findBackend(const InferenceEngine::ParamMap& params = {});

private:
    EngineBackendConfigurator();
};

//------------------------------------------------------------------------------
class Allocator : public InferenceEngine::IAllocator {
public:
    using Ptr = std::shared_ptr<Allocator>;
    using CPtr = std::shared_ptr<const Allocator>;

    /** @brief Wrap remote memory. Backend should get all required data from paramMap */
    virtual void* wrapRemoteMemory(const InferenceEngine::ParamMap& /*paramMap*/) noexcept;
    // TODO: need update methods to remove Kmb from parameters
    /** @deprecated These functions below should not be used */
    virtual void* wrapRemoteMemoryHandle(const int& remoteMemoryFd, const size_t size, void* memHandle) noexcept = 0;
    virtual void* wrapRemoteMemoryOffset(
        const int& remoteMemoryFd, const size_t size, const size_t& memOffset) noexcept = 0;

    // FIXME: temporary exposed to allow executor to use vpux::Allocator
    virtual unsigned long getPhysicalAddress(void* handle) noexcept = 0;
};

//------------------------------------------------------------------------------
class Executor;

class IDevice : public InferenceEngine::details::IRelease {
public:
    virtual std::shared_ptr<Allocator> getAllocator() const = 0;

    virtual std::shared_ptr<Executor> createExecutor(
        const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) = 0;

    virtual std::string getName() const = 0;

    void Release() noexcept override { delete this; }
};

class Device final {
private:
    // Device stores instances of classes inherited from IDevice. The instances come from _plg library.
    // Device has to keep pointer to _plg to avoid situations when the shared library unloaded earlier than
    // an instance of IDevice
    std::shared_ptr<IDevice> _actual = nullptr;
    InferenceEngine::details::SharedObjectLoader::Ptr _plg = nullptr;

public:
    using Ptr = std::shared_ptr<Device>;
    using CPtr = std::shared_ptr<const Device>;

    Device(const std::shared_ptr<IDevice> device, InferenceEngine::details::SharedObjectLoader::Ptr plg)
        : _actual(device), _plg(plg) {}
    std::shared_ptr<Allocator> getAllocator() const { return _actual->getAllocator(); }

    virtual std::shared_ptr<Executor> createExecutor(
        const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) {
        return _actual->createExecutor(networkDescription, config);
    }

    std::string getName() const { return _actual->getName(); }

    ~Device() { _actual = nullptr; }
};

//------------------------------------------------------------------------------
using PreprocMap = std::map<std::string, const InferenceEngine::PreProcessInfo>;
class Executor {
public:
    using Ptr = std::shared_ptr<Executor>;
    using CPtr = std::shared_ptr<const Executor>;

    virtual void setup(const InferenceEngine::ParamMap& params) = 0;
    virtual Executor::Ptr clone() const { THROW_IE_EXCEPTION << "Not implemented"; }

    virtual void push(const InferenceEngine::BlobMap& inputs) = 0;
    virtual void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) = 0;

    virtual void pull(InferenceEngine::BlobMap& outputs) = 0;

    virtual bool isPreProcessingSupported(const InferenceEngine::PreProcessInfo& preProcessInfo) const = 0;
    virtual std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() = 0;
    virtual InferenceEngine::Parameter getParameter(const std::string& paramName) const = 0;

    virtual ~Executor() = default;
};

}  // namespace vpux

namespace InferenceEngine {
namespace details {
template <>
class SOCreatorTrait<vpux::IEngineBackend> {
public:
    static constexpr auto name = "CreateVPUXEngineBackend";
};
}  // namespace details
}  // namespace InferenceEngine
