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
#include <vpux_config.hpp>

namespace vpux {

int extractIdFromDeviceName(const std::string& name);
bool isBlobAllocatedByAllocator(const InferenceEngine::Blob::Ptr& blob,
                                const std::shared_ptr<InferenceEngine::IAllocator>& allocator);

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
    virtual const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& paramMap) const;
    /** @brief Provide a list of names of all devices, with which user can work directly */
    virtual const std::vector<std::string> getDeviceNames() const;
    /** @brief Get name of backend */
    virtual const std::string getName() const = 0;
    /** @brief Get a list of supported options */
    virtual std::unordered_set<std::string> getSupportedOptions() const;

protected:
    ~IEngineBackend() = default;
};

class EngineBackendConfigurator;

class EngineBackend final {
public:
    virtual ~EngineBackend() = default;
    virtual const std::shared_ptr<Device> getDevice() const;
    virtual const std::shared_ptr<Device> getDevice(const std::string& specificDeviceName) const;
    virtual const std::shared_ptr<Device> getDevice(const InferenceEngine::ParamMap& paramMap) const;
    virtual const std::vector<std::string> getDeviceNames() const {
        return _impl->getDeviceNames();
    }
    virtual const std::string getName() const {
        return _impl->getName();
    }
    virtual const std::unordered_set<std::string> getSupportedOptions() const {
        return _impl->getSupportedOptions();
    }

private:
    friend class EngineBackendConfigurator;

    using IEngineBackendPtr = InferenceEngine::details::SOPointer<IEngineBackend>;
    IEngineBackendPtr _impl = {};

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
    virtual void* wrapRemoteMemory(const InferenceEngine::ParamMap& paramMap) noexcept;
    // TODO: need update methods to remove Kmb from parameters
    /** @deprecated These functions below should not be used */
    virtual void* wrapRemoteMemoryHandle(const int& remoteMemoryFd, const size_t size, void* memHandle) noexcept = 0;
    virtual void* wrapRemoteMemoryOffset(const int& remoteMemoryFd, const size_t size,
                                         const size_t& memOffset) noexcept = 0;

    // FIXME: temporary exposed to allow executor to use vpux::Allocator
    virtual unsigned long getPhysicalAddress(void* handle) noexcept = 0;
};

//------------------------------------------------------------------------------
class AllocatorWrapper : public Allocator {
private:
    // AllocatorWrapper has to keep pointer to _plg to avoid situations when the shared library unloaded earlier than
    // an instance of Allocator
    std::shared_ptr<Allocator> _actual;
    InferenceEngine::details::SharedObjectLoader::Ptr _plg;

public:
    AllocatorWrapper(const std::shared_ptr<Allocator> actual, InferenceEngine::details::SharedObjectLoader::Ptr plg)
            : _actual(actual), _plg(plg) {
    }

    virtual void* lock(void* handle, InferenceEngine::LockOp op = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        return _actual->lock(handle, op);
    }
    virtual void unlock(void* handle) noexcept override {
        return _actual->unlock(handle);
    }
    virtual void* alloc(size_t size) noexcept override {
        return _actual->alloc(size);
    }
    virtual bool free(void* handle) noexcept override {
        return _actual->free(handle);
    }

    virtual void* wrapRemoteMemory(const InferenceEngine::ParamMap& paramMap) noexcept override {
        return _actual->wrapRemoteMemory(paramMap);
    }
    virtual void* wrapRemoteMemoryHandle(const int& remoteMemoryFd, const size_t size,
                                         void* memHandle) noexcept override {
        return _actual->wrapRemoteMemoryHandle(remoteMemoryFd, size, memHandle);
    }
    virtual void* wrapRemoteMemoryOffset(const int& remoteMemoryFd, const size_t size,
                                         const size_t& memOffset) noexcept override {
        return _actual->wrapRemoteMemoryOffset(remoteMemoryFd, size, memOffset);
    }
    virtual unsigned long getPhysicalAddress(void* handle) noexcept override {
        return _actual->getPhysicalAddress(handle);
    }

    ~AllocatorWrapper() {
        _actual = nullptr;
    };
};

//------------------------------------------------------------------------------
class Executor;

class IDevice : public std::enable_shared_from_this<IDevice> {
public:
    virtual std::shared_ptr<Allocator> getAllocator() const = 0;
    /** @brief Get allocator, which is configured/suitable for provided params
     * @example Each backend may have many allocators, each of which suitable for different RemoteMemory param */
    virtual std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& paramMap) const;

    virtual std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& networkDescription,
                                                     const VPUXConfig& config) = 0;

    virtual std::string getName() const = 0;

protected:
    ~IDevice() = default;
};

class Device final {
private:
    // Device stores instances of classes inherited from IDevice. The instances come from _plg library.
    // Device has to keep pointer to _plg to avoid situations when the shared library unloaded earlier than
    // an instance of IDevice
    std::shared_ptr<IDevice> _actual = nullptr;
    InferenceEngine::details::SharedObjectLoader::Ptr _plg = nullptr;
    std::shared_ptr<AllocatorWrapper> _allocatorWrapper = nullptr;

public:
    using Ptr = std::shared_ptr<Device>;
    using CPtr = std::shared_ptr<const Device>;

    Device(const std::shared_ptr<IDevice> device, InferenceEngine::details::SharedObjectLoader::Ptr plg)
            : _actual(device), _plg(plg) {
        if (_actual->getAllocator()) {
            _allocatorWrapper = std::make_shared<AllocatorWrapper>(_actual->getAllocator(), _plg);
        }
    }

    std::shared_ptr<Allocator> getAllocator() const {
        return _allocatorWrapper;
    }
    std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& paramMap) {
        return std::make_shared<AllocatorWrapper>(_actual->getAllocator(paramMap), _plg);
    }

    virtual std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& networkDescription,
                                                     const VPUXConfig& config) {
        return _actual->createExecutor(networkDescription, config);
    }

    std::string getName() const {
        return _actual->getName();
    }

    ~Device() {
        _actual = nullptr;
    }
};

//------------------------------------------------------------------------------
using PreprocMap = std::map<std::string, const InferenceEngine::PreProcessInfo>;
class Executor {
public:
    using Ptr = std::shared_ptr<Executor>;
    using CPtr = std::shared_ptr<const Executor>;

    virtual void setup(const InferenceEngine::ParamMap& params) = 0;
    virtual Executor::Ptr clone() const {
        THROW_IE_EXCEPTION << "Not implemented";
    }

    virtual void push(const InferenceEngine::BlobMap& inputs) = 0;
    virtual void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) = 0;

    virtual void pull(InferenceEngine::BlobMap& outputs) = 0;

    virtual bool isPreProcessingSupported(const PreprocMap& preProcMap) const = 0;
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
