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

#include <vpu/kmb_params.hpp>

#include "vpux_compiler.hpp"
#include "vpux_config.hpp"

namespace vpux {

class IDevice;

class IEngineBackend : public InferenceEngine::details::IRelease {
public:
    virtual const std::map<std::string, std::shared_ptr<IDevice>>& getDevices() const = 0;

    virtual void Release() noexcept override { delete this; }
};

class Allocator : public InferenceEngine::IAllocator {
public:
    // TODO: need update methods to remove Kmb from parameters
    virtual void* wrapRemoteMemoryHandle(
        const KmbRemoteMemoryFD& remoteMemoryFd, const size_t& size, void* memHandle) noexcept = 0;
    virtual void* wrapRemoteMemoryOffset(
        const KmbRemoteMemoryFD& remoteMemoryFd, const size_t& size, const KmbOffsetParam& memOffset) noexcept = 0;

    // FIXME: temporary exposed to allow executor to use vpux::Allocator
    virtual unsigned long getPhysicalAddress(void* handle) noexcept = 0;
};

class Executor;

class IDevice : public InferenceEngine::details::IRelease {
public:
    virtual std::shared_ptr<Allocator> getAllocator() const = 0;

    // TODO: uncomment once we have a concrete executor for the backend
    /* virtual std::shared_ptr<Executor> createExecutor( */
    /* const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) = 0; */

    virtual std::string getName() const = 0;

    virtual void Release() noexcept override { delete this; }
};

class Device final {
    // Device stores instances of classes inherited from IDevice. The instances come from _plg library.
    // Device has to keep pointer to _plg to avoid situations when the shared library unloaded earlier than
    // an instance of IDevice
    std::shared_ptr<IDevice> _actual = nullptr;
    InferenceEngine::details::SharedObjectLoader::Ptr _plg = nullptr;

public:
    Device(const std::shared_ptr<IDevice> device, InferenceEngine::details::SharedObjectLoader::Ptr plg)
        : _actual(device), _plg(plg) {}
    std::shared_ptr<Allocator> getAllocator() const { return _actual->getAllocator(); }

    // TODO: uncomment once we have a concrete executor for the backend
    /* virtual std::shared_ptr<Executor> createExecutor( */
    /* const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) = 0; */

    std::string getName() const { return _actual->getName(); }

    ~Device() { _actual = nullptr; }
};

class EngineBackendConfigurator;

class EngineBackend final {
    friend class EngineBackendConfigurator;

    using IEngineBackendPtr = InferenceEngine::details::SOPointer<IEngineBackend>;
    IEngineBackendPtr _impl = {};
    const std::map<std::string, std::shared_ptr<Device>> _devices = {};

public:
    const std::map<std::string, std::shared_ptr<Device>>& getDevices() const { return _devices; }

private:
    EngineBackend(std::string name);
    EngineBackend() = default;
    const std::map<std::string, std::shared_ptr<Device>> createDeviceMap();
};

using PreprocMap = std::map<std::string, const InferenceEngine::PreProcessInfo>;
class Executor {
public:
    using Ptr = std::shared_ptr<Executor>;

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

class EngineBackendConfigurator {
public:
    static std::shared_ptr<EngineBackend> findBackend(const InferenceEngine::ParamMap& params = {});

private:
    EngineBackendConfigurator();
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
