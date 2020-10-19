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

#include "vpux.hpp"

#include <file_utils.h>

#include <details/ie_exception.hpp>
#include <details/ie_so_pointer.hpp>
#include <memory>

namespace vpux {

inline std::string getLibPostfix() {
#ifdef __unix__
    return ".so";
#else
    return ".dll";
#endif
}

enum class EngineBackendType : uint8_t {
    VPUAL = 1,
    HDDL2 = 2,
};

//------------------------------------------------------------------------------
// TODO Deprecated
const std::map<std::string, std::shared_ptr<Device>> EngineBackend::createDeviceMap() {
    std::map<std::string, std::shared_ptr<Device>> devices;

    if (_impl->getName() == "VPUAL") {
        const auto& concreteDevices = _impl->getDevices();
        for (const auto& device : concreteDevices) {
            devices.insert({device.first, std::make_shared<Device>(device.second, _impl)});
        }

        return devices;
    }
    return {};
}

// TODO _devices lists should not be forced initialized here
EngineBackend::EngineBackend(std::string pathToLib): _impl(pathToLib), _devices(std::move(createDeviceMap())) {}

const std::shared_ptr<Device> EngineBackend::getDevice() const {
    return std::make_shared<Device>(_impl->getDevice(), _impl);
}

const std::shared_ptr<Device> EngineBackend::getDevice(const std::string& specificDeviceName) const {
    return std::make_shared<Device>(_impl->getDevice(specificDeviceName), _impl);
}

const std::shared_ptr<Device> EngineBackend::getDevice(const InferenceEngine::ParamMap& paramMap) const {
    return std::make_shared<Device>(_impl->getDevice(paramMap), _impl);
}

//------------------------------------------------------------------------------
std::shared_ptr<EngineBackend> EngineBackendConfigurator::findBackend(const InferenceEngine::ParamMap& /*params*/) {
    const auto root = InferenceEngine::getIELibraryPath();
#if defined(__arm__) || defined(__aarch64__)
    const auto type = EngineBackendType::VPUAL;
#else
    const auto type = EngineBackendType::HDDL2;
#endif
    switch (type) {
    case EngineBackendType::VPUAL: {
        // TODO: fix name if VPUAL works for Windows
        std::string so_path = root + "/libvpual_backend" + getLibPostfix();
        return std::shared_ptr<EngineBackend>(new EngineBackend(so_path));
    }
    case EngineBackendType::HDDL2: {
        std::string so_path = root + "/libhddl2_backend" + getLibPostfix();
        return std::shared_ptr<EngineBackend>(new EngineBackend(so_path));
    }

    default:
        return std::shared_ptr<EngineBackend>(new EngineBackend());
    }

    return nullptr;
}
const std::shared_ptr<IDevice> IEngineBackend::getDevice() const { THROW_IE_EXCEPTION << "Not implemented"; }
const std::shared_ptr<IDevice> IEngineBackend::getDevice(const std::string&) const {
    THROW_IE_EXCEPTION << "Not implemented";
}
const std::shared_ptr<IDevice> IEngineBackend::getDevice(const InferenceEngine::ParamMap&) const {
    THROW_IE_EXCEPTION << "Not implemented";
}
const std::vector<std::string> IEngineBackend::getDeviceNames() const { THROW_IE_EXCEPTION << "Not implemented"; }
const std::string IEngineBackend::getName() const { THROW_IE_EXCEPTION << "Not implemented"; }
const std::map<std::string, std::shared_ptr<IDevice>>& IEngineBackend::getDevices() const {
    THROW_IE_EXCEPTION << "Not implemented";
}
void* Allocator::wrapRemoteMemory(const InferenceEngine::ParamMap&) noexcept {
    std::cerr << "Not implemented" << std::endl;
    return nullptr;
}
}  // namespace vpux
