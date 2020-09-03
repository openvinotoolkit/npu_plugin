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

enum class EngineBackendType : uint8_t {
    VPUAL = 1,
    HDDLUnite = 2,
};

const std::map<std::string, std::shared_ptr<Device>> EngineBackend::createDeviceMap() {
    std::map<std::string, std::shared_ptr<Device>> devices;

    const auto& concreteDevices = _impl->getDevices();
    for (const auto& device : concreteDevices) {
        devices.insert({device.first, std::make_shared<Device>(device.second, _impl)});
    }

    return devices;
}

EngineBackend::EngineBackend(std::string name): _impl(std::move(name)), _devices(std::move(createDeviceMap())) {}

std::shared_ptr<EngineBackend> EngineBackendConfigurator::findBackend(const InferenceEngine::ParamMap& /*params*/) {
    const auto root = InferenceEngine::getIELibraryPath();
#if defined(__arm__) || defined(__aarch64__)
    const auto type = EngineBackendType::VPUAL;
#else
    const auto type = EngineBackendType::HDDLUnite;
#endif
    switch (type) {
    case EngineBackendType::VPUAL: {
        // TODO: fix name if VPUAL works for Windows
        std::string so_path = root + "/libvpual_backend.so";
        return std::shared_ptr<EngineBackend>(new EngineBackend(so_path));
    }
    default:
        return std::shared_ptr<EngineBackend>(new EngineBackend());
    }

    return nullptr;
}

}  // namespace vpux
