//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpual_device.hpp"

#include <memory>

#include <device_helpers.hpp>
#include "vpual_core_nn_executor.hpp"
#include "vpusmm_allocator.hpp"

namespace vpux {

VpualDevice::VpualDevice(const std::string& name, const InferenceEngine::VPUXConfigParams::VPUXPlatform& platform)
        : _name(name), _platform(platform) {
    const auto id = utils::getSliceIdByDeviceName(_name);
    _allocator = std::make_shared<VpusmmAllocator>(id);
}

std::shared_ptr<Executor> VpualDevice::createExecutor(const NetworkDescription::Ptr& networkDescription,
                                                      const Config& config) {
    const auto& vpusmmAllocator = std::dynamic_pointer_cast<VpusmmAllocator>(_allocator);
    if (vpusmmAllocator == nullptr) {
        IE_THROW() << "Incompatible allocator passed into vpual_backend";
    }

    const auto id = utils::getSliceIdByDeviceName(_name);
    const auto& executor =
            std::make_shared<VpualCoreNNExecutor>(networkDescription, vpusmmAllocator, id, _platform, config);

    return executor;
}

std::shared_ptr<Allocator> VpualDevice::getAllocator() const {
    return _allocator;
}
std::shared_ptr<Allocator> VpualDevice::getAllocator(const InferenceEngine::ParamMap&) const {
    // TODO Add validation that input param map can be handled by allocator
    return getAllocator();
}

std::string VpualDevice::getName() const {
    return _name;
}

}  // namespace vpux
