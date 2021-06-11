//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpual_device.hpp"

#include <memory>

#include <device_helpers.hpp>
#include "vpual_core_nn_executor.hpp"
#include "vpusmm_allocator.hpp"

namespace vpux {

VpualDevice::VpualDevice(const std::string& name,
    const InferenceEngine::VPUXConfigParams::VPUXPlatform& platform): _name(name), _platform(platform) {
    const auto id = utils::getSliceIdByDeviceName(name);
    _allocator = std::make_shared<VpusmmAllocator>(id);
}

std::shared_ptr<Executor> VpualDevice::createExecutor(
    const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) {
    const auto& vpusmmAllocator = std::dynamic_pointer_cast<VpusmmAllocator>(_allocator);
    if (vpusmmAllocator == nullptr) {
        IE_THROW() << "Incompatible allocator passed into vpual_backend";
    }
    _config.parseFrom(config);

    const auto id = utils::getSliceIdByDeviceName(_name);
    const auto& executor = std::make_shared<VpualCoreNNExecutor>(networkDescription, vpusmmAllocator, id, _platform, _config);

    return executor;
}

std::shared_ptr<Allocator> VpualDevice::getAllocator() const { return _allocator; }
std::shared_ptr<Allocator> VpualDevice::getAllocator(const InferenceEngine::ParamMap&) const {
    // TODO Add validation that input param map can be handled by allocator
    return getAllocator();
}

std::string VpualDevice::getName() const { return _name; }
}  // namespace vpux
