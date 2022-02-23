//
// Copyright Intel Corporation.
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

#include "vpux/IMD/device.hpp"
#include "vpux/IMD/executor.hpp"
#include "vpux/IMD/platform_helpers.hpp"

#include "vpux/al/config/common.hpp"

using namespace vpux;
using namespace ov::intel_vpux;
using namespace InferenceEngine::VPUXConfigParams;

vpux::IMD::DeviceImpl::DeviceImpl(VPUXPlatform platform): _platform(platform) {
    VPUX_THROW_UNLESS(platformSupported(platform), "Unsupported VPUX platform '{0}'", platform);
}

std::shared_ptr<Allocator> vpux::IMD::DeviceImpl::getAllocator() const {
    return nullptr;
}

std::shared_ptr<Allocator> vpux::IMD::DeviceImpl::getAllocator(const InferenceEngine::ParamMap&) const {
    return nullptr;
}

std::shared_ptr<Executor> vpux::IMD::DeviceImpl::createExecutor(const NetworkDescription::Ptr& network,
                                                                const Config& config) {
    return std::make_shared<IMD::ExecutorImpl>(_platform, network, config);
}

std::string vpux::IMD::DeviceImpl::getName() const {
    auto platformName = stringifyEnum(_platform);
    VPUX_THROW_UNLESS(platformName.consume_front("VPU"), "Unsupported VPUX platform '{0}'", _platform);
    return platformName.str();
}
