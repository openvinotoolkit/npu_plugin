//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/backend.hpp"

#include "vpux/IMD/device.hpp"
#include "vpux/IMD/parsed_config.hpp"
#include "vpux/vpux_plugin_params.hpp"
#include "vpux_private_config.hpp"
#include "vpux_private_properties.hpp"

#include "device_helpers.hpp"

const std::shared_ptr<vpux::IDevice> vpux::IMD::BackendImpl::getDevice() const {
    InferenceEngine::VPUXConfigParams::VPUXPlatform platform;

    platform = InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720;

    return std::make_shared<IMD::DeviceImpl>(platform);
}

const std::shared_ptr<vpux::IDevice> vpux::IMD::BackendImpl::getDevice(const std::string& name) const {
    const auto platform = utils::getPlatformByDeviceName(name);
    return std::make_shared<IMD::DeviceImpl>(platform);
}

const std::shared_ptr<vpux::IDevice> vpux::IMD::BackendImpl::getDevice(const InferenceEngine::ParamMap& params) const {
    const auto it = params.find(InferenceEngine::VPUX_PARAM_KEY(DEVICE_ID));
    VPUX_THROW_WHEN(it == params.end(), "DEVICE_ID parameter was not provided");
    return getDevice(it->second.as<std::string>());
}

const std::vector<std::string> vpux::IMD::BackendImpl::getDeviceNames() const {
    return {"3700", "3720", "4000"};
}

const std::string vpux::IMD::BackendImpl::getName() const {
    return "IMD";
}

void vpux::IMD::BackendImpl::registerOptions(OptionsDesc& options) const {
    options.add<IMD::MV_TOOLS_PATH>();
    options.add<IMD::LAUNCH_MODE>();
    options.add<IMD::MV_RUN_TIMEOUT>();
}

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& obj, const vpux::Config&) {
    obj = std::make_shared<vpux::IMD::BackendImpl>();
}
