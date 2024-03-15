//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/backend.hpp"

#include "vpux/IMD/device.hpp"
#include "vpux/IMD/parsed_properties.hpp"
#include "vpux/vpux_plugin_params.hpp"
#include "vpux_private_properties.hpp"

#include "device_helpers.hpp"

namespace vpux {

const std::shared_ptr<IDevice> IMDBackend::getDevice() const {
    InferenceEngine::VPUXConfigParams::VPUXPlatform platform;

    platform = InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720;

    return std::make_shared<IMDDevice>(platform);
}

const std::shared_ptr<IDevice> IMDBackend::getDevice(const std::string& name) const {
    const auto platform = utils::getPlatformByDeviceName(name);
    return std::make_shared<IMDDevice>(platform);
}

const std::shared_ptr<IDevice> IMDBackend::getDevice(const ov::AnyMap& params) const {
    const auto it = params.find(InferenceEngine::VPUX_PARAM_KEY(DEVICE_ID));
    VPUX_THROW_WHEN(it == params.end(), "DEVICE_ID parameter was not provided");
    return getDevice(it->second.as<std::string>());
}

const std::vector<std::string> IMDBackend::getDeviceNames() const {
    return {"3700", "3720"};
}

const std::string IMDBackend::getName() const {
    return "IMD";
}

void IMDBackend::registerOptions(OptionsDesc& options) const {
    options.add<MV_TOOLS_PATH>();
    options.add<LAUNCH_MODE>();
    options.add<MV_RUN_TIMEOUT>();
}

OPENVINO_PLUGIN_API void CreateVPUXEngineBackend(std::shared_ptr<IEngineBackend>& obj, const Config&) {
    obj = std::make_shared<IMDBackend>();
}

}  // namespace vpux
