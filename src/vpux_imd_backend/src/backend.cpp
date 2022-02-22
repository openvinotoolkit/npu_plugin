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

#include "vpux/IMD/backend.hpp"

#include "vpux/IMD/device.hpp"
#include "vpux/IMD/parsed_config.hpp"
#include "vpux/vpux_plugin_params.hpp"
#include "vpux_private_config.hpp"
#include "vpux_private_properties.hpp"

#include "device_helpers.hpp"

using namespace vpux;
using namespace ov::intel_vpux;
using namespace InferenceEngine::VPUXConfigParams;

const std::shared_ptr<IDevice> vpux::IMD::BackendImpl::getDevice() const {
    return std::make_shared<IMD::DeviceImpl>(VPUXPlatform::VPU3720);
}

const std::shared_ptr<IDevice> vpux::IMD::BackendImpl::getDevice(const std::string& name) const {
    const auto platform = utils::getPlatformByDeviceName(name);
    return std::make_shared<IMD::DeviceImpl>(platform);
}

const std::shared_ptr<IDevice> vpux::IMD::BackendImpl::getDevice(const InferenceEngine::ParamMap& params) const {
    const auto it = params.find(InferenceEngine::VPUX_PARAM_KEY(DEVICE_ID));
    VPUX_THROW_WHEN(it == params.end(), "DEVICE_ID parameter was not provided");
    return getDevice(it->second.as<std::string>());
}

const std::vector<std::string> vpux::IMD::BackendImpl::getDeviceNames() const {
    return {"3720", "3400", "3700"};
}

const std::string vpux::IMD::BackendImpl::getName() const {
    return "IMD";
}

void vpux::IMD::BackendImpl::registerOptions(OptionsDesc& options) const {
    options.add<IMD::MV_TOOLS_PATH>();
    options.add<IMD::LAUNCH_MODE>();
    options.add<IMD::MV_RUN_TIMEOUT>();
}

INFERENCE_PLUGIN_API(void) CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& obj) {
    obj = std::make_shared<IMD::BackendImpl>();
}
