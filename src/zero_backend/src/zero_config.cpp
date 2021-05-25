//
// Copyright 2021 Intel Corporation.
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

#include "zero_config.h"

#include <map>
#include <string>
#include <unordered_set>
#include <vpux/vpux_plugin_config.hpp>

#include "zero_private_config.h"

namespace vpux {

ZeroConfig::ZeroConfig() {
    _runTimeOptions = merge(vpux::VPUXConfig::getRunTimeOptions(), {VPUX_ZERO_CONFIG_KEY(ZE_SYNC_TYPE)});
}

void ZeroConfig::parse(const std::map<std::string, std::string>& config) {
    vpux::VPUXConfig::parse(config);

    static const std::unordered_map<std::string, InferenceEngine::VPUXConfigParams::ze_syncType> ze_syncType = {
            {VPUX_ZERO_CONFIG_VALUE(ZE_FENCE), InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE},
            {VPUX_ZERO_CONFIG_VALUE(ZE_EVENT), InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT}};
    setOption(_ze_syncType, ze_syncType, config, VPUX_ZERO_CONFIG_KEY(ZE_SYNC_TYPE));
}
}  // namespace vpux
