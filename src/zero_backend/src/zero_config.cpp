//
// Copyright 2021 Intel Corporation.
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

#include "zero_config.h"

#include <cpp_interfaces/exception2status.hpp>
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
