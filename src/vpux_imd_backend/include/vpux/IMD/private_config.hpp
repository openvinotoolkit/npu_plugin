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

#pragma once

#include <vpux/vpux_plugin_config.hpp>

#include <string>

namespace InferenceEngine {
namespace VPUXConfigParams {

#define VPUX_IMD_CONFIG_KEY(name) InferenceEngine::VPUXConfigParams::_CONFIG_KEY(VPUX_IMD_##name)
#define VPUX_IMD_CONFIG_VALUE(name) InferenceEngine::VPUXConfigParams::VPUX_IMD_##name

#define DECLARE_VPUX_IMD_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPUX_IMD_##name)
#define DECLARE_VPUX_IMD_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPUX_IMD_##name)

// Path to MV tools
DECLARE_VPUX_IMD_CONFIG_KEY(MV_TOOLS_PATH);

// Launch mode
DECLARE_VPUX_IMD_CONFIG_KEY(LAUNCH_MODE);
DECLARE_VPUX_IMD_CONFIG_VALUE(MOVI_SIM);
DECLARE_VPUX_IMD_CONFIG_KEY(MV_RUN_TIMEOUT);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
