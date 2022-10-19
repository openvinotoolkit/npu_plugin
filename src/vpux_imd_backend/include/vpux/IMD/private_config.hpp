//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
DECLARE_VPUX_IMD_CONFIG_VALUE(MOVI_DEBUG);
DECLARE_VPUX_IMD_CONFIG_KEY(MV_RUN_TIMEOUT);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
