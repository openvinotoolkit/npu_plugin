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

#pragma once

#include <string>
#include <vpux/vpux_plugin_config.hpp>

namespace InferenceEngine {
namespace VPUXConfigParams {

#define VPUX_ZERO_CONFIG_KEY(name) InferenceEngine::VPUXConfigParams::_CONFIG_KEY(VPUX_ZERO_##name)
#define VPUX_ZERO_CONFIG_VALUE(name) InferenceEngine::VPUXConfigParams::VPUX_ZERO_##name

#define DECLARE_VPUX_ZERO_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPUX_ZERO_##name)
#define DECLARE_VPUX_ZERO_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPUX_ZERO_##name)

/**
 * @enum ze_syncType
 * @brief inference synchronize type (zero API)
 */
enum class ze_syncType : uint32_t {
	ZE_FENCE = 0,
	ZE_EVENT = 1 };

DECLARE_VPUX_ZERO_CONFIG_KEY(ZE_SYNC_TYPE);
DECLARE_VPUX_ZERO_CONFIG_VALUE(ZE_FENCE);
DECLARE_VPUX_ZERO_CONFIG_VALUE(ZE_EVENT);
}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
