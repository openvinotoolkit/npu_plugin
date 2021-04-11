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
