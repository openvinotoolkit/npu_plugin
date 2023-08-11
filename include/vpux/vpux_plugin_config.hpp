//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @deprecated Configuration API v1.0 would be deprecated in 2023.1 release.
 * It was left due to backward compatibility needs.
 * As such usage of this version of API is discouraged.
 * Prefer Configuration API v2.0.
 *
 * @file vpu_plugin_config.hpp
 */

#pragma once

#include <vpux/vpux_compiler_config.hpp>

//
// VPUX plugin options
//

#define VPUX_CONFIG_KEY(name) InferenceEngine::VPUXConfigParams::_CONFIG_KEY(VPUX_##name)
#define VPUX_CONFIG_VALUE(name) InferenceEngine::VPUXConfigParams::name

#define DECLARE_VPUX_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPUX_##name)
#define DECLARE_VPUX_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(name)

namespace InferenceEngine {
namespace VPUXConfigParams {

//
// VPUX plugin options
//

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
