//
// Copyright 2020 Intel Corporation.
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


/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
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

// TODO Only to support old config options. Must be removed in future

#define VPU_KMB_CONFIG_KEY(name) InferenceEngine::VPUXConfigParams::_CONFIG_KEY(VPU_KMB_##name)
#define VPU_KMB_CONFIG_VALUE(name) InferenceEngine::VPUXConfigParams::VPU_KMB_##name

#define KMB_CONFIG_KEY(name) InferenceEngine::VPUXConfigParams::_CONFIG_KEY(KMB_##name)
#define KMB_CONFIG_VALUE(name) InferenceEngine::VPUXConfigParams::KMB_##name

#define DECLARE_VPU_KMB_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_KMB_##name)
#define DECLARE_VPU_KMB_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_KMB_##name)

#define DECLARE_KMB_CONFIG_KEY(name) DECLARE_CONFIG_KEY(KMB_##name)
#define DECLARE_KMB_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(KMB_##name)

namespace InferenceEngine {
namespace VPUXConfigParams {

//
// VPUX plugin options
//

/**
 * @enum VPUXPlatform
 * @brief VPUX device
 */
enum class VPUXPlatform: int {
    AUTO            = 0,    // auto detection
    MA2490          = 1,    // Keem bay A0
    MA2490_B0       = 2,    // Keem bay B0
    MA3100          = 3,    // Thunder bay harbor A0
    MA3720          = 4,    // Meteor lake
};

/**
 * @brief [Only for VPUX Plugin]
 * Type: Arbitrary string.
 * This option allows to specify device.
 * If specified device is not available then creating infer request will throw an exception.
 */
DECLARE_VPUX_CONFIG_KEY(PLATFORM);
DECLARE_VPUX_CONFIG_VALUE(AUTO);
DECLARE_VPUX_CONFIG_VALUE(MA2490);
DECLARE_VPUX_CONFIG_VALUE(MA2490_B0);
DECLARE_VPUX_CONFIG_VALUE(MA3100);
DECLARE_VPUX_CONFIG_VALUE(MA3720);

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 2.
 */
DECLARE_VPUX_CONFIG_KEY(THROUGHPUT_STREAMS);

/**
 * @deprecated Use VPUX_THROUGHPUT_STREAMS instead
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 2.
 */
DECLARE_KMB_CONFIG_KEY(THROUGHPUT_STREAMS);

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 0. SetNumUpaShaves is not called in that case.
 * Number of shaves to be used by NNCore plug-in during inference
 */
DECLARE_VPUX_CONFIG_KEY(INFERENCE_SHAVES);

/**
 * Type: Arbitrary string. Default is "-1".
 * This option allows to specify CSRAM size in bytes
 * When the size is -1, low-level SW is responsible for determining the required amount of CSRAM
 * When the size is 0, CSRAM isn't used
 */
DECLARE_VPUX_CONFIG_KEY(CSRAM_SIZE);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
