//
// Copyright 2019 Intel Corporation.
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

#include <vpu/vpu_compiler_config.hpp>

//
// KMB plugin options
//

#define VPU_KMB_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_KMB_##name)
#define VPU_KMB_CONFIG_VALUE(name) InferenceEngine::VPUConfigParams::VPU_KMB_##name

#define KMB_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(KMB_##name)
#define KMB_CONFIG_VALUE(name) InferenceEngine::VPUConfigParams::KMB_##name

#define DECLARE_VPU_KMB_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_KMB_##name)
#define DECLARE_VPU_KMB_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_KMB_##name)

#define DECLARE_KMB_CONFIG_KEY(name) DECLARE_CONFIG_KEY(KMB_##name)
#define DECLARE_KMB_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(KMB_##name)

namespace InferenceEngine {
namespace VPUConfigParams {

//
// KMB plugin options
//

/**
 * @brief [Only for kmbPlugin]
 * Type: Arbitrary string.
 * This option allows to specify device.
 * If specified device is not available then creating infer request will throw an exception.
 */
DECLARE_VPU_KMB_CONFIG_KEY(PLATFORM);

/**
 * @brief [Only for kmbPlugin]
 * Type: integer, default is 1.
 */
DECLARE_KMB_CONFIG_KEY(THROUGHPUT_STREAMS);

#ifdef ENABLE_M2I
/**
 * @brief [Only for kmbPlugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Media-to-Inference (M2I) module for image pre-processing
 */
DECLARE_VPU_KMB_CONFIG_KEY(USE_M2I);
#endif

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
