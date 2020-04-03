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
 * @brief A header that defines advanced related properties for VPU compiler.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_compiler_config.hpp
 */

#pragma once

namespace InferenceEngine {
namespace VPUConfigParams {

/**
 * @brief [Only for vpu compiler]
 * name of xml (IR) file name to serialize prepared for sending to mcmCompiler CNNNetwork
 * Type: Arbitrary string. Empty means "no serialization", default: "";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(SERIALIZE_CNN_BEFORE_COMPILE_FILE);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "NO".
 * Add NC layout to the list of supported output layouts
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(ALLOW_NC_OUTPUT);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "NO".
 * Add FP32 precision to the list of supported output precisions
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(ALLOW_FP32_OUTPUT);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
