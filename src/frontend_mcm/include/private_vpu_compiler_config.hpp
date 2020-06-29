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

#pragma once

#include <vpu/vpu_compiler_config.hpp>

namespace InferenceEngine {
namespace VPUConfigParams {

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("."), default: "";
 * path where the mcmCompilator resulting files (blobs, json, dots, etc.) should be placed
 * in folders named "<TARGET_DESCRIPTOR>/<COMPILATION_DESCRIPTOR>"
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS_PATH);

/**
 * @brief [Only for vpu compiler]
 * Type: Arbitrary string. Empty means ("<network name>"), default: "";
 * name of mcmCompilator resulting files (blobs, json, dots, etc.)
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB);
/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(PARSING_ONLY);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(GENERATE_JSON);

/**
 * @brief [Only for vpu compiler]
 * Type: "YES/NO", default is "NO".
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(GENERATE_DOT);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
