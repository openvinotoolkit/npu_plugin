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

#include <string>
#include <vpux/vpux_plugin_config.hpp>

namespace InferenceEngine {
namespace VPUXConfigParams {

/**
 * @brief [Only for VPUX Plugin]
 * Type: "RGB", "BGR", default is "BGR"
 * This option allows to specify output format of image after SIPP preprocessing.
 * Does not affect preprocessing running on CPU. If a wrong value specified an exception will be thrown
 */
DECLARE_VPUX_CONFIG_KEY(GRAPH_COLOR_FORMAT);
DECLARE_VPUX_CONFIG_VALUE(BGR);
DECLARE_VPUX_CONFIG_VALUE(RGB);

/**
 * @brief [Only for VPUX Plugin]
 * Type: Arbitrary string. Default is "0".
 * This option allows to specify CSRAM size.
 * When the size is 0, GetPrefetchBufferSize is called to determine the required amount of CSRAM.
 */
DECLARE_VPUX_CONFIG_KEY(CSRAM_SIZE);

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 4.
 * Number of shaves to be used by SIPP during preprocessing
 */
DECLARE_VPUX_CONFIG_KEY(PREPROCESSING_SHAVES);

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 8.
 * Lines per iteration value to be used by SIPP during preprocessing
 */
DECLARE_VPUX_CONFIG_KEY(PREPROCESSING_LPI);

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Media-to-Inference (M2I) module for image pre-processing
 */
DECLARE_VPUX_CONFIG_KEY(USE_M2I);

/**
 * @deprecated Use VPUX_USE_M2I instead
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Media-to-Inference (M2I) module for image pre-processing
 */
DECLARE_VPU_KMB_CONFIG_KEY(USE_M2I);

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "YES"
 * This option allows to use Streaming Image Processing Pipeline (SIPP) for image pre-processing
 */
DECLARE_VPUX_CONFIG_KEY(USE_SIPP);

/**
 * @deprecated Use VPUX_USE_SIPP instead
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "YES"
 * This option allows to use Streaming Image Processing Pipeline (SIPP) for image pre-processing
 */
DECLARE_VPU_KMB_CONFIG_KEY(USE_SIPP);

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 1.
 * Number of executor streams
 */
DECLARE_VPUX_CONFIG_KEY(EXECUTOR_STREAMS);

/**
 * @deprecated Use VPUX_EXECUTOR_STREAMS instead
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 1.
 * Number of executor streams
 */
DECLARE_VPU_KMB_CONFIG_KEY(EXECUTOR_STREAMS);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
