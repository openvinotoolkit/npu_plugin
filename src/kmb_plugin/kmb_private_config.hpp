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
#include <vpu/kmb_plugin_config.hpp>

namespace InferenceEngine {
namespace VPUConfigParams {

/**
 * @brief [Only for kmbPlugin]
 * Type: "RGB", "BGR", default is "BGR"
 * This option allows to specify output format of image after SIPP preprocessing.
 * Does not affect preprocessing running on CPU. If a wrong value specified an exception will be thrown
 */
DECLARE_VPU_KMB_CONFIG_KEY(SIPP_OUT_COLOR_FORMAT);

/**
 * @brief [Only for kmbPlugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to force conversion of input from NCHW to NHWC ignoring TensorDesc info
 */
DECLARE_VPU_KMB_CONFIG_KEY(FORCE_NCHW_TO_NHWC);

/**
 * @brief [Only for kmbPlugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Streaming Image Processing Pipeline (SIPP) for image pre-processing
 */
DECLARE_VPU_KMB_CONFIG_KEY(USE_SIPP);

/**
 * @brief [Only for kmbPlugin]
 * Type: integer, default is 4.
 * Number of shaves to be used by SIPP during preprocessing
 */
DECLARE_VPU_KMB_CONFIG_KEY(PREPROCESSING_SHAVES);

/**
 * @brief [Only for kmbPlugin]
 * Type: integer, default is 8.
 * Lines per iteration value to be used by SIPP during preprocessing
 */
DECLARE_VPU_KMB_CONFIG_KEY(PREPROCESSING_LPI);

/**
 * @brief [Only for kmbPlugin]
 * Type: "YES/NO", default is "YES".
 */
DECLARE_VPU_KMB_CONFIG_KEY(KMB_EXECUTOR);

/**
 * @brief [Only for kmbPlugin]
 * Type: integer, default is 0.
 * This option allows to set prefetch buffer size in bytes
 */
DECLARE_VPU_KMB_CONFIG_KEY(PREFETCH_BUFFER_SIZE);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
