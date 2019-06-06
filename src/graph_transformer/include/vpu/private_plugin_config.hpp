//
// Copyright 2017-2018 Intel Corporation.
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

#include <vpu/vpu_plugin_config.hpp>

namespace InferenceEngine {
namespace VPUConfigParams {

//
// Common options
//

// Compilation

DECLARE_VPU_CONFIG_KEY(NUMBER_OF_SHAVES);
DECLARE_VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES);

DECLARE_VPU_CONFIG_KEY(HW_ADAPTIVE_MODE);

DECLARE_VPU_CONFIG_KEY(PERF_REPORT_MODE);
DECLARE_VPU_CONFIG_VALUE(PER_LAYER);
DECLARE_VPU_CONFIG_VALUE(PER_STAGE);

// Optimizations

DECLARE_VPU_CONFIG_KEY(COPY_OPTIMIZATION);
DECLARE_VPU_CONFIG_KEY(HW_INJECT_STAGES);
DECLARE_VPU_CONFIG_KEY(HW_POOL_CONV_MERGE);
DECLARE_VPU_CONFIG_KEY(PACK_DATA_IN_CMX);

// Debug

DECLARE_VPU_CONFIG_KEY(DETECT_NETWORK_BATCH);

DECLARE_VPU_CONFIG_KEY(ALLOW_FP32_MODELS);

DECLARE_VPU_CONFIG_KEY(HW_WHITE_LIST);
DECLARE_VPU_CONFIG_KEY(HW_BLACK_LIST);

DECLARE_VPU_CONFIG_KEY(NONE_LAYERS);

DECLARE_VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS);

//
// Myriad plugin options
//

DECLARE_VPU_MYRIAD_CONFIG_KEY(WATCHDOG);
INFERENCE_ENGINE_DEPRECATED
DECLARE_VPU_CONFIG_KEY(WATCHDOG);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
