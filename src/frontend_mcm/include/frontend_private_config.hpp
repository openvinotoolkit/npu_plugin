//
// Copyright 2020 Intel Corporation.
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

/**
 * @brief A header that defines advanced related properties for VPU compiler.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_compiler_config.hpp
 */

#pragma once

#include <vpux/vpux_compiler_config.hpp>

namespace InferenceEngine {
namespace VPUXConfigParams {

/**
 * @brief [Only for vpu compiler]
 * name of xml (IR) file name to serialize prepared for sending to mcmCompiler CNNNetwork
 * Type: Arbitrary string. Empty means "no serialization", default: "";
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(SERIALIZE_CNN_BEFORE_COMPILE_FILE);

DECLARE_VPU_COMPILER_CONFIG_KEY(REFERENCE_MODE);

DECLARE_VPU_COMPILER_CONFIG_KEY(ALLOW_U8_INPUT_FOR_FP16_MODELS);

DECLARE_VPU_COMPILER_CONFIG_KEY(SCALESHIFT_FUSING);

DECLARE_VPU_COMPILER_CONFIG_KEY(ALLOW_PERMUTE_ND);

DECLARE_VPU_COMPILER_CONFIG_KEY(NUM_CLUSTER);

DECLARE_VPU_COMPILER_CONFIG_KEY(OPTIMIZE_INPUT_PRECISION);

/**
 * @brief [Currently supported only for MCM Compiler + Level0 backend]
 * Type: "YES", "NO", default is "NO"
 * This option allows to perform FP32/FP16 to U8 input quantization on VPUX Plugin side via CPU
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(FORCE_PLUGIN_INPUT_QUANTIZATION);

/**
 * @brief [Currently supported only for MCM Compiler]
 * Type: "YES", "NO", default is "YES" for Windows platform, "NO" for others
 * This option allows to perform FP16 to FP32 output conversion on VPUX Plugin side via CPU
 */
DECLARE_VPU_COMPILER_CONFIG_KEY(OUTPUT_FP16_TO_FP32_HOST_CONVERSION);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
