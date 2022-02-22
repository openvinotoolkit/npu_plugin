//
// Copyright 2022 Intel Corporation.
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

#pragma once

#include <openvino/runtime/properties.hpp>
#include <string>
#include <vpux/vpux_plugin_config.hpp>

#include "vpux_private_config.hpp"

namespace ov {
namespace intel_vpux {

/**
 * @brief [Only for VPUX Plugin]
 * Type: Arbitrary string.
 * This option allows to specify device.
 * If specified device is not available then creating infer request will throw an exception.
 */
static constexpr ov::Property<InferenceEngine::VPUXConfigParams::VPUXPlatform> vpux_platform{"VPUX_PLATFORM"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: "RGB", "BGR", default is "BGR"
 * This option allows to specify output format of image after SIPP preprocessing.
 * Does not affect preprocessing running on CPU. If a wrong value specified an exception will be thrown
 */
static constexpr ov::Property<InferenceEngine::ColorFormat> graph_color_format{"VPUX_GRAPH_COLOR_FORMAT"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 4.
 * Number of shaves to be used by SIPP during preprocessing
 */
static constexpr ov::Property<int64_t> preprocessing_shaves{"VPUX_PREPROCESSING_SHAVES"};

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 8.
 * Lines per iteration value to be used by SIPP during preprocessing
 */
static constexpr ov::Property<int64_t> preprocessing_lpi{"VPUX_PREPROCESSING_LPI"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 1.
 * Number of preprocessing pipelines to be used by particular network,
 * these pipelines will work in parallel and make preprocessing
 * for all infer requests of this network
 */
static constexpr ov::Property<int64_t> preprocessing_pipes{"VPUX_PREPROCESSING_PIPES"};

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Media-to-Inference (M2I) module for image pre-processing
 */
static constexpr ov::Property<bool> use_m2i{"VPUX_USE_M2I"};

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "NO"
 * This option allows to use Media-to-Inference (M2I)
 * SHAVE only version module for image pre-processing
 */
static constexpr ov::Property<bool> use_shave_only_m2i{"VPUX_USE_SHAVE_ONLY_M2I"};

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: "YES", "NO", default is "YES"
 * This option allows to use Streaming Image Processing Pipeline (SIPP) for image pre-processing
 */
static constexpr ov::Property<bool> use_sipp{"VPUX_USE_SIPP"};

/**
 * @brief [Only for VPUAL Subplugin]
 * Type: integer, default is 1.
 * Number of executor streams
 */
static constexpr ov::Property<int64_t> executor_streams{"VPUX_EXECUTOR_STREAMS"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is 5 minutes = 60 * 1000 * 5.
 * Time interval during which to wait for backend pull to complete
 */
static constexpr ov::Property<int64_t> inference_timeout{"VPUX_INFERENCE_TIMEOUT"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is MLIR.
 * Type of VPU compiler to be used for compilation of a network
 */
static constexpr ov::Property<InferenceEngine::VPUXConfigParams::CompilerType> compiler_type{"VPUX_COMPILER_TYPE"};

static constexpr ov::Property<std::string> compilation_mode{"VPUX_COMPILATION_MODE"};

/**
 * @brief [Only for VPUX compiler]
 * Type: std::string, default is empty.
 * Config for HW-mode's pipeline
 * Available values: low-precision=true/low-precision=false
 */
static constexpr ov::Property<std::string> compilation_mode_params{"VPUX_COMPILATION_MODE_PARAMS"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: integer, default is None
 * Number of DPU groups
 */
static constexpr ov::Property<int64_t> dpu_groups{"VPUX_DPU_GROUPS"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is NONE
 * NONE - do not print profiling info
 * TEXT, JSON - print detailed profiling info during inference in requested format
 */
static constexpr ov::Property<InferenceEngine::VPUXConfigParams::ProfilingOutputTypeArg> print_profiling{
        "VPUX_PRINT_PROFILING"};

/**
 * @brief [Only for VPUX Plugin]
 * Type: string, default is empty.
 * File that contains profiling output.
 * std::cout is used if this string is empty
 */
static constexpr ov::Property<std::string> profiling_output_file{"VPUX_PROFILING_OUTPUT_FILE"};

}  // namespace intel_vpux
}  // namespace ov
