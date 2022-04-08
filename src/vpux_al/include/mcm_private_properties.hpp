//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

/**
 * @brief A header that defines advanced related properties for VPU MCM compiler.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 */

#pragma once

#include <vpux/vpux_compiler_config.hpp>
#include "vpux/utils/core/logger.hpp"

#include <openvino/runtime/properties.hpp>

namespace ov {
namespace intel_vpux {

/**
 * @brief [Only for MCM compiler]
 * name of xml (IR) file name to serialize prepared for sending to mcmCompiler CNNNetwork
 * Type: Arbitrary string. Empty means "no serialization", default: "";
 */
static constexpr ov::Property<std::string> serialize_cnn_before_compile_file{
        "VPU_COMPILER_SERIALIZE_CNN_BEFORE_COMPILE_FILE"};

static constexpr ov::Property<std::string> reference_mode{"VPU_COMPILER_REFERENCE_MODE"};

static constexpr ov::Property<std::string> allow_u8_input_for_fp16_mode{"VPU_COMPILER_ALLOW_U8_INPUT_FOR_FP16_MODELS"};

static constexpr ov::Property<std::string> scale_shift_fusing{"VPU_COMPILER_SCALESHIFT_FUSING"};

static constexpr ov::Property<std::string> allow_permute_nd{"VPU_COMPILER_ALLOW_PERMUTE_ND"};

static constexpr ov::Property<std::string> num_cluster{"VPU_COMPILER_NUM_CLUSTER"};

static constexpr ov::Property<std::string> optimize_input_precision{"VPU_COMPILER_OPTIMIZE_INPUT_PRECISION"};

/**
 * @brief [Currently supported only for MCM Compiler + Level0 backend]
 * Type: "YES", "NO", default is "NO"
 * This option allows to perform FP32/FP16 to U8 input quantization on VPUX Plugin side via CPU
 */
static constexpr ov::Property<bool> force_plugin_input_quantization{"VPU_COMPILER_FORCE_PLUGIN_INPUT_QUANTIZATION"};

/**
 * @brief [Currently supported only for MCM Compiler]
 * Type: "YES", "NO", default is "NO"
 * This option allows to perform FP16 to FP32 output conversion on VPUX Plugin side via CPU
 */
static constexpr ov::Property<bool> output_fp16_to_fp32_host_conversion{
        "VPU_COMPILER_OUTPUT_FP16_TO_FP32_HOST_CONVERSION"};

/**
 * @brief [Only for MCM compiler]
 * Type: Arbitrary string. Empty means ("."), default: "";
 * path where the mcmCompilator resulting files (blobs, json, dots, etc.) should be placed
 * in folders named "<TARGET_DESCRIPTOR>/<COMPILATION_DESCRIPTOR>"
 */
static constexpr ov::Property<std::string> compilation_results_path{"VPU_COMPILER_COMPILATION_RESULTS_PATH"};

/**
 * @brief [Only for MCM compiler]
 * Type: Arbitrary string. Empty means ("<network name>"), default: "";
 * name of mcmCompilator resulting files (blobs, json, dots, etc.)
 */
static constexpr ov::Property<std::string> compilation_results{"VPU_COMPILER_COMPILATION_RESULTS"};

/**
 * @brief [Only for MCM compiler]
 * Type: "YES/NO", default is "YES".
 */
static constexpr ov::Property<bool> generate_blob{"VPU_COMPILER_GENERATE_BLOB"};
/**
 * @brief [Only for MCM compiler]
 * Type: "YES/NO", default is "YES".
 */
static constexpr ov::Property<bool> parsing_only{"VPU_COMPILER_PARSING_ONLY"};

/**
 * @brief [Only for MCM compiler]
 * Type: "YES/NO", default is "YES".
 */
static constexpr ov::Property<bool> generate_json{"VPU_COMPILER_GENERATE_JSON"};

/**
 * @brief [Only for MCM compiler]
 * Type: "YES/NO", default is "NO".
 */
static constexpr ov::Property<bool> generate_dot{"VPU_COMPILER_GENERATE_DOT"};

/**
 * @brief [Only for MCM compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of layer name:strategy
 * Multiple entries separated by comma. eg, conv1:SplitOverK,conv2:SplitOverH
 * Overrides the GO split strategy to be used for a given layer.
 * Adds to GlobalConfigParams
   "split_strategy":
    [   { "name_filter": "conv1",
          "strategy": "SplitOverK" }
    ]
 */
static constexpr ov::Property<std::string> layer_split_strategies{"VPU_COMPILER_LAYER_SPLIT_STRATEGIES"};

/**
 * @brief [Only for MCM compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of layer name:streamsW:streamsH:streamsC:streamsK:streamsN
 * Multiple entries separated by comma. eg, conv1:1:1:1:1,conv2:1:2:1:2:1
 * Overrides the GO streaming strategy to be used for a given layer.
 * Adds to GlobalConfigParams
   "streaming_strategy":
   [ {
     "name_filter": "conv1",
     "splits":
      [ { "W": 1 }, { "H": 1 }, { "C": 1 }, { "K": 1 }, { "N": 1 } ]
    } ]
 */
static constexpr ov::Property<std::string> layer_stream_strategies{"VPU_COMPILER_LAYER_STREAM_STRATEGIES"};

/**
 * @brief [Only for MCM compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of layer name:input_sparsity:output_sparsity:weights_sparsity
 * Multiple entries separated by comma. eg, conv1:true:true:true,conv2:false:true:false
 * Overrides the GO sparsity strategy to be used for a given layer.
 * Adds to GlobalConfigParams
   "sparsity_strategy":
   [ {
     "name_filter": "conv1",
     "inputActivationSparsity": true,
     "outputActivationSparsity": true,
     "weightsSparsity": true
    } ]
 */
static constexpr ov::Property<std::string> layer_sparsity_strategies{"VPU_COMPILER_LAYER_SPARSITY_STRATEGIES"};

/**
 * @brief [Only for MCM compiler]
 * Type: std::string, default is empty.
 * Semicolon separated list of layer name:location
 * Multiple entries separated by comma. eg, conv1:DDR,conv2:NNCMX
 * Overrides the GO location strategy to be used for a given layer.
 * Adds to GlobalConfigParams
   "tensor_placement_override":
   [ {
     "name_filter": "conv1",
     "mem_location": "DDR"
    } ]
 */
static constexpr ov::Property<std::string> layer_location_strategies{"VPU_COMPILER_LAYER_LOCATION_STRATEGIES"};

/**
 * @brief [Only for MCM compiler]
 * Describe log level for mcmCompiler
 * This option should be used with values: PluginConfigParams::LOG_INFO (default),
 * PluginConfigParams::LOG_ERROR, PluginConfigParams::LOG_WARNING,
 * PluginConfigParams::LOG_NONE, PluginConfigParams::LOG_DEBUG, PluginConfigParams::LOG_TRACE
 */
static constexpr ov::Property<vpux::LogLevel> log_level{"VPUX_LOG_LEVEL"};

}  // namespace intel_vpux
}  // namespace ov
