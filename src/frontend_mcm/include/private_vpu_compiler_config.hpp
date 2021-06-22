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

#pragma once

#include <vpux/vpux_compiler_config.hpp>

namespace InferenceEngine {
namespace VPUXConfigParams {

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

/**
 * @brief [Only for vpu compiler]
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
DECLARE_VPU_COMPILER_CONFIG_KEY(LAYER_SPLIT_STRATEGIES);

/**
 * @brief [Only for vpu compiler]
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
DECLARE_VPU_COMPILER_CONFIG_KEY(LAYER_STREAM_STRATEGIES);

/**
 * @brief [Only for vpu compiler]
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
DECLARE_VPU_COMPILER_CONFIG_KEY(LAYER_SPARSITY_STRATEGIES);

/**
 * @brief [Only for vpu compiler]
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
DECLARE_VPU_COMPILER_CONFIG_KEY(LAYER_LOCATION_STRATEGIES);

}  // namespace VPUXConfigParams
}  // namespace InferenceEngine
