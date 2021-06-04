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

#include <ie_data.h>

#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>

#include <schema/graphfile/graphfile_generated.h>

namespace vpu {
namespace MCMAdapter {

using graphTensors = flatbuffers::Vector<flatbuffers::Offset<MVCNN::TensorReference>>;

/**
 * @brief Get IE network inputs from graph blob
 * @param header The inputs of the graph blob struct
 * @return Network inputs in IE format
 */
InferenceEngine::InputsDataMap getNetworkInputs(const graphTensors& inputs);

/**
 * @brief Get IE network outputs from graph blob
 * @param header The outputs of the graph blob struct
 * @return Network outputs in IE format
 */
InferenceEngine::OutputsDataMap getNetworkOutputs(const graphTensors& outputs);
}  // namespace MCMAdapter
}  // namespace vpu
