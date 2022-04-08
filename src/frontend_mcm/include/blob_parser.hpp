//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
