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

// clang-format off

#include "mcm_config.h"
#include <cpp/ie_cnn_network.h>
#include <ngraph/function.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ie = InferenceEngine;

//
// Usefull environment variables:
//
//   * NGRAPH_ENABLE_VISUALIZE_TRACING=1
//   * NGRAPH_VISUALIZE_TRACING_FORMAT=dot
//   * NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES=1
//   * NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES=1
//

std::vector<char> compileNGraph(
        const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName,
        const ie::InputsDataMap& inputsInfo,
        const ie::OutputsDataMap& outputsInfo,
        const vpu::MCMConfig& config);

// clang-format on
