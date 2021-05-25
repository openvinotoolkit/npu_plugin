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

namespace vpu {
namespace MCMAdapter {
void getNetworkInputs(const void* data, InferenceEngine::InputsDataMap& networkInputs);
void getNetworkOutputs(const void* data, InferenceEngine::OutputsDataMap& networkOutputs);
}  // namespace MCMAdapter
}  // namespace vpu
