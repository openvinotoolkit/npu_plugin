//
// Copyright 2019 Intel Corporation.
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

#include <hddl2_executable_network.h>
#include <hddl2_helpers.h>
#include <hddl2_infer_request.h>

#include <memory>

InferenceEngine::InferRequestInternal::Ptr vpu::HDDL2Plugin::ExecutableNetwork::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<HDDL2InferRequest>(networkInputs, networkOutputs);
}

vpu::HDDL2Plugin::ExecutableNetwork::ExecutableNetwork(InferenceEngine::ICNNNetwork& network) {
    std::cout << "ExecutableNetwork with ICNN netw constr call" << std::endl;
    UNUSED(network);
}

vpu::HDDL2Plugin::ExecutableNetwork::ExecutableNetwork() {
    std::cout << "ExecutableNetwork empty constr call" << std::endl;
}
