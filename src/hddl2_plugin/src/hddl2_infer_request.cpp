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

#include "hddl2_infer_request.h"

#include <map>
#include <string>

#include "hddl2_helpers.h"

vpu::HDDL2Plugin::HDDL2InferRequest::HDDL2InferRequest(
    const InferenceEngine::InputsDataMap& networkInputs, const InferenceEngine::OutputsDataMap& networkOutputs)
    : InferRequestInternal(networkInputs, networkOutputs) {
    std::cout << "HDDL2InferRequest costr call" << std::endl;
}

void vpu::HDDL2Plugin::HDDL2InferRequest::InferImpl() { std::cout << "InferImpl call" << std::endl; }

void vpu::HDDL2Plugin::HDDL2InferRequest::GetPerformanceCounts(
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const {
    UNUSED(perfMap);
    std::cout << "GetPerformanceCounts call" << std::endl;
}
