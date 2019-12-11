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

#pragma once

#include <hddlunite/Inference.h>

#include <memory>
#include <string>
#include <vector>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"

namespace vpu {
namespace HDDL2Plugin {

class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<ExecutableNetwork>;

    explicit ExecutableNetwork(InferenceEngine::ICNNNetwork& network);
    explicit ExecutableNetwork(const std::string& blobFilename);

    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(
        InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) override;

private:
    std::vector<char> _graphBlob;

    std::vector<HddlUnite::Device> _devices;
    HddlUnite::Inference::Graph::Ptr _graph;
};

}  //  namespace HDDL2Plugin
}  //  namespace vpu
