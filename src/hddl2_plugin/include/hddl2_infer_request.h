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

#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vpu {
namespace HDDL2Plugin {

class HDDL2InferRequest : public InferenceEngine::InferRequestInternal {
public:
    HDDL2InferRequest(const InferenceEngine::InputsDataMap& networkInputs,
        const InferenceEngine::OutputsDataMap& networkOutputs, HddlUnite::Inference::Graph::Ptr graph);
    ~HDDL2InferRequest() override;

    void InferImpl() override;
    void GetPerformanceCounts(
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const override;

    void InferSync();
    void Infer() override;

private:
    void GetResult();

    std::vector<HddlUnite::Device> _devices;
    HddlUnite::Inference::Graph::Ptr _graph;
    std::vector<HddlUnite::Inference::AuxBlob::Type> types {HddlUnite::Inference::AuxBlob::Type::TimeTaken};
    HddlUnite::Inference::InferData::Ptr _inferData;

    size_t _inputSize;
    size_t _outputSize;

    HddlUnite::SMM::RemoteMemory::Ptr _inputRemoteMemory;
    HddlUnite::SMM::RemoteMemory::Ptr _outputRemoteMemory;
};

}  //  namespace HDDL2Plugin
}  //  namespace vpu
