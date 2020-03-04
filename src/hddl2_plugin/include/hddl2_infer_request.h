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

#include <HddlUnite.h>
#include <Inference.h>

#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "hddl_unite/hddl2_infer_data.h"
#include "hddl_unite/hddl2_unite_graph.h"

namespace vpu {
namespace HDDL2Plugin {

class HDDL2InferRequest : public InferenceEngine::InferRequestInternal {
public:
    HDDL2InferRequest(const InferenceEngine::InputsDataMap& networkInputs,
        const InferenceEngine::OutputsDataMap& networkOutputs, const HddlUniteGraph::Ptr& loadedGraph,
        const HDDL2RemoteContext::Ptr& context);

    void InferImpl() override;
    void GetPerformanceCounts(
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const override;

    void GetResult();

protected:
    HddlUniteGraph::Ptr _loadedGraphPtr = nullptr;
    HddlUniteInferData::Ptr _inferDataPtr = nullptr;

    // TODO [Workaround] This variable should be inside infer data, but since we are creating it before inference, we
    // need to store it here
    HDDL2RemoteContext::Ptr _context = nullptr;
};

}  //  namespace HDDL2Plugin
}  //  namespace vpu
