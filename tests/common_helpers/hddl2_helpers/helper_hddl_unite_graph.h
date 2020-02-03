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

#include "InferGraph.h"
#include "hddl2_helpers/models/precompiled_resnet.h"

//------------------------------------------------------------------------------
//      class HddlUnite_Graph_Helper
//------------------------------------------------------------------------------
class HddlUnite_Graph_Helper {
public:
    using Ptr = std::shared_ptr<HddlUnite_Graph_Helper>;

    HddlUnite_Graph_Helper();
    HddlUnite_Graph_Helper(const HddlUnite::WorkloadContext& workloadContext);

    HddlUnite::Inference::Graph::Ptr getGraph();

protected:
    HddlUnite::Inference::Graph::Ptr _graphPtr = nullptr;

    const std::string _graphName = PrecompiledResNet_Helper::resnet.graphName;
    const std::string _graphPath = PrecompiledResNet_Helper::resnet.graphPath;
};

//------------------------------------------------------------------------------
//      class HddlUnite_Graph_Helper Implementation
//------------------------------------------------------------------------------
inline HddlUnite_Graph_Helper::HddlUnite_Graph_Helper() {
    HddlStatusCode statusCode = HddlUnite::Inference::loadGraph(
            _graphPtr, _graphName, _graphPath);
    if (statusCode != HDDL_OK) {
        THROW_IE_EXCEPTION << "Failed to load graph";
    }
}

inline HddlUnite_Graph_Helper::HddlUnite_Graph_Helper(const HddlUnite::WorkloadContext& workloadContext) {
    HddlStatusCode statusCode = HddlUnite::Inference::loadGraph(
            _graphPtr, _graphName, _graphPath, {workloadContext});
    if (statusCode != HDDL_OK) {
        THROW_IE_EXCEPTION << "Failed to load graph";
    }
}

inline HddlUnite::Inference::Graph::Ptr HddlUnite_Graph_Helper::getGraph() {
    return _graphPtr;
}
