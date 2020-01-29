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

#include <hddl2_exceptions.h>
#include <hddl_unite/hddl2_unite_graph.h>

#include <string>

using namespace vpu::HDDL2Plugin;

HddlUniteGraph::HddlUniteGraph(const HDDL2Graph::Ptr& graph, const HDDL2RemoteContext::Ptr& context) {
    HddlStatusCode statusCode;

    const std::string graphName = graph->getGraphName();
    const std::string graphData = graph->getGraphBlob();

    if (context != nullptr) {
        try {
            _contextPtr = std::dynamic_pointer_cast<HDDL2RemoteContext>(context);
        } catch (...) {
            THROW_IE_EXCEPTION << "Invalid remote context pointer";
        }
        auto workloadContext = _contextPtr->getHddlUniteWorkloadContext();
        statusCode = HddlUnite::Inference::loadGraph(
            _graphPtr, graphName, graphData.data(), graphData.size(), {*workloadContext});
    } else {
        statusCode = HddlUnite::Inference::loadGraph(_graphPtr, graphName, graphData.data(), graphData.size());
    }

    if (statusCode != HddlStatusCode::HDDL_OK) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Load graph error: " << statusCode;
    }
    if (_graphPtr == nullptr) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Graph information is not provided";
    }
}

HddlUniteGraph::~HddlUniteGraph() {
    if (_graphPtr != nullptr) {
        HddlUnite::Inference::unloadGraph(_graphPtr);
    }
}
