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

#include "InferGraph.h"
#include "models/precompiled_resnet.h"

//------------------------------------------------------------------------------
class HddlUnite_Graph_Helper {
public:
    using Ptr = std::shared_ptr<HddlUnite_Graph_Helper>;

    HddlUnite_Graph_Helper();
    ~HddlUnite_Graph_Helper();
    explicit HddlUnite_Graph_Helper(const HddlUnite::WorkloadContext& workloadContext);

    HddlUnite::Inference::Graph::Ptr getGraph();

protected:
    HddlUnite::Inference::Graph::Ptr _graphPtr = nullptr;

    const std::string _graphName = PrecompiledResNet_Helper::resnet50.graphName;
    const std::string _graphPath = PrecompiledResNet_Helper::resnet50.graphPath;
};

//------------------------------------------------------------------------------
inline HddlUnite_Graph_Helper::HddlUnite_Graph_Helper() {
    HddlStatusCode statusCode = HddlUnite::Inference::loadGraph(
            _graphPtr, _graphName, _graphPath);
    if (statusCode != HDDL_OK) {
        IE_THROW() << "Failed to load graph";
    }
}

inline HddlUnite_Graph_Helper::~HddlUnite_Graph_Helper() {
    HddlUnite::Inference::unloadGraph(_graphPtr);
}


inline HddlUnite_Graph_Helper::HddlUnite_Graph_Helper(const HddlUnite::WorkloadContext& workloadContext) {
    HddlStatusCode statusCode = HddlUnite::Inference::loadGraph(
            _graphPtr, _graphName, _graphPath, {workloadContext});
    if (statusCode != HDDL_OK) {
        IE_THROW() << "Failed to load graph";
    }
}

inline HddlUnite::Inference::Graph::Ptr HddlUnite_Graph_Helper::getGraph() {
    return _graphPtr;
}
