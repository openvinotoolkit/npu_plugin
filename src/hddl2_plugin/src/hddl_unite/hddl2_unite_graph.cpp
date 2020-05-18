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

#include <Inference.h>
#include <WorkloadContext.h>
#include <hddl2_exceptions.h>
#include <hddl_unite/hddl2_unite_graph.h>

#include <string>

using namespace vpu::HDDL2Plugin;

static const HddlUnite::Device::Ptr getUniteDeviceByID(const std::string& deviceID) {
    if (deviceID.empty()) return nullptr;

    std::vector<HddlUnite::Device> cores;
    getAvailableDevices(cores);
    const auto deviceIt = std::find_if(cores.begin(), cores.end(), [&deviceID](const HddlUnite::Device& device) {
        return std::to_string(device.getSwDeviceId()) == deviceID;
    });
    if (deviceIt == cores.end()) {
        return nullptr;
    }
    return std::make_shared<HddlUnite::Device>(*deviceIt);
}

HddlUniteGraph::HddlUniteGraph(const Graph::Ptr& graphPtr, const std::string& deviceID) {
    HddlStatusCode statusCode;

    const std::string graphName = graphPtr->getGraphName();
    const std::string graphData = graphPtr->getGraphBlob();

    std::vector<HddlUnite::Device> devices_to_use = {};

    const HddlUnite::Device::Ptr core = getUniteDeviceByID(deviceID);
    if (core != nullptr) {
        devices_to_use.push_back(*core);
        std::cout << "Graph: " << graphName << " to device id: " << core->getSwDeviceId()
                  << " | Device: " << core->getName() << std::endl;
    } else {
        // TODO Use logger here
        std::cout << "All devices will be used." << std::endl;
    }

    statusCode =
        HddlUnite::Inference::loadGraph(_uniteGraphPtr, graphName, graphData.data(), graphData.size(), devices_to_use);

    if (statusCode != HddlStatusCode::HDDL_OK) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Load graph error: " << statusCode;
    }
    if (_uniteGraphPtr == nullptr) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Graph information is not provided";
    }
}

HddlUniteGraph::HddlUniteGraph(const Graph::Ptr& graphPtr, const HDDL2RemoteContext::Ptr& contextPtr) {
    HddlStatusCode statusCode;
    if (contextPtr == nullptr) {
        THROW_IE_EXCEPTION << "Workload context is null";
    }

    const std::string graphName = graphPtr->getGraphName();
    const std::string graphData = graphPtr->getGraphBlob();

    HddlUnite::WorkloadContext::Ptr workloadContext = contextPtr->getHddlUniteWorkloadContext();

    statusCode = HddlUnite::Inference::loadGraph(
        _uniteGraphPtr, graphName, graphData.data(), graphData.size(), {*workloadContext});

    if (statusCode != HddlStatusCode::HDDL_OK) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Load graph error: " << statusCode;
    }
    if (_uniteGraphPtr == nullptr) {
        THROW_IE_EXCEPTION << HDDLUNITE_ERROR_str << "Graph information is not provided";
    }
}

HddlUniteGraph::~HddlUniteGraph() {
    if (_uniteGraphPtr != nullptr) {
        HddlUnite::Inference::unloadGraph(_uniteGraphPtr);
    }
}

void HddlUniteGraph::InferAsync(const HddlUniteInferData::Ptr& data) {
    if (data == nullptr) {
        THROW_IE_EXCEPTION << "Data for inference is null!";
    }
    if (_uniteGraphPtr == nullptr) {
        THROW_IE_EXCEPTION << "Graph is null!";
    }

    HddlStatusCode inferStatus = HddlUnite::Inference::inferAsync(*_uniteGraphPtr, data->getHddlUniteInferData());

    if (inferStatus != HddlStatusCode::HDDL_OK) {
        THROW_IE_EXCEPTION << "InferAsync FAILED! return code:" << inferStatus;
    }
}
