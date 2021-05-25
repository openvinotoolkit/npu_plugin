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

// System
#include <string>
// Plugin
#include "hddl2_exceptions.h"
#include "hddl_unite/hddl2_unite_graph.h"
#include "vpux_metrics.h"
// Low level
#include <Inference.h>
#include <WorkloadContext.h>

namespace vpux {
namespace hddl2 {
namespace IE = InferenceEngine;

using namespace vpu;

static const HddlUnite::Device::Ptr getUniteDeviceByID(const std::string& deviceID) {
    if (deviceID.empty())
        return nullptr;

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

static const std::unordered_map<std::string, std::string> getUniteConfigByPluginConfig(const vpux::VPUXConfig& config) {
    std::unordered_map<std::string, std::string> hddlUniteConfig = {};
    const auto csram_size = config.CSRAMSize();
    hddlUniteConfig.insert(std::make_pair("CSRAM_SIZE", std::to_string(csram_size)));
    return hddlUniteConfig;
}

HddlUniteGraph::HddlUniteGraph(const vpux::NetworkDescription::CPtr& network, const std::string& deviceID,
                               const vpux::VPUXConfig& config)
        : _logger(std::make_shared<Logger>("Graph", config.logLevel(), consoleOutput())) {
    if (!network) {
        throw std::invalid_argument("Network pointer is null!");
    }
    HddlStatusCode statusCode;

    const std::string graphName = network->getName();
    const std::vector<char> graphData = network->getCompiledNetwork();

    std::vector<HddlUnite::Device> devices_to_use = {};

    const HddlUnite::Device::Ptr core = getUniteDeviceByID(deviceID);
    if (core != nullptr) {
        devices_to_use.push_back(*core);
        _logger->info("Graph: %s to device id: %d | Device: %s ", graphName, core->getSwDeviceId(), core->getName());
    } else {
        _logger->info("All devices will be used.");
    }

    // TODO we need to get number of NN shaves and threads via config, not as parameters
    // [Track number: S#39350]
    const auto nnThreadNum = config.throughputStreams();
    const auto nnShaveNum = config.numberOfNnCoreShaves();
    const auto& hddlUniteConfig = getUniteConfigByPluginConfig(config);
    statusCode = HddlUnite::Inference::loadGraph(_uniteGraphPtr, graphName, graphData.data(), graphData.size(),
                                                 devices_to_use, nnThreadNum, nnShaveNum, hddlUniteConfig);

    // FIXME This error handling part should be refactored according to new api
    if (statusCode == HddlStatusCode::HDDL_CONNECT_ERROR) {
        IE_THROW(NetworkNotLoaded);
    } else if (statusCode != HddlStatusCode::HDDL_OK) {
        IE_THROW() << HDDLUNITE_ERROR_str << "Load graph error: " << statusCode;
    }

    if (_uniteGraphPtr == nullptr) {
        IE_THROW() << HDDLUNITE_ERROR_str << "Graph information is not provided";
    }
}

HddlUniteGraph::HddlUniteGraph(const vpux::NetworkDescription::CPtr& network,
                               const HddlUnite::WorkloadContext::Ptr& workloadContext, const vpux::VPUXConfig& config)
        : _logger(std::make_shared<Logger>("Graph", config.logLevel(), consoleOutput())) {
    HddlStatusCode statusCode;
    if (workloadContext == nullptr) {
        IE_THROW() << "Workload context is null";
    }

    const std::string graphName = network->getName();
    const std::vector<char> graphData = network->getCompiledNetwork();

    // TODO we need to get number of NN shaves and threads via config, not as parameters
    // [Track number: S#39350]
    const auto nnThreadNum = config.throughputStreams();
    const auto nnShaveNum = config.numberOfNnCoreShaves();
    const auto& hddlUniteConfig = getUniteConfigByPluginConfig(config);
    statusCode = HddlUnite::Inference::loadGraph(_uniteGraphPtr, graphName, graphData.data(), graphData.size(),
                                                 {*workloadContext}, nnThreadNum, nnShaveNum, hddlUniteConfig);

    if (statusCode != HddlStatusCode::HDDL_OK) {
        IE_THROW() << HDDLUNITE_ERROR_str << "Load graph error: " << statusCode;
    }
    if (_uniteGraphPtr == nullptr) {
        IE_THROW() << HDDLUNITE_ERROR_str << "Graph information is not provided";
    }
}

HddlUniteGraph::~HddlUniteGraph() {
    if (_uniteGraphPtr != nullptr) {
        HddlUnite::Inference::unloadGraph(_uniteGraphPtr);
    }
}

void HddlUniteGraph::InferAsync(const InferDataAdapter::Ptr& data) const {
    if (data == nullptr) {
        IE_THROW() << "Data for inference is null!";
    }
    if (_uniteGraphPtr == nullptr) {
        IE_THROW() << "Graph is null!";
    }

    HddlStatusCode inferStatus = HddlUnite::Inference::inferAsync(*_uniteGraphPtr, data->getHDDLUniteInferData());

    if (inferStatus != HddlStatusCode::HDDL_OK) {
        IE_THROW() << "InferAsync FAILED! return code:" << inferStatus;
    }
}

}  // namespace hddl2
}  // namespace vpux
