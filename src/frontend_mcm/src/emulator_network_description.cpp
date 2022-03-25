//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "emulator_network_description.hpp"

#include "vpux/al/config/common.hpp"

#include "blob_parser.hpp"
#include "mcm_adapter.hpp"

#include <utility>

using namespace vpux;

namespace {
DataMap inputsDataMapToDataMap(const InferenceEngine::InputsDataMap& inputs) {
    DataMap dataMap = {};
    for (auto&& in : inputs) {
        dataMap.insert({in.first, in.second->getInputData()});
    }

    return dataMap;
}
DataMap outputsDataMapToDataMap(const InferenceEngine::OutputsDataMap& outputs) {
    DataMap dataMap = {};
    for (auto&& out : outputs) {
        dataMap.insert({out.first, out.second});
    }

    return dataMap;
}
}  // namespace

namespace vpu {
namespace MCMAdapter {

EmulatorNetworkDescription::EmulatorNetworkDescription(const std::vector<char>& compiledNetwork, const Config& config,
                                                       const std::string& name)
        : _name{name},
          _logger("EmulatorNetworkDescription", config.get<LOG_LEVEL>()),
          _dataMapPlaceholder{},
          _compiledNetwork{compiledNetwork},
          _quantParams{},
          _ovParameters{},
          _ovResults{} {
    IE_ASSERT(!_compiledNetwork.empty());

    const auto* graphFilePtr = MVCNN::GetGraphFile(compiledNetwork.data());
    IE_ASSERT(graphFilePtr != nullptr);
    const auto graphHeader = graphFilePtr->header();
    IE_ASSERT(graphHeader != nullptr);
    const auto metaInfo = MCMAdapter::deserializeMetaData(*graphHeader, config);
    const InferenceEngine::InputsDataMap& deserializedInputs = metaInfo._inputs;
    const InferenceEngine::OutputsDataMap& deserializedOutputs = metaInfo._outputs;

    if (deserializedInputs.empty()) {
        IE_THROW() << "EmulatorNetworkDescription: meta-data does not contain inputs.";
    }

    if (deserializedOutputs.empty()) {
        IE_THROW() << "EmulatorNetworkDescription: meta-data does not contain outputs.";
    }

    const auto graphInputs = graphHeader->net_input();
    IE_ASSERT(graphInputs != nullptr);
    const auto deviceInputs = MCMAdapter::getNetworkInputs(*graphInputs);
    _deviceInputs = inputsDataMapToDataMap(deviceInputs);

    const auto graphOutputs = graphHeader->net_output();
    IE_ASSERT(graphOutputs != nullptr);
    const auto deviceOutputs = MCMAdapter::getNetworkOutputs(*graphOutputs);
    _deviceOutputs = outputsDataMapToDataMap(deviceOutputs);

    _networkInputs = inputsDataMapToDataMap(deserializedInputs);
    _networkOutputs = outputsDataMapToDataMap(deserializedOutputs);
}

const DataMap& EmulatorNetworkDescription::getInputsInfo() const {
    _logger.info("EmulatorNetworkDescription::getInputsInfo()");
    return _networkInputs;
}

const DataMap& EmulatorNetworkDescription::getOutputsInfo() const {
    _logger.info("EmulatorNetworkDescription::getOutputsInfo()");
    return _networkOutputs;
}

const DataMap& EmulatorNetworkDescription::getDeviceInputsInfo() const {
    _logger.info("EmulatorNetworkDescription::getDeviceInputsInfo()");
    return _deviceInputs;
}

const DataMap& EmulatorNetworkDescription::getDeviceOutputsInfo() const {
    _logger.info("EmulatorNetworkDescription::getDeviceOutputsInfo()");
    return _deviceOutputs;
}

const DataMap& EmulatorNetworkDescription::getDeviceProfilingOutputsInfo() const {
    _logger.info("EmulatorNetworkDescription::getDeviceProfilingsInfo()");
    return _dataMapPlaceholder;
}

const std::vector<OVRawNode>& EmulatorNetworkDescription::getOVParameters() const {
    _logger.info("EmulatorNetworkDescription::getOVParameters()");
    return _ovParameters;
}

const std::vector<OVRawNode>& EmulatorNetworkDescription::getOVResults() const {
    _logger.info("EmulatorNetworkDescription::getOVResults()");
    return _ovResults;
}

const QuantizationParamMap& EmulatorNetworkDescription::getQuantParamsInfo() const {
    _logger.info("EmulatorNetworkDescription::getQuantParamsInfo()");
    return _quantParams;
}

const std::vector<char>& EmulatorNetworkDescription::getCompiledNetwork() const {
    return _compiledNetwork;
}

const void* EmulatorNetworkDescription::getNetworkModel() const {
    return _compiledNetwork.data();
}

std::size_t EmulatorNetworkDescription::getNetworkModelSize() const {
    return _compiledNetwork.size();
}

const std::string& EmulatorNetworkDescription::getName() const {
    return _name;
}

}  // namespace MCMAdapter
}  // namespace vpu
