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

// clang-format off
// Can get compile error, if the order of the headers will be changed.

#include <algorithm>
#include <utility>

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <kmb_executable_network.h>
#include <net_pass.h>

#include <cnn_network_ngraph_impl.hpp>
#include "ngraph_mcm_frontend/frontend.hpp"

// clang-format on

using namespace InferenceEngine;

namespace vpu {
namespace KmbPlugin {

void ExecutableNetwork::ConfigureExecutor(const std::string& networkName) {
    if (_config.exclusiveAsyncRequests()) {
        ExecutorManager* executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor("KMB");
    }
    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
}

void ExecutableNetwork::LoadBlob() {
    IE_PROFILING_AUTO_SCOPE(LoadBlob);
    std::pair<InferenceEngine::InputsDataMap, InferenceEngine::OutputsDataMap> portsInfo =
        MCMAdapter::deserializeMetaData(_graphBlob, _config);
    const InferenceEngine::InputsDataMap& deserializedInputs = portsInfo.first;
    const InferenceEngine::OutputsDataMap& deserializedOutputs = portsInfo.second;
    const bool newFormat = (deserializedInputs.size() > 0) && (deserializedOutputs.size() > 0);
    _executor->allocateGraph(_graphBlob, deserializedInputs, deserializedOutputs, newFormat);
    _runtimeInputs = _executor->getRuntimeInputs();
    _runtimeOutputs = _executor->getRuntimeOutputs();
    if (newFormat) {
        _networkInputs = deserializedInputs;
        _networkOutputs = deserializedOutputs;
    } else {
        _networkInputs = _runtimeInputs;
        _networkOutputs = _runtimeOutputs;
    }
}

ExecutableNetwork::ExecutableNetwork(ICNNNetwork& network, const KmbConfig& config): _config(config) {
    IE_PROFILING_AUTO_SCOPE(ExecutableNetwork);

    _netName = network.getName();
    _supportedMetrics = {METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)};

    _logger = std::make_shared<Logger>("ExecutableNetwork", _config.logLevel(), consoleOutput());
    _executor = std::make_shared<KmbExecutor>(_config);

    if (_config.useNGraphParser()) {
        if (const auto cnnNGraphNet = dynamic_cast<ie::details::CNNNetworkNGraphImpl*>(&network)) {
            if (const auto func = cnnNGraphNet->getFunction()) {
                _logger->info("Using NGraph parser");
                InputsDataMap inputsInfo;
                network.getInputsInfo(inputsInfo);

                OutputsDataMap outputsInfo;
                network.getOutputsInfo(outputsInfo);

                _graphBlob = compileNGraph(func, network.getName(), inputsInfo, outputsInfo, _config);
            } else {
                _logger->warning("Failed to read NGraph func");
            }
        } else {
            _logger->warning("Failed to read NGraph network");
        }
    } else {
        _logger->info("NGraph parser disabled");
    }

    if (_graphBlob.empty()) {
        _logger->info("Using CNNNetwork parser");
#ifdef ENABLE_MCM_COMPILER
        // HACK: convert nGraph to old CNNNetwork to fix LP transformations

        std::shared_ptr<ICNNNetwork> convertedNetwork;
        auto actualNetwork = &network;

        if (auto networkNGraph = dynamic_cast<ie::details::CNNNetworkNGraphImpl*>(&network)) {
            convertedNetwork = networkNGraph->getCNNNetwork();
            actualNetwork = convertedNetwork.get();
        }

        MCMAdapter::compileNetwork(*actualNetwork, _config, _graphBlob);
#else
        THROW_IE_EXCEPTION << "Network compilation is disabled";
#endif
    }

    if (_config.loadNetworkAfterCompilation()) {
        LoadBlob();
        ConfigureExecutor(_netName);
    }
}

ExecutableNetwork::ExecutableNetwork(std::istream& strm, const KmbConfig& config): _config(config) {
    IE_PROFILING_AUTO_SCOPE(ExecutableNetwork);
    _logger = std::make_shared<Logger>("ExecutableNetwork", _config.logLevel(), consoleOutput());
    _executor = std::make_shared<KmbExecutor>(_config);

    std::ostringstream blobContentStream;
    blobContentStream << strm.rdbuf();
    const std::string& blobContentString = blobContentStream.str();
    std::copy(blobContentString.begin(), blobContentString.end(), std::back_inserter(_graphBlob));
    LoadBlob();
    ConfigureExecutor("ExecutableNetwork");
}

void ExecutableNetwork::GetMetric(const std::string& name, Parameter& result, ResponseDesc* resp) const {
    UNUSED(resp);
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(4u));
    } else {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }
}

void ExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config, ResponseDesc* /* resp */) {
    for (const auto& entry : config) {
        _parsedConfig[entry.first] = entry.second;
    }
}

void ExecutableNetwork::GetConfig(const std::string& name, Parameter& result, ResponseDesc* /* resp */) const {
    auto res = _parsedConfig.find(name);
    if (res != _parsedConfig.end()) {
        result = res->second;
    } else {
        THROW_IE_EXCEPTION << name << " not found in the ExecutableNetwork config";
    }
}

ie::ITaskExecutor::Ptr ExecutableNetwork::getNextTaskExecutor() {
    std::string id = _taskExecutorGetResultIds.front();

    _taskExecutorGetResultIds.pop();
    _taskExecutorGetResultIds.push(id);

    ie::ExecutorManager* executorManager = ie::ExecutorManager::getInstance();
    ie::ITaskExecutor::Ptr taskExecutor = executorManager->getExecutor(id);

    return taskExecutor;
}

}  // namespace KmbPlugin
}  // namespace vpu
