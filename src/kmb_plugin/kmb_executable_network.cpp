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
#include <generic_ie.hpp>
#include <convert_function_to_cnn_network.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>

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
    _executor->allocateGraph(_graphBlob);
    _networkInputs = _executor->getNetworkInputs();
    _networkOutputs = _executor->getNetworkOutputs();
}

ExecutableNetwork::ExecutableNetwork(ICNNNetwork& network, const KmbConfig& config): _config(config) {
    IE_PROFILING_AUTO_SCOPE(ExecutableNetwork);

    _supportedMetrics = {METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)};

    _logger = std::make_shared<Logger>("ExecutableNetwork", _config.logLevel(), consoleOutput());
    _executor = std::make_shared<KmbExecutor>(_config);

    if (_config.useNGraphParser()) {
        if (const auto func = std::const_pointer_cast<ngraph::Function>(network.getFunction())) {
            InputsDataMap inputsInfo;
            network.getInputsInfo(inputsInfo);

            OutputsDataMap outputsInfo;
            network.getOutputsInfo(outputsInfo);

            _graphBlob = compileNGraph(func, network.getName(), inputsInfo, outputsInfo, _config);
        }
    }

    if (_graphBlob.empty()) {
#ifdef ENABLE_MCM_COMPILER
        // HACK: convert nGraph to old CNNNetwork to fix LP transformations

        std::shared_ptr<ICNNNetwork> convertedNetwork;
        auto actualNetwork = &network;

        if (network.getFunction()) {
            auto nGraphFunc = network.getFunction();
            // Disable shape inference (WA for generic operations)
            ::ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

            // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
            ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(nGraphFunc);
            ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(nGraphFunc);
            convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, network);
            actualNetwork = convertedNetwork.get();
        }

        MCMAdapter::compileNetwork(*actualNetwork, _config, _graphBlob);
#else
        THROW_IE_EXCEPTION << "Network compilation is disabled";
#endif
    }

    if (_config.loadNetworkAfterCompilation()) {
        LoadBlob();
        ConfigureExecutor(network.getName());
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

void ExecutableNetwork::GetMetric(const std::string& name, Parameter& result, ResponseDesc*) const {
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(4u));
    } else {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }
}

}  // namespace KmbPlugin
}  // namespace vpu
