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
#include <memory>
#include <utility>

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <ie_itt.hpp>
#include <net_pass.h>
#include <generic_ie.hpp>
#include <convert_function_to_cnn_network.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>

#include <vpux_compiler.hpp>
#include <vpux.hpp>

#include "kmb_remote_context.h"
#include "file_reader.h"
#include "kmb_executable_network.h"

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

std::atomic<int> ExecutableNetwork::loadBlobCounter{1};

void ExecutableNetwork::LoadBlob() {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "LoadBlob");
    const std::string networkName = "net" + std::to_string(loadBlobCounter);
    loadBlobCounter++;  // increment blob static counter to make unique network ID
    const auto& deviceId = ::utils::extractIdFromDeviceName(_device->getName());
    _executor = std::make_shared<KmbExecutor>(_networkDescription, _device->getAllocator(), deviceId, _config);

    _networkInputs = vpux::helpers::dataMapIntoInputsDataMap(_networkDescription->getInputsInfo());
    _networkOutputs = vpux::helpers::dataMapIntoOutputsDataMap(_networkDescription->getOutputsInfo());
    _netName = _networkDescription->getName();
}

ExecutableNetwork::ExecutableNetwork(const KmbConfig& config, const std::shared_ptr<vpux::Device>& device)
    : _config(config), _compiler(vpux::ICompiler::create(vpux::CompilerType::MCMCompiler)), _device(device) {}

ExecutableNetwork::ExecutableNetwork(
    ICNNNetwork& network, const KmbConfig& config, const std::shared_ptr<vpux::Device>& device)
    : ExecutableNetwork(config, device) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "ExecutableNetwork");

    _netName = network.getName();
    _supportedMetrics = {METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)};

    _logger = std::make_shared<Logger>("ExecutableNetwork", _config.logLevel(), consoleOutput());

    const bool kmb_use_ngraph = (NULL != getenv("KMB_USE_NGRAPH_PARSER"));
    if (kmb_use_ngraph || _config.useNGraphParser()) {
        if (const auto func = network.getFunction()) {
            _logger->info("Using NGraph parser");
            InputsDataMap inputsInfo;
            network.getInputsInfo(inputsInfo);

            OutputsDataMap outputsInfo;
            network.getOutputsInfo(outputsInfo);

            _networkDescription = _compiler->compile(func, network.getName(), inputsInfo, outputsInfo, _config);
        } else {
            _logger->warning("Failed to read NGraph network");
        }
    } else {
        _logger->info("NGraph parser disabled");
        _logger->info("Using CNNNetwork parser");
        // HACK: convert nGraph to old CNNNetwork to fix LP transformations

        std::shared_ptr<ICNNNetwork> convertedNetwork;
        auto actualNetwork = &network;
        (void)actualNetwork;

        if (network.getFunction()) {
            auto nGraphFunc = network.getFunction();
            // Disable shape inference (WA for generic operations)
            ::ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

            // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
            ngraph::pass::ConvertPriorBox().run_on_function(nGraphFunc);
            ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(nGraphFunc);
            ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(nGraphFunc);
            convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, network, true);
            actualNetwork = convertedNetwork.get();
        }

        _networkDescription = _compiler->compile(*actualNetwork, _config);
    }
    if (_device) {
        LoadBlob();
        ConfigureExecutor(_netName);
    }
}

ExecutableNetwork::ExecutableNetwork(
    std::istream& strm, const KmbConfig& config, const std::shared_ptr<vpux::Device>& device)
    : ExecutableNetwork(config, device) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "ExecutableNetwork");
    _logger = std::make_shared<Logger>("ExecutableNetwork", _config.logLevel(), consoleOutput());

    _networkDescription = _compiler->parse(strm, _config);
    if (_device) {
        LoadBlob();
        ConfigureExecutor("ExecutableNetwork");
    }
}

void ExecutableNetwork::GetMetric(const std::string& name, Parameter& result, ResponseDesc*) const {
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
