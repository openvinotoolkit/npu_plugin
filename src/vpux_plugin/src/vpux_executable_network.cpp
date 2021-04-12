//
// Copyright 2019-2020 Intel Corporation.
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

// System
#include <fstream>

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <legacy/convert_function_to_cnn_network.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <threading/ie_executor_manager.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>
#include <transformations/op_conversions/reduce_l1_decomposition.hpp>
#include <transformations/op_conversions/reduce_l2_decomposition.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>

// Plugin
#include "vpux_async_infer_request.h"
#include "vpux_exceptions.h"
#include "vpux_executable_network.h"
#include "vpux_infer_request.h"
// Subplugin
#include "vpux.hpp"
#include "vpux_compiler.hpp"

namespace vpux {
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static Executor::Ptr getExecutorForInference(const Executor::Ptr& executor, const vpu::Logger::Ptr& logger) {
    if (executor == nullptr) {
        IE_THROW() << NO_EXECUTOR_FOR_INFERENCE;
    }

    try {
        return executor->clone();
    } catch (const std::exception& exc) {
        logger->warning("getExecutorForInference: executor threw an exception: %s", exc.what());
        return executor;
    } catch (...) {
        logger->warning("getExecutorForInference: executor threw an unknown exception");
        return executor;
    }
}

//------------------------------------------------------------------------------
//      Shared init ctor
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(const VPUXConfig& config, const Device::Ptr& device)
        : _config(config),
          _logger(std::make_shared<vpu::Logger>("ExecutableNetwork", config.logLevel(), vpu::consoleOutput())),
          _device(device),
          _compiler(Compiler::create(config)),
          _supportedMetrics({METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)}) {
    // FIXME: HACK: platform is supposed to be detected at this point
    // but due to different reasons it is not possible now.
    // KMB B0 platform is hard-coded the main one
    // Track number: TBD
    if (device != nullptr) {
        const_cast<VPUXConfig&>(_config).update({{VPUX_CONFIG_KEY(PLATFORM), VPUX_CONFIG_VALUE(VPU3700)}});
    }
}

//------------------------------------------------------------------------------
//      Load network
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(IE::CNNNetwork& network, const Device::Ptr& device, const VPUXConfig& config)
        : ExecutableNetwork(config, device) {
    // FIXME: This is a copy-paste from kmb_executable_network.cpp
    // should be fixed after switching to VPUX completely
    if (const auto func = network.getFunction()) {
        IE::InputsDataMap inputsInfo = network.getInputsInfo();
        IE::OutputsDataMap outputsInfo = network.getOutputsInfo();

        _networkPtr = _compiler->compile(func, network.getName(), inputsInfo, outputsInfo, _config);
    } else {
        _logger->warning("Failed to read NGraph network");
        IE_THROW() << "Failed to read NGraph network";
    }

    _executorPtr = createExecutor(_networkPtr, config, device);
    ConfigureStreamsExecutor(network.getName());
}

//------------------------------------------------------------------------------
//      Import network
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(std::istream& networkModel, const Device::Ptr& device, const VPUXConfig& config)
        : ExecutableNetwork(config, device) {
    const std::string networkName = "net" + std::to_string(loadBlobCounter);
    _networkPtr = _compiler->parse(networkModel, _config, networkName);
    _executorPtr = createExecutor(_networkPtr, config, device);
    _networkInputs = helpers::dataMapIntoInputsDataMap(_networkPtr->getInputsInfo());
    _networkOutputs = helpers::dataMapIntoOutputsDataMap(_networkPtr->getOutputsInfo());
    ConfigureStreamsExecutor(networkName);
}

void ExecutableNetwork::ConfigureStreamsExecutor(const std::string& networkName) {
    size_t maxTaskExecutorGetResultCount = 1;
    if (_config.exclusiveAsyncRequests()) {
        IE::ExecutorManager* executorManager = IE::ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor("VPUX");
        maxTaskExecutorGetResultCount = 1;
    } else {
        _taskExecutor = std::make_shared<IE::CPUStreamsExecutor>(
                IE::IStreamsExecutor::Config{"VPUXPlugin executor", _config.executorStreams()});
        maxTaskExecutorGetResultCount = _config.executorStreams();
    }

    for (size_t i = 0; i < maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_VPUXResultExecutor" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
}

IE::ITaskExecutor::Ptr ExecutableNetwork::getNextTaskExecutor() {
    std::string id = _taskExecutorGetResultIds.front();

    _taskExecutorGetResultIds.pop();
    _taskExecutorGetResultIds.push(id);

    IE::ExecutorManager* executorManager = IE::ExecutorManager::getInstance();
    IE::ITaskExecutor::Ptr taskExecutor = executorManager->getExecutor(id);

    return taskExecutor;
}

//------------------------------------------------------------------------------
//      Create infer requests
//------------------------------------------------------------------------------
IE::InferRequestInternal::Ptr ExecutableNetwork::CreateInferRequestImpl(const IE::InputsDataMap networkInputs,
                                                                        const IE::OutputsDataMap networkOutputs) {
    const auto inferExecutor = getExecutorForInference(_executorPtr, _logger);
    const auto allocator = _device->getAllocator();
    return std::make_shared<InferRequest>(networkInputs, networkOutputs, inferExecutor, _config, _networkName,
                                          allocator);
}

InferenceEngine::IInferRequest::Ptr ExecutableNetwork::CreateInferRequest() {
    const auto inferExecutor = getExecutorForInference(_executorPtr, _logger);
    const auto allocator = _device->getAllocator();
    auto syncRequestImpl = std::make_shared<InferRequest>(_networkInputs, _networkOutputs, inferExecutor, _config,
                                                          _networkName, allocator);

    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());

    auto resultExecutor = getNextTaskExecutor();

    auto asyncThreadSafeImpl =
            std::make_shared<AsyncInferRequest>(syncRequestImpl, _taskExecutor, resultExecutor, _callbackExecutor);

    InferenceEngine::IInferRequest::Ptr asyncRequest;
    asyncRequest.reset(new InferenceEngine::InferRequestBase(asyncThreadSafeImpl));
    asyncThreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
    return asyncRequest;
}

//------------------------------------------------------------------------------
//      Export
//------------------------------------------------------------------------------
void ExecutableNetwork::ExportImpl(std::ostream& model) {
    auto graphBlob = _networkPtr->getCompiledNetwork();
    model.write(graphBlob.data(), graphBlob.size());
}

void ExecutableNetwork::Export(const std::string& modelFileName) {
    std::ofstream modelFile(modelFileName, std::ios::binary);

    if (modelFile.is_open()) {
        ExportImpl(modelFile);
    } else {
        IE_THROW() << "The " << modelFileName << " file can not be opened for export.";
    }
}

IE::Parameter ExecutableNetwork::GetMetric(const std::string& name) const {
    if (name == METRIC_KEY(NETWORK_NAME)) {
        if (_networkPtr != nullptr) {
            IE_SET_METRIC_RETURN(NETWORK_NAME, _networkPtr->getName());
        } else {
            IE_THROW() << "GetMetric: network is not initialized";
        }
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(8u));
    } else {
        IE_THROW(NotImplemented);
    }

    return {};
}

//------------------------------------------------------------------------------
std::atomic<int> ExecutableNetwork::loadBlobCounter{1};

Executor::Ptr ExecutableNetwork::createExecutor(const NetworkDescription::Ptr& network, const VPUXConfig& config,
                                                const Device::Ptr& device) {
    loadBlobCounter++;  // increment blob static counter to make unique network ID
    if (network == nullptr) {
        IE_THROW() << "Network is null!";
    }

    // Default executor is nullptr, allow only perform export
    Executor::Ptr executor = nullptr;
    if (device != nullptr) {
        executor = device->createExecutor(network, config);
    }
    _networkName = _networkPtr->getName();
    return executor;
}

}  // namespace vpux
