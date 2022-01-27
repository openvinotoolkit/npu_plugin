//
// Copyright 2019-2020 Intel Corporation.
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
#include <fstream>

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <legacy/convert_function_to_cnn_network.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <threading/ie_executor_manager.hpp>

// Plugin
#include "vpux/utils/IE/config.hpp"
#include "vpux_async_infer_request.h"
#include "vpux_exceptions.h"
#include "vpux_executable_network.h"
#include "vpux_infer_request.h"

// Subplugin
#include "vpux.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux_compiler.hpp"

namespace vpux {
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static Executor::Ptr getExecutorForInference(const Executor::Ptr& executor, Logger logger) {
    if (executor == nullptr) {
        IE_THROW() << NO_EXECUTOR_FOR_INFERENCE;
    }

    try {
        return executor->clone();
    } catch (const std::exception& exc) {
        logger.warning("getExecutorForInference: executor threw an exception: {0}", exc.what());
        return executor;
    } catch (...) {
        logger.warning("getExecutorForInference: executor threw an unknown exception");
        return executor;
    }
}

//------------------------------------------------------------------------------
//      Shared init ctor
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(const Config& config, const Device::Ptr& device)
        : _config(config),
          _logger("ExecutableNetwork", config.get<LOG_LEVEL>()),
          _device(device),
          _compiler(Compiler::create(config)),
          _supportedMetrics({METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)}) {
}

//------------------------------------------------------------------------------
//      Load network
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(const IE::CNNNetwork& orignet, const Device::Ptr& device, const Config& config)
        : ExecutableNetwork(config, device) {
    // FIXME: This is a copy-paste from kmb_executable_network.cpp
    // should be fixed after switching to VPUX completely
    IE::CNNNetwork network = IE::details::cloneNetwork(orignet);
    if (const auto func = network.getFunction()) {
        IE::InputsDataMap inputsInfo = network.getInputsInfo();
        IE::OutputsDataMap outputsInfo = network.getOutputsInfo();
        try {
            _networkPtr = _compiler->compile(func, network.getName(), inputsInfo, outputsInfo, _config);
        } catch (const std::exception& ex) {
            IE_THROW() << ex.what();
        } catch (...) {
            _logger.error("Unexpected exception");
            IE_THROW() << "VPUX ExecutableNetwork got unexpected exception from compiler";
        }
    } else {
        _logger.warning("Failed to read NGraph network");
        IE_THROW() << "Failed to read NGraph network";
    }

    // TODO: Fix this WA for EISW-22783, EISW-25449
    bool IE_VPUX_CREATE_EXECUTOR = true;
    if (const auto var = std::getenv("IE_VPUX_CREATE_EXECUTOR")) {
        IE_VPUX_CREATE_EXECUTOR = vpux::envVarStrToBool("IE_VPUX_CREATE_EXECUTOR", var);
    }

    if (IE_VPUX_CREATE_EXECUTOR) {
        _executorPtr = createExecutor(_networkPtr, _config, device);
        ConfigureStreamsExecutor(network.getName());
    }
}

//------------------------------------------------------------------------------
//      Import network
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(std::istream& networkModel, const Device::Ptr& device, const Config& config)
        : ExecutableNetwork(config, device) {
    try {
        const std::string networkName = "net" + std::to_string(loadBlobCounter);
        _networkPtr = _compiler->parse(networkModel, _config, networkName);
        _executorPtr = createExecutor(_networkPtr, _config, device);
        _networkInputs = helpers::dataMapIntoInputsDataMap(_networkPtr->getInputsInfo());
        _networkOutputs = helpers::dataMapIntoOutputsDataMap(_networkPtr->getOutputsInfo());
        setInputs(helpers::ovRawNodesIntoOVNodes(_networkPtr->getOVParameters(), false));
        setOutputs(helpers::ovRawNodesIntoOVNodes(_networkPtr->getOVResults(), true));
        ConfigureStreamsExecutor(networkName);
    } catch (const std::exception& ex) {
        IE_THROW() << ex.what();
    } catch (...) {
        _logger.error("Unexpected exception");
        IE_THROW() << "VPUX ExecutableNetwork got unexpected exception from compiler";
    }
}

void ExecutableNetwork::ConfigureStreamsExecutor(const std::string& networkName) {
    size_t maxTaskExecutorGetResultCount = 1;
    if (_config.get<EXCLUSIVE_ASYNC_REQUESTS>()) {
        IE::ExecutorManager* executorManager = IE::ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor("VPUX");
        maxTaskExecutorGetResultCount = 1;
    } else {
        _taskExecutor = std::make_shared<IE::CPUStreamsExecutor>(IE::IStreamsExecutor::Config{
                "VPUXPlugin executor", checked_cast<int>(_config.get<EXECUTOR_STREAMS>())});
        maxTaskExecutorGetResultCount = _config.get<EXECUTOR_STREAMS>();
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
IE::IInferRequestInternal::Ptr ExecutableNetwork::CreateInferRequestImpl(const IE::InputsDataMap networkInputs,
                                                                         const IE::OutputsDataMap networkOutputs) {
    if (!_executorPtr) {
        _executorPtr = createExecutor(_networkPtr, _config, _device);
        ConfigureStreamsExecutor(_networkName);
    }
    const auto inferExecutor = getExecutorForInference(_executorPtr, _logger);
    const auto allocator = _device->getAllocator();
    return std::make_shared<InferRequest>(networkInputs, networkOutputs, inferExecutor, _config, _networkName,
                                          _parameters, _results, allocator);
}

InferenceEngine::IInferRequestInternal::Ptr ExecutableNetwork::CreateInferRequest() {
    if (!_executorPtr) {
        _executorPtr = createExecutor(_networkPtr, _config, _device);
        ConfigureStreamsExecutor(_networkName);
    }
    const auto inferExecutor = getExecutorForInference(_executorPtr, _logger);
    const auto allocator = _device->getAllocator();
    auto syncRequestImpl = std::make_shared<InferRequest>(_networkInputs, _networkOutputs, inferExecutor, _config,
                                                          _networkName, _parameters, _results, allocator);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    return std::make_shared<AsyncInferRequest>(syncRequestImpl, _taskExecutor, getNextTaskExecutor(),
                                               _callbackExecutor);
}

//------------------------------------------------------------------------------
//      Export
//------------------------------------------------------------------------------

namespace {
std::uint32_t hash(const std::vector<char>& data) {
    std::uint32_t result = 1171117u;
    for (const char& c : data)
        result = ((result << 7) + result) + static_cast<uint32_t>(c);
    return result;
}

}  // namespace

void ExecutableNetwork::Export(std::ostream& model) {
    auto graphBlob = _networkPtr->getCompiledNetwork();
    model.write(graphBlob.data(), graphBlob.size());
    std::stringstream str;
    str << "Blob hash: " << std::hex << hash(graphBlob);
    _logger.info("{0}", str.str());
}

void ExecutableNetwork::Export(const std::string& modelFileName) {
    std::ofstream modelFile(modelFileName, std::ios::binary);

    if (modelFile.is_open()) {
        Export(modelFile);
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

Executor::Ptr ExecutableNetwork::createExecutor(const NetworkDescription::Ptr& network, const Config& config,
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
