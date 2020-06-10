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

#include <hddl2_async_infer_request.h>
#include <hddl2_executable_network.h>
#include <hddl2_helpers.h>
#include <hddl2_infer_request.h>
#include <hddl2_metrics.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <threading/ie_executor_manager.hpp>
#include <vector>

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

static HDDL2RemoteContext::Ptr castIEContextToHDDL2(const IE::RemoteContext::Ptr& ieContext) {
    HDDL2RemoteContext::Ptr pluginContext = nullptr;

    if (ieContext == nullptr) {
        return pluginContext;
    }

    try {
        pluginContext = std::dynamic_pointer_cast<HDDL2RemoteContext>(ieContext);
    } catch (const std::exception& ex) {
        THROW_IE_EXCEPTION << "Incorrect context for HDDL2 Plugin! Error: " << ex.what();
    }
    return pluginContext;
}

ExecutableNetwork::ExecutableNetwork(
    const std::string& blobFilename, const HDDL2Config& config, const IE::RemoteContext::Ptr& ieContext)
    : _config(config), _logger(std::make_shared<Logger>("ExecutableNetwork", config.logLevel(), consoleOutput())) {
    _graphPtr = std::make_shared<ImportedGraph>(blobFilename, config);
    _context = castIEContextToHDDL2(ieContext);
    if (_context == nullptr) {
        _loadedGraph = std::make_shared<HddlUniteGraph>(_graphPtr, config.device_id(), config.logLevel());
    } else {
        _loadedGraph = std::make_shared<HddlUniteGraph>(_graphPtr, _context, config.logLevel());
    }

    this->_networkInputs = _graphPtr->getInputsInfo();
    this->_networkOutputs = _graphPtr->getOutputsInfo();
}

ExecutableNetwork::ExecutableNetwork(
    IE::ICNNNetwork& network, const HDDL2Config& config, const IE::RemoteContext::Ptr& ieContext)
    : _config(config), _logger(std::make_shared<Logger>("ExecutableNetwork", config.logLevel(), consoleOutput())) {
    _graphPtr = std::make_shared<CompiledGraph>(network, config);
    _context = castIEContextToHDDL2(ieContext);

    if (HDDL2Metrics::isServiceAvailable()) {
        if (_context == nullptr) {
            _loadedGraph = std::make_shared<HddlUniteGraph>(_graphPtr, config.device_id(), config.logLevel());
        } else {
            _loadedGraph = std::make_shared<HddlUniteGraph>(_graphPtr, _context, config.logLevel());
        }
    } else {
        _logger->warning("HDDL2 Scheduler service is not available. "
                         "Please make sure KMB_INSTALL_DIR is specified.");
    }
}

ExecutableNetwork::ExecutableNetwork(
    std::istream& networkModel, const HDDL2Config& config, const InferenceEngine::RemoteContext::Ptr& ieContext)
    : _config(config), _logger(std::make_shared<Logger>("ExecutableNetwork", config.logLevel(), consoleOutput())) {
    _graphPtr = std::make_shared<ImportedGraph>(networkModel, config);
    _context = castIEContextToHDDL2(ieContext);
    if (_context == nullptr) {
        _loadedGraph = std::make_shared<HddlUniteGraph>(_graphPtr, config.device_id(), config.logLevel());
    } else {
        _loadedGraph = std::make_shared<HddlUniteGraph>(_graphPtr, _context, config.logLevel());
    }
    this->_networkInputs = _graphPtr->getInputsInfo();
    this->_networkOutputs = _graphPtr->getOutputsInfo();
}

IE::InferRequestInternal::Ptr vpu::HDDL2Plugin::ExecutableNetwork::CreateInferRequestImpl(
    const IE::InputsDataMap networkInputs, const IE::OutputsDataMap networkOutputs) {
    if (_loadedGraph == nullptr) {
        THROW_IE_EXCEPTION << "Can not create infer request without network loaded on device";
    }

    return std::make_shared<HDDL2InferRequest>(networkInputs, networkOutputs, _loadedGraph, _context, _config);
}

void ExecutableNetwork::ExportImpl(std::ostream& model) {
    auto graphBlob = _graphPtr->getGraphBlob();
    model.write(graphBlob.data(), graphBlob.size());
}

void ExecutableNetwork::Export(const std::string& modelFileName) {
    std::ofstream modelFile(modelFileName, std::ios::binary);

    if (modelFile.is_open()) {
        ExportImpl(modelFile);
    } else {
        THROW_IE_EXCEPTION << "The " << modelFileName << " file can not be opened for export.";
    }
}

void ExecutableNetwork::CreateInferRequest(InferenceEngine::IInferRequest::Ptr& asyncRequest) {
    auto syncRequestImpl =
        std::make_shared<HDDL2InferRequest>(_networkInputs, _networkOutputs, _loadedGraph, _context, _config);

    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());

    const std::string resultExecutorName = "HDDL2ResultExecutor";
    auto resultExecutor = IE::ExecutorManager::getInstance()->getExecutor(resultExecutorName);

    const std::string waitExecutorName = "HDDL2WaitExecutor";
    auto waitExecutor = IE::ExecutorManager::getInstance()->getExecutor(waitExecutorName);

    auto asyncTreadSafeImpl = std::make_shared<HDDL2AsyncInferRequest>(
        syncRequestImpl, _taskExecutor, resultExecutor, waitExecutor, _callbackExecutor);
    asyncRequest.reset(
        new InferenceEngine::InferRequestBase<InferenceEngine::AsyncInferRequestThreadSafeDefault>(asyncTreadSafeImpl),
        [](InferenceEngine::IInferRequest* p) {
            p->Release();
        });
    asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
}
