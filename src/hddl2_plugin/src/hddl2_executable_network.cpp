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
#include <hddl2_exceptions.h>
#include <hddl2_executable_network.h>
#include <hddl2_infer_request.h>
#include <hddl2_metrics.h>

#include <fstream>
#include <memory>
#include <string>
#include <threading/ie_executor_manager.hpp>
#include <vector>
#include <vpux_compiler.hpp>

namespace vpu {
namespace HDDL2Plugin {

namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Shared init ctor
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(const vpu::HDDL2Config& config)
    : _config(config),
      _logger(std::make_shared<Logger>("ExecutableNetwork", config.logLevel(), consoleOutput())),
      _compiler(vpux::ICompiler::create(vpux::CompilerType::MCMCompiler)) {}

//------------------------------------------------------------------------------
//      Load network
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(
    IE::ICNNNetwork& network, const vpu::HDDL2Config& config, const IE::RemoteContext::Ptr& ieContext)
    : ExecutableNetwork(config) {
    _networkPtr = _compiler->compile(network, _config);
    _executorPtr = vpux::HDDL2::HDDL2Executor::prepareExecutor(_networkPtr, config, ieContext);
}

//------------------------------------------------------------------------------
//      Import network
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(
    std::istream& networkModel, const vpu::HDDL2Config& config, const InferenceEngine::RemoteContext::Ptr& ieContext)
    : ExecutableNetwork(config) {
    _networkPtr = _compiler->parse(networkModel, _config);
    _executorPtr = vpux::HDDL2::HDDL2Executor::prepareExecutor(_networkPtr, config, ieContext);

    _networkInputs = vpux::helpers::dataMapIntoInputsDataMap(_networkPtr->getInputsInfo());
    _networkOutputs = vpux::helpers::dataMapIntoOutputsDataMap(_networkPtr->getOutputsInfo());
}

//------------------------------------------------------------------------------
//      Create infer requests
//------------------------------------------------------------------------------
IE::InferRequestInternal::Ptr vpu::HDDL2Plugin::ExecutableNetwork::CreateInferRequestImpl(
    const IE::InputsDataMap networkInputs, const IE::OutputsDataMap networkOutputs) {
    if (_executorPtr == nullptr) {
        THROW_IE_EXCEPTION << NO_EXECUTOR_FOR_INFERENCE;
    }
    const auto inferExecutor = _executorPtr->clone();
    return std::make_shared<HDDL2InferRequest>(networkInputs, networkOutputs, inferExecutor, _config);
}

void ExecutableNetwork::CreateInferRequest(InferenceEngine::IInferRequest::Ptr& asyncRequest) {
    if (_executorPtr == nullptr) {
        THROW_IE_EXCEPTION << NO_EXECUTOR_FOR_INFERENCE;
    }

    const auto inferExecutor = _executorPtr->clone();
    auto syncRequestImpl = std::make_shared<HDDL2InferRequest>(_networkInputs, _networkOutputs, inferExecutor, _config);

    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());

    const std::string resultExecutorName = "HDDL2ResultExecutor";
    auto resultExecutor = IE::ExecutorManager::getInstance()->getExecutor(resultExecutorName);

    auto asyncThreadSafeImpl =
        std::make_shared<HDDL2AsyncInferRequest>(syncRequestImpl, _taskExecutor, resultExecutor, _callbackExecutor);
    asyncRequest.reset(
        new InferenceEngine::InferRequestBase<InferenceEngine::AsyncInferRequestThreadSafeDefault>(asyncThreadSafeImpl),
        [](InferenceEngine::IInferRequest* p) {
            p->Release();
        });
    asyncThreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
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
        THROW_IE_EXCEPTION << "The " << modelFileName << " file can not be opened for export.";
    }
}

}  // namespace HDDL2Plugin
}  // namespace vpu
