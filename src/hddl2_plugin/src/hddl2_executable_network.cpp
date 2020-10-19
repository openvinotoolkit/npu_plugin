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
// IE
#include <legacy/net_pass.h>

#include <generic_ie.hpp>
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
// Plugin
#include "hddl2_async_infer_request.h"
#include "hddl2_exceptions.h"
#include "hddl2_executable_network.h"
#include "hddl2_infer_request.h"
#include "hddl2_metrics.h"
// Subplugin
#include "vpux.hpp"
#include "vpux_compiler.hpp"

namespace vpu {
namespace HDDL2Plugin {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static vpux::Executor::Ptr createExecutor(const vpux::NetworkDescription::Ptr& network, const vpux::VPUXConfig& config,
    std::shared_ptr<vpux::Device>& device) {
    if (network == nullptr) {
        THROW_IE_EXCEPTION << "Network is null!";
    }
    // Default executor is nullptr, allow only perform export
    vpux::Executor::Ptr executor = nullptr;
    if (device != nullptr) {
        executor = device->createExecutor(network, config);
    }
    return executor;
}

static vpux::Executor::Ptr getExecutorForInference(const vpux::Executor::Ptr& executor) {
    if (executor == nullptr) {
        THROW_IE_EXCEPTION << NO_EXECUTOR_FOR_INFERENCE;
    }
    return executor->clone();
}
//------------------------------------------------------------------------------
//      Shared init ctor
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(const vpux::VPUXConfig& config)
    : _config(config),
      _logger(std::make_shared<Logger>("ExecutableNetwork", config.logLevel(), consoleOutput())),
      _compiler(vpux::Compiler::create(vpux::CompilerType::MCMCompiler)) {}

//------------------------------------------------------------------------------
//      Load network
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(
    IE::ICNNNetwork& network, std::shared_ptr<vpux::Device>& device, const vpux::VPUXConfig& config)
    : ExecutableNetwork(config) {
    _supportedMetrics = {METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)};
    // FIXME: This is a copy-paste from kmb_executable_network.cpp
    // should be fixed after switching to VPUX completely
    if (_config.useNGraphParser()) {
        if (const auto func = network.getFunction()) {
            _logger->info("Using NGraph parser");
            IE::InputsDataMap inputsInfo;
            network.getInputsInfo(inputsInfo);

            IE::OutputsDataMap outputsInfo;
            network.getOutputsInfo(outputsInfo);

            _networkPtr = _compiler->compile(func, network.getName(), inputsInfo, outputsInfo, _config);
        } else {
            _logger->warning("Failed to read NGraph network");
            THROW_IE_EXCEPTION << "Failed to read NGraph network";
        }
    } else {
        _logger->warning("Using Legacy parser");
        // HACK: convert nGraph to old CNNNetwork to fix LP transformations
        std::shared_ptr<IE::ICNNNetwork> convertedNetwork;
        auto actualNetwork = &network;

        if (network.getFunction()) {
            auto nGraphFunc = network.getFunction();
            ngraph::pass::Manager manager;
            ngraph::pass::ConvertPriorBox().run_on_function(
                nGraphFunc);  // strict requirement: ConvertPriorBox should be first

            manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
            manager.run_passes(nGraphFunc);
            // Disable shape inference (WA for generic operations)
            ::ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

            // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
            ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(nGraphFunc);

            manager.register_pass<ngraph::pass::ReduceL1Decomposition>();  // in CommonOptimizations.
            manager.register_pass<ngraph::pass::ReduceL2Decomposition>();
            auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();
            decomp->add_matcher<ngraph::pass::ConvertReduceMeanToPooling>();
            decomp->add_matcher<ngraph::pass::ConvertReduceMaxToPooling>();
            decomp->add_matcher<ngraph::pass::ConvertReduceSumToPooling>();
            decomp->set_name("ngraph::pass::Decompositions");
            manager.run_passes(nGraphFunc);

            ngraph::pass::ConstantFolding().run_on_function(nGraphFunc);
            ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(nGraphFunc);
            convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, network, true);
            actualNetwork = convertedNetwork.get();
        }

        _networkPtr = _compiler->compile(*actualNetwork, _config);
    }
    _executorPtr = createExecutor(_networkPtr, config, device);
}

//------------------------------------------------------------------------------
//      Import network
//------------------------------------------------------------------------------
ExecutableNetwork::ExecutableNetwork(
    std::istream& networkModel, std::shared_ptr<vpux::Device>& device, const vpux::VPUXConfig& config)
    : ExecutableNetwork(config) {
    _networkPtr = _compiler->parse(networkModel, _config);
    _executorPtr = createExecutor(_networkPtr, config, device);
    _networkInputs = vpux::helpers::dataMapIntoInputsDataMap(_networkPtr->getInputsInfo());
    _networkOutputs = vpux::helpers::dataMapIntoOutputsDataMap(_networkPtr->getOutputsInfo());
}

//------------------------------------------------------------------------------
//      Create infer requests
//------------------------------------------------------------------------------
IE::InferRequestInternal::Ptr vpu::HDDL2Plugin::ExecutableNetwork::CreateInferRequestImpl(
    const IE::InputsDataMap networkInputs, const IE::OutputsDataMap networkOutputs) {
    auto inferExecutor = getExecutorForInference(_executorPtr);
    return std::make_shared<HDDL2InferRequest>(networkInputs, networkOutputs, inferExecutor, _config);
}

InferenceEngine::IInferRequest::Ptr ExecutableNetwork::CreateInferRequest() {
    auto inferExecutor = getExecutorForInference(_executorPtr);
    auto syncRequestImpl = std::make_shared<HDDL2InferRequest>(_networkInputs, _networkOutputs, inferExecutor, _config);

    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());

    const std::string resultExecutorName = "HDDL2ResultExecutor";
    auto resultExecutor = IE::ExecutorManager::getInstance()->getExecutor(resultExecutorName);

    auto asyncThreadSafeImpl =
        std::make_shared<HDDL2AsyncInferRequest>(syncRequestImpl, _taskExecutor, resultExecutor, _callbackExecutor);

    InferenceEngine::IInferRequest::Ptr asyncRequest;
    asyncRequest.reset(
        new InferenceEngine::InferRequestBase<InferenceEngine::AsyncInferRequestThreadSafeDefault>(asyncThreadSafeImpl),
        [](InferenceEngine::IInferRequest* p) {
            p->Release();
        });
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
        THROW_IE_EXCEPTION << "The " << modelFileName << " file can not be opened for export.";
    }
}

IE::Parameter ExecutableNetwork::GetMetric(const std::string& name) const {
    if (name == METRIC_KEY(NETWORK_NAME)) {
        if (_networkPtr != nullptr) {
            IE_SET_METRIC_RETURN(NETWORK_NAME, _networkPtr->getName());
        } else {
            THROW_IE_EXCEPTION << "GetMetric: network is not initialized";
        }
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(8u));
    } else {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    return {};
}

}  // namespace HDDL2Plugin
}  // namespace vpu
