//
// Copyright 2022 Intel Corporation.
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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <debug.h>
#include <ie_blob.h>
#include <blob_factory.hpp>

#include "zero_executor.h"
#include "zero_infer_request.h"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"

#include <device_helpers.hpp>
#include "vpux/utils/IE/blob.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/checked_cast.hpp"

namespace IE = InferenceEngine;
using namespace vpux;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static void checkNetworkPrecision(const IE::Precision& precision) {
    if (precision != IE::Precision::FP32 && precision != IE::Precision::FP16 && precision != IE::Precision::U8 &&
        precision != IE::Precision::I8 && precision != IE::Precision::I32 && precision != IE::Precision::U32) {
        IE_THROW(ParameterMismatch) << "Unsupported input precision: " << precision
                                    << "! Supported precisions: FP32, FP16, U8, I8, I32, U32";
    }
}

static IE::Blob::Ptr allocateLocalBlob(const IE::TensorDesc& tensorDesc, void* dataPtr) {
    checkNetworkPrecision(tensorDesc.getPrecision());

    IE::Blob::Ptr blob;
    if (dataPtr) {
        blob = make_blob_with_precision(tensorDesc, dataPtr);
    } else {
        blob = make_blob_with_precision(tensorDesc);
        if (blob) {
            blob->allocate();
        }
    }
    if (nullptr == blob) {
        IE_THROW() << "Can't make blob.";
    }
    return blob;
}

//------------------------------------------------------------------------------
ZeroInferRequest::ZeroInferRequest(const IE::InputsDataMap& networkInputs, const IE::OutputsDataMap& networkOutputs,
                                   const Executor::Ptr& executor, const Config& config, const std::string& /*netName*/,
                                   const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                   const std::vector<std::shared_ptr<const ov::Node>>& results,
                                   const std::shared_ptr<InferenceEngine::IAllocator>& allocator)
        : IInferRequest(networkInputs, networkOutputs),
          _executorPtr(executor),
          _config(config),
          _logger("ZeroInferRequest", config.get<LOG_LEVEL>()),
          _allocator(allocator) {
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        IE_THROW() << "No information about network's output/input.";
    }
    _parameters = parameters;
    _results = results;

    // we assume that _executorPtr contains ZeroExecutor ptr only
    auto& pipeline = static_cast<ZeroExecutor*>(_executorPtr.get())->getPipeline();
    const auto& deviceInputs = static_cast<ZeroExecutor*>(_executorPtr.get())->getNetworkDesc().getDeviceInputsInfo();

    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;
        const IE::TensorDesc inputTensorDesc = networkInput.second->getTensorDesc();

        if (isRepackingRequired(inputTensorDesc, mapArguments(deviceInputs, inputName)->getTensorDesc())) {
            _inputs[inputName] = allocateLocalBlob(inputTensorDesc, nullptr);
        } else {
            _inputs[inputName] =
                    allocateLocalBlob(inputTensorDesc, mapArguments(pipeline._inputs_host_mem_map, inputName).data());
        }
    }

    const auto& deviceeOutputs =
            static_cast<ZeroExecutor*>(_executorPtr.get())->getNetworkDesc().getDeviceOutputsInfo();
    for (const auto& networkOutput : _networkOutputs) {
        const std::string outputName = networkOutput.first;
        const IE::TensorDesc outputTensorDesc = networkOutput.second->getTensorDesc();

        if (isRepackingRequired(outputTensorDesc, mapArguments(deviceeOutputs, outputName)->getTensorDesc())) {
            _outputs[outputName] = allocateLocalBlob(outputTensorDesc, nullptr);
        } else {
            _outputs[outputName] = allocateLocalBlob(outputTensorDesc,
                                                     mapArguments(pipeline._outputs_host_mem_map, outputName).data());
        }
    }
}

void ZeroInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void ZeroInferRequest::InferAsync() {
    _logger.debug("InferRequest::InferAsync started");
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "InferAsync");

    execDataPreprocessing(_inputs);
    _executorPtr->push(_inputs);
}

void ZeroInferRequest::GetResult() {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "GetResult");
    _executorPtr->pull(_outputs);
    _logger.debug("InferRequest::GetResult finished");
}

std::map<std::string, IE::InferenceEngineProfileInfo> ZeroInferRequest::GetPerformanceCounts() const {
    if (_config.get<PERF_COUNT>()) {
        return _executorPtr->getLayerStatistics();
    } else {
        return {};
    }
}
