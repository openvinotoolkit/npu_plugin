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

#define NOMINMAX
#include <ie_blob.h>
#include <ie_plugin.hpp>
#include <description_buffer.hpp>
#include <debug.h>
#include <ie_layouts.h>
#include <precision_utils.h>

#include <vpu/utils/perf_report.hpp>
#include <vpu/utils/ie_helpers.hpp>

#include "kmb_executable_network.h"
#include "kmb_infer_request.h"

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;

#define MEMCPY(dst, src, bytes) std::copy_n((src), (bytes), (dst))

KmbInferRequest::KmbInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                        InferenceEngine::OutputsDataMap networkOutputs,
                                        DataInfo& inputInfo,
                                        DataInfo& outputInfo,
                                        const std::vector<StageMetaInfo> &blobMetaData,
                                        const std::shared_ptr<KmbConfig> &kmbConfig,
                                        const Logger::Ptr &log,
                                        const KmbExecutorPtr &executor) :
        InferRequestInternal(networkInputs, networkOutputs), _executor(executor),
        _log(log), _stagesMetaData(blobMetaData), _config(kmbConfig),
        _inputInfo(inputInfo), _outputInfo(outputInfo) {
    _deviceLayout = NCHW;

    if (_config->compileConfig.forceLayout == ComputeLayout::NCHW)
        _deviceLayout = NCHW;
    if (_config->compileConfig.forceLayout == ComputeLayout::NHWC)
        _deviceLayout = NHWC;
    // allocate inputs
    IE_ASSERT(_networkInputs.size() == 1) << "Do not support more than 1 input";
    for (auto &networkInput : _networkInputs) {
        // TODO: use TensorDesc instead of deprecated methods
        SizeVector dims      = networkInput.second->getTensorDesc().getDims();
        Precision  precision = networkInput.second->getTensorDesc().getPrecision();
        Layout     layout    = networkInput.second->getTensorDesc().getLayout();

        if (precision != Precision::FP32 &&
            precision != Precision::FP16 &&
            precision != Precision::U8 &&
            precision != Precision::I8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input precision: "
                                   << precision << "! Supported precisions: FP32, FP16, U8, I8";
        }

        Blob::Ptr inputBlob = make_blob_with_precision(TensorDesc(
            precision,
            dims,
            layout), executor->getAllocator());
        inputBlob->allocate();

        _inputs[networkInput.first] = inputBlob;
    }
    // allocate outputs
    IE_ASSERT(_networkOutputs.size() == 1) << "Do not support more than 1 output";
    for (auto &networkOutput : _networkOutputs) {
        SizeVector dims      = networkOutput.second->getTensorDesc().getDims();
        Precision  precision = networkOutput.second->getTensorDesc().getPrecision();
        Layout     layout    = networkOutput.second->getTensorDesc().getLayout();

        if (precision != Precision::FP32 &&
            precision != Precision::FP16 &&
            precision != Precision::U8 &&
            precision != Precision::I8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output precision: "
                                << precision << "! Supported precisions: FP32, FP16, U8, I8";
        }

        Blob::Ptr outputBlob = make_blob_with_precision(TensorDesc(
            precision,
            dims,
            layout), executor->getAllocator());

        outputBlob->allocate();
        _outputs[networkOutput.first] = outputBlob;
    }

    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "Internal error: no information about network's output/input";
    }
}
void KmbInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void KmbInferRequest::InferAsync() {
    for (auto input : _inputs) {
        auto const inputBlobPtr = input.second;
        auto inputBlobPrecision = inputBlobPtr->getTensorDesc().getPrecision();
        if (inputBlobPrecision != Precision::FP16
            && inputBlobPrecision != Precision::FP32
            && inputBlobPrecision != Precision::I8
            && inputBlobPrecision != Precision::U8)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input blob precision";
    }
    for (auto output : _outputs) {
        auto const outputBlobPtr = output.second;
        auto outputBlobPrecision = outputBlobPtr->getTensorDesc().getPrecision();
        if (outputBlobPrecision != Precision::FP16
            && outputBlobPrecision != Precision::FP32
            && outputBlobPrecision != Precision::I8
            && outputBlobPrecision != Precision::U8)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output blob precision";
    }

    // execute input pre-processing
    execDataPreprocessing(_inputs, true);  // "true" stands for serial preprocessing in case of OpenMP

    auto dataName = _networkInputs.begin()->first;
    auto foundInputBlob = _inputs.find(dataName);
    if (foundInputBlob == _inputs.end())
        THROW_IE_EXCEPTION << "Error: input [" << dataName << "] is not provided.";

    void* inputPtr = _inputs.begin()->second->buffer();
    size_t inputSize = _inputInfo.totalSize;

    _executor->queueInference(inputPtr, inputSize, nullptr, 0);
}

void KmbInferRequest::GetResult() {
    auto dataName = _networkOutputs.begin()->first;
    auto foundInputBlob = _outputs.find(dataName);
    if (foundInputBlob == _outputs.end())
        THROW_IE_EXCEPTION << "Error: output [" << dataName << "] is not provided.";

    void* outputPtr = _outputs.begin()->second->buffer();
    size_t outputSize = _outputInfo.totalSize;


    _executor->getResult(outputPtr, outputSize);
}

void KmbInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const {
    THROW_IE_EXCEPTION << "KmbInferRequest::GetPerformanceCounts is not implemented\n";
}
