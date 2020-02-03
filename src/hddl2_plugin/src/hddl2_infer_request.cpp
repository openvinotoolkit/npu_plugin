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

#include "hddl2_infer_request.h"

#include <InferBlob.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

static void checkNetworkPrecision(const IE::Precision& precision) {
    if (precision != IE::Precision::FP32 && precision != IE::Precision::FP16 && precision != IE::Precision::U8 &&
        precision != IE::Precision::I8) {
        THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input precision: " << precision
                           << "! Supported precisions: FP32, FP16, U8, I8";
    }
}

static InferenceEngine::Blob::Ptr allocateLocalBlob(const IE::TensorDesc& tensorDesc) {
    checkNetworkPrecision(tensorDesc.getPrecision());

    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(tensorDesc);
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "InputBlob is nullptr.";
    }
    blob->allocate();
    return blob;
}

HDDL2InferRequest::HDDL2InferRequest(const IE::InputsDataMap& networkInputs, const IE::OutputsDataMap& networkOutputs,
    const HddlUniteGraph::Ptr& loadedGraph, const HDDL2RemoteContext::Ptr& context)
    : InferRequestInternal(networkInputs, networkOutputs), _loadedGraphPtr(loadedGraph) {
    _inferDataPtr = std::make_shared<HddlUniteInferData>(context);

    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;
        const IE::TensorDesc inputTensorDesc = networkInput.second->getTensorDesc();

        _inputs[inputName] = allocateLocalBlob(inputTensorDesc);
    }

    for (auto& networkOutput : _networkOutputs) {
        const std::string outputName = networkOutput.first;
        const IE::TensorDesc outputTensorDesc = networkOutput.second->getTensorDesc();

        _outputs[outputName] = allocateLocalBlob(outputTensorDesc);
    }
}

void HDDL2InferRequest::InferImpl() {
    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;

        auto foundInputBlob = _inputs.find(inputName);
        if (foundInputBlob == _inputs.end()) {
            THROW_IE_EXCEPTION << "Error: input [" << inputName << "] is not provided.";
        }
        const IE::Blob::Ptr inputBlobPtr = foundInputBlob->second;

        _inferDataPtr->prepareInput(inputName, inputBlobPtr);
    }

    for (const auto& networkOutput : _networkOutputs) {
        const std::string outputName = networkOutput.first;

        auto foundOutputBlob = _outputs.find(outputName);
        if (foundOutputBlob == _outputs.end()) {
            THROW_IE_EXCEPTION << "Error: output [" << outputName << "] is not provided.";
        }
        const IE::Blob::Ptr outputBlobPtr = foundOutputBlob->second;

        _inferDataPtr->prepareOutput(outputName, outputBlobPtr);
    }
    _loadedGraphPtr->InferSync(_inferDataPtr);
}

void vpu::HDDL2Plugin::HDDL2InferRequest::GetPerformanceCounts(
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const {
    UNUSED(perfMap);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}
