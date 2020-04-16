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
#include <ie_blob.h>
#include <ie_layouts.h>
#include <precision_utils.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "hddl2_remote_blob.h"
#include "ie_algorithm.hpp"
#include "ie_utils.hpp"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
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

static void copyDataToBlob(const IE::Blob::Ptr& dest, const void* source, size_t size) {
    if (source == nullptr) {
        THROW_IE_EXCEPTION << "Source data is nullptr!";
    }
    if (dest->byteSize() != size) {
        THROW_IE_EXCEPTION << "Output size mismatch between HddlUnite: " << size
                           << " and expected output: " << dest->byteSize();
    }
    IE::MemoryBlob::Ptr mblob = IE::as<IE::MemoryBlob>(dest);
    if (!mblob) {
        THROW_IE_EXCEPTION << "Failed output blob type!";
    }
    auto lockedMemory = mblob->wmap();
    void* data = lockedMemory.as<void*>();
    auto result = ie_memcpy(data, dest->byteSize(), source, size);
    if (result != 0) {
        THROW_IE_EXCEPTION << "Failed to copy memory.";
    }
}

//------------------------------------------------------------------------------
HDDL2InferRequest::HDDL2InferRequest(const IE::InputsDataMap& networkInputs, const IE::OutputsDataMap& networkOutputs,
    const HddlUniteGraph::Ptr& loadedGraph, const HDDL2RemoteContext::Ptr& context, const HDDL2Config& config)
    : InferRequestInternal(networkInputs, networkOutputs),
      _loadedGraphPtr(loadedGraph),
      _context(context),
      _config(config),
      _logger(std::make_shared<Logger>("HDDL2InferRequest", config.logLevel(), consoleOutput())) {
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

void HDDL2InferRequest::Infer() {
    checkBlobs();
    InferImpl();
}

void HDDL2InferRequest::InferImpl() {
    InferAsync();
    WaitInferDone();
    GetResult();
}

void HDDL2InferRequest::InferAsync() {
    // TODO [Design flaw] InferData need to know if preprocessing required on creation.
    bool needUnitePreProcessing = false;

    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;
        const IE::Blob::Ptr inputBlobPtr = _inputs.find(inputName)->second;
        if (preProcessingRequired(networkInput.second, inputBlobPtr)) {
            needUnitePreProcessing = true;
        }
        if (inputBlobPtr->is<HDDL2RemoteBlob>()) {
            needUnitePreProcessing |= (inputBlobPtr->as<HDDL2RemoteBlob>()->getROIPtr() != nullptr);
        }
    }

    _inferDataPtr = std::make_shared<HddlUniteInferData>(needUnitePreProcessing, _context);

    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;
        const IE::InputInfo::Ptr inputDesc = networkInput.second;

        // TODO [Design flaw] At this point we have input blob or preprocessing blob specified inside _preProcData
        if (_preProcData.find(inputName) != _preProcData.end()) {
            const IE::PreProcessDataPtr preprocessData = _preProcData.find(inputName)->second;
            const IE::Blob::Ptr blobForPreprocessing = preprocessData->getRoiBlob();

            _inferDataPtr->prepareUniteInput(blobForPreprocessing, inputDesc);
        } else {
            const IE::Blob::Ptr inputBlobPtr = _inputs.find(inputName)->second;
            _inferDataPtr->prepareUniteInput(inputBlobPtr, inputDesc);
        }
    }

    for (const auto& networkOutput : _networkOutputs) {
        const std::string outputName = networkOutput.first;
        const IE::Blob::Ptr outputBlobPtr = _outputs.find(outputName)->second;

        _inferDataPtr->prepareUniteOutput(outputBlobPtr, networkOutput.second);
    }

    _loadedGraphPtr->InferAsync(_inferDataPtr);
}

void HDDL2InferRequest::WaitInferDone() { _inferDataPtr->waitInferDone(); }

void HDDL2InferRequest::GetResult() {
    if (_networkOutputs.size() != 1) {
        THROW_IE_EXCEPTION << "Only one output is supported!";
    }

    const std::string outputName = _networkOutputs.begin()->first;
    auto foundOutputBlob = _outputs.find(outputName);
    if (foundOutputBlob == _outputs.end()) {
        THROW_IE_EXCEPTION << "Error: output [" << outputName << "] is not provided.";
    }
    IE::Blob::Ptr outputBlobPtr = foundOutputBlob->second;

    const std::string outputUniteData = _inferDataPtr->getOutputData(outputName);

    const auto networkOutputPrecision = _networkOutputs.begin()->second->getPrecision();
    const auto blobOutputPrecision = outputBlobPtr->getTensorDesc().getPrecision();

    InferenceEngine::TensorDesc networkTensorDesc = _networkOutputs.begin()->second->getTensorDesc();
    InferenceEngine::TensorDesc outputBlobTensorDesc = outputBlobPtr->getTensorDesc();

    if (networkOutputPrecision == IE::Precision::FP32 || blobOutputPrecision == IE::Precision::FP32) {
        auto tempUniteOutputTensorDesc = networkTensorDesc;
        // MCM Compiler will work with FP16 instead of FP32, so we need to set output precision manually
        tempUniteOutputTensorDesc.setPrecision(IE::Precision::FP16);

        IE::Blob::Ptr tempFP16Blob = make_blob_with_precision(tempUniteOutputTensorDesc);
        tempFP16Blob->allocate();
        copyDataToBlob(tempFP16Blob, outputUniteData.data(), outputUniteData.size());
        if (tempUniteOutputTensorDesc.getPrecision() != blobOutputPrecision) {
            outputBlobPtr = utils::convertPrecision(tempFP16Blob, outputBlobTensorDesc.getPrecision());
        } else {
            outputBlobPtr = tempFP16Blob;
        }
    } else {
        if (outputUniteData.size() != outputBlobPtr->byteSize()) {
            THROW_IE_EXCEPTION << "Output size mismatch between HddlUnite and network expected output";
        }
        copyDataToBlob(outputBlobPtr, outputUniteData.data(), outputUniteData.size());
    }
    _outputs[outputName] = outputBlobPtr;
}

void vpu::HDDL2Plugin::HDDL2InferRequest::GetPerformanceCounts(
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const {
    UNUSED(perfMap);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

void HDDL2InferRequest::SetBlob(const char* name, const InferenceEngine::Blob::Ptr& data) {
    if (!data->is<HDDL2RemoteBlob>()) {
        InferenceEngine::InferRequestInternal::SetBlob(name, data);
        return;
    }

    IE_PROFILING_AUTO_SCOPE(SetBlob)
    if (name == nullptr) {
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }
    if (!data) THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
    const bool compoundBlobPassed = data->is<IE::CompoundBlob>();

    IE::InputInfo::Ptr foundInput;
    IE::DataPtr foundOutput;
    size_t dataSize = data->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        if (foundInput->getPrecision() != data->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding to user input precision";
        }

        const bool preProcRequired = preProcessingRequired(foundInput, data);
        if (compoundBlobPassed && !preProcRequired) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }

        if (preProcRequired) {
            if (_preProcData.find(name) == _preProcData.end()) {
                _preProcData.emplace(name, IE::CreatePreprocDataHelper());
            }
            _preProcData[name]->isApplicable(data, _inputs[name]);
            _preProcData[name]->setRoiBlob(data);
        } else {
            size_t inputSize = InferenceEngine::details::product(foundInput->getTensorDesc().getDims());
            if (dataSize != inputSize) {
                THROW_IE_EXCEPTION << "Input blob size is not equal network input size (" << dataSize
                                   << "!=" << inputSize << ").";
            }
            _inputs[name] = data;
        }
    } else {
        if (compoundBlobPassed) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }
        size_t outputSize = InferenceEngine::details::product(foundOutput->getDims());
        if (dataSize != outputSize) {
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size (" << dataSize
                               << "!=" << outputSize << ").";
        }
        if (foundOutput->getPrecision() != data->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding to user output precision";
        }
        _outputs[name] = data;
    }
}

void HDDL2InferRequest::checkBlobs() {
    for (auto const& input : _inputs) {
        if (!input.second->is<HDDL2RemoteBlob>()) checkBlob(input.second, input.first, true);
    }
    for (auto const& output : _outputs) {
        if (!output.second->is<HDDL2RemoteBlob>()) checkBlob(output.second, output.first, false);
    }
}
