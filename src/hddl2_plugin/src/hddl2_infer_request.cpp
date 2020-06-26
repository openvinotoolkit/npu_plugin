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
#include <vpu/utils/ie_helpers.hpp>

#include "hddl2_remote_blob.h"
#include "ie_algorithm.hpp"
#include "ie_utils.hpp"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------

// TODO [Track number: S#21391]
// FIXME: does not work for batch != 1
static bool is2DTensor(const InferenceEngine::SizeVector& dims) {
    size_t ones = std::count(dims.begin(), dims.end(), 1);
    return (dims.size() - ones) == 1;
}

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
    IE_PROFILING_AUTO_SCOPE(InferAsync)
    // TODO [Design flaw] InferData need to know if preprocessing required on creation.
    bool needUnitePreProcessing = false;
    InferenceEngine::BlobMap updatedInputs;

    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;
        auto foundInputBlob = _inputs.find(inputName);
        if (foundInputBlob == _inputs.end()) {
            THROW_IE_EXCEPTION << "Error: input [" << inputName << "] is not provided.";
        }

        const IE::Blob::Ptr inputBlobPtr = foundInputBlob->second;
        if (preProcessingRequired(networkInput.second, inputBlobPtr)) {
            needUnitePreProcessing = true;
        }
        if (inputBlobPtr->is<HDDL2RemoteBlob>()) {
            needUnitePreProcessing |= (inputBlobPtr->as<HDDL2RemoteBlob>()->getROIPtr() != nullptr);
        }

        const auto deviceInputLayout = GetSupportedLayout();

        // TODO [Design flaw] If preprocessing is required, blob is stored inside _preprocData, not in _networkInputs.
        if (!needUnitePreProcessing) {
            updatedInputs[foundInputBlob->first] = prepareInputForInference(foundInputBlob->second, deviceInputLayout);
        } else {
            updatedInputs[foundInputBlob->first] = foundInputBlob->second;
        }
    }

    std::call_once(_onceFlagInferData, [&] {
        _inferDataPtr = std::make_shared<HddlUniteInferData>(needUnitePreProcessing, _context);
    });

    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;
        const IE::InputInfo::Ptr inputDesc = networkInput.second;

        // TODO [Design flaw] At this point we have input blob or preprocessing blob specified inside _preProcData
        if (_preProcData.find(inputName) != _preProcData.end()) {
            const IE::PreProcessDataPtr preprocessData = _preProcData.find(inputName)->second;
            const IE::Blob::Ptr blobForPreprocessing = preprocessData->getRoiBlob();

            _inferDataPtr->prepareUniteInput(blobForPreprocessing, inputDesc);
        } else {
            auto foundInputBlob = updatedInputs.find(inputName);
            if (foundInputBlob == updatedInputs.end()) {
                THROW_IE_EXCEPTION << "Error: input [" << inputName << "] is not provided.";
            }
            const IE::Blob::Ptr inputBlobPtr = foundInputBlob->second;
            _inferDataPtr->prepareUniteInput(inputBlobPtr, inputDesc);
        }
    }

    for (const auto& networkOutput : _networkOutputs) {
        const std::string outputName = networkOutput.first;
        auto foundOutputBlob = _outputs.find(outputName);
        if (foundOutputBlob == _outputs.end()) {
            THROW_IE_EXCEPTION << "Error: output [" << outputName << "] is not provided.";
        }
        const IE::Blob::Ptr outputBlobPtr = foundOutputBlob->second;

        _inferDataPtr->prepareUniteOutput(outputBlobPtr, networkOutput.second);
    }

    _loadedGraphPtr->InferAsync(_inferDataPtr);
}

void HDDL2InferRequest::WaitInferDone() {
    IE_PROFILING_AUTO_SCOPE(WaitInferDone)
    _inferDataPtr->waitInferDone();
}

static IE::Blob::Ptr reallocateBlobToLayout(const IE::Blob::Ptr& blob, const IE::Layout layout) {
    IE::Blob::Ptr newBlob =
        make_blob_with_precision({blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), layout});
    newBlob->allocate();
    vpu::copyBlob(blob, newBlob);
    return newBlob;
}

IE::Blob::Ptr HDDL2InferRequest::prepareInputForInference(
    const IE::Blob::Ptr& actualInput, const IE::Layout& expectedLayout) {
    IE_PROFILING_AUTO_SCOPE(prepareInputForInference);
    if (actualInput->getTensorDesc().getLayout() == IE::Layout::NHWC ||
        /** Currently we ignore information of what type of remote blob we are using **/
        actualInput->is<IE::RemoteBlob>() ||
        /** Repacking for NV12 Blob is not required, compound blob should be handled other way **/
        // TODO Add repacking for compound blob case
        actualInput->is<IE::NV12Blob>() || actualInput->is<ie::CompoundBlob>()) {
        return actualInput;
    }

    _logger->info("Input blob is inconsistent with network input. Need to do re-layout.");
    IE::Blob::Ptr inputForInference;

    if (is2DTensor(actualInput->getTensorDesc().getDims())) {
        auto tensorDims = actualInput->getTensorDesc().getDims();
        for (size_t dimInd = actualInput->getTensorDesc().getDims().size(); dimInd < 4; dimInd++) {
            tensorDims.push_back(1);
        }
        IE::TensorDesc TensorDesc = {actualInput->getTensorDesc().getPrecision(), tensorDims, expectedLayout};
        inputForInference = make_blob_with_precision(TensorDesc);
        inputForInference->allocate();

        ie_memcpy(
            inputForInference->buffer(), inputForInference->byteSize(), actualInput->buffer(), actualInput->byteSize());
    } else {
        inputForInference = reallocateBlobToLayout(actualInput, expectedLayout);
    }

    return inputForInference;
}

void HDDL2InferRequest::GetResult() {
    IE_PROFILING_AUTO_SCOPE(GetResult)
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
        if (networkOutputPrecision == IE::Precision::U8 || blobOutputPrecision == IE::Precision::U8) {
            THROW_IE_EXCEPTION << "Error: output precision conversion from " << networkOutputPrecision << " to "
                               << blobOutputPrecision << " is not supported.";
        }
        auto tempUniteOutputTensorDesc = networkTensorDesc;
        // MCM Compiler will work with FP16 instead of FP32, so we need to set output precision manually
        tempUniteOutputTensorDesc.setPrecision(IE::Precision::FP16);
        if (outputBlobPtr->getTensorDesc().getDims().size() == 4) {
            tempUniteOutputTensorDesc.setLayout(IE::Layout::NHWC);
        }

        IE::Blob::Ptr tempFP16Blob = make_blob_with_precision(tempUniteOutputTensorDesc);
        tempFP16Blob->allocate();
        copyDataToBlob(tempFP16Blob, outputUniteData.data(), outputUniteData.size());
        if (tempUniteOutputTensorDesc.getPrecision() != blobOutputPrecision) {
            outputBlobPtr = utils::convertPrecision(tempFP16Blob, outputBlobTensorDesc.getPrecision());
        } else {
            outputBlobPtr = tempFP16Blob;
        }
    } else {
        if (networkOutputPrecision == IE::Precision::U8 && blobOutputPrecision == IE::Precision::FP16) {
            THROW_IE_EXCEPTION << "Error: output precision conversion from " << networkOutputPrecision << " to "
                               << blobOutputPrecision << " is not supported.";
        }
        if (outputUniteData.size() != outputBlobPtr->byteSize()) {
            THROW_IE_EXCEPTION << "Output size mismatch between HddlUnite and network expected output";
        }
        copyDataToBlob(outputBlobPtr, outputUniteData.data(), outputUniteData.size());
        if (!is2DTensor(outputBlobPtr->getTensorDesc().getDims())) {
            outputBlobPtr->getTensorDesc().setLayout(IE::Layout::NHWC);
        }
    }
    if (is2DTensor(outputBlobPtr->getTensorDesc().getDims())) {
        _outputs[outputName] = outputBlobPtr;
    } else {
        if (outputBlobPtr->getTensorDesc().getLayout() != networkTensorDesc.getLayout()) {
            _outputs[outputName] = reallocateBlobToLayout(outputBlobPtr, networkTensorDesc.getLayout());
        } else {
            _outputs[outputName] = outputBlobPtr;
        }
    }
}

void vpu::HDDL2Plugin::HDDL2InferRequest::GetPerformanceCounts(
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const {
    if (_config.performance_counting()) {
        _inferDataPtr->getHddlUnitePerfCounters(perfMap);
    }
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
