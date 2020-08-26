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
#include <ie_itt.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <vpu/utils/ie_helpers.hpp>

#include "hddl2_executor.h"
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

static IE::Blob::Ptr allocateLocalBlob(const IE::TensorDesc& tensorDesc) {
    checkNetworkPrecision(tensorDesc.getPrecision());

    IE::Blob::Ptr blob = make_blob_with_precision(tensorDesc);
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "InputBlob is nullptr.";
    }
    blob->allocate();
    return blob;
}

//------------------------------------------------------------------------------
HDDL2InferRequest::HDDL2InferRequest(const InferenceEngine::InputsDataMap& networkInputs,
    const InferenceEngine::OutputsDataMap& networkOutputs, const vpux::Executor::Ptr& executor,
    const vpu::HDDL2Config& config)
    : InferRequestInternal(networkInputs, networkOutputs),
      _executorPtr(executor),
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
    GetResult();
}

/**
 * @brief Create map with preProcessing info and move all preProcessing blobs to inputs BlobMap
 * @param[in/out] inputs Map with NN blobs. PP blobs should be placed instead for some inputs.
 * @param[in] networkInputs Contains information of pre-processing, which should be done
 * @param[in] preProcData Container with blobs, which should be preprocessed
 * @return Map with preprocess information
 */
vpux::PreprocMap HDDL2InferRequest::preparePreProcessing(InferenceEngine::BlobMap& inputs,
    const InferenceEngine::InputsDataMap& networkInputs,
    const std::map<std::string, InferenceEngine::PreProcessDataPtr>& preProcData) {
    vpux::PreprocMap preProcMap;
    for (auto& input : networkInputs) {
        const std::string inputName = input.second->name();
        const auto& preProcDataIt = preProcData.find(inputName);
        if (preProcDataIt != preProcData.end()) {
            const IE::Blob::Ptr& blobForPreProcessing = preProcDataIt->second->getRoiBlob();
            if (preProcessingRequired(input.second, blobForPreProcessing)) {
                IE::Blob::Ptr blobForPreProc = preProcDataIt->second->getRoiBlob();
                /// If pre-processing required, we need use PP blobs instead of NN for inputs
                inputs.at(inputName) = blobForPreProcessing;
                preProcMap.emplace(input.first, input.second->getPreProcess());
            }
        }
    }
    return preProcMap;
}

void HDDL2InferRequest::InferAsync() {
    // TODO [Track number: S#36866]
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "InferAsync");
    const auto preProcMap = preparePreProcessing(_inputs, _networkInputs, _preProcData);
    _executorPtr->push(_inputs, preProcMap);
}

void HDDL2InferRequest::GetResult() {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "GetResult");
    _executorPtr->pull(_outputs);
}

void vpu::HDDL2Plugin::HDDL2InferRequest::GetPerformanceCounts(
    std::map<std::string, IE::InferenceEngineProfileInfo>& perfMap) const {
    if (_config.performance_counting()) {
        perfMap = _executorPtr->getLayerStatistics();
    }
}

void HDDL2InferRequest::SetBlob(const char* name, const IE::Blob::Ptr& data) {
    if (!data->is<HDDL2RemoteBlob>()) {
        IE::InferRequestInternal::SetBlob(name, data);
        return;
    }

    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "SetBlob");
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
            size_t inputSize = IE::details::product(foundInput->getTensorDesc().getDims());
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
        size_t outputSize = IE::details::product(foundOutput->getDims());
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
