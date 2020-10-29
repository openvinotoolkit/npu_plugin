//
// Copyright 2020 Intel Corporation.
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
#include <memory>
#include <string>
// IE
#include "ie_compound_blob.h"
#include "ie_preprocess_data.hpp"
// Plugin
#include "hddl_unite/infer_data_adapter.h"
// Low-level
#include "Inference.h"

namespace vpu {
namespace HDDL2Plugin {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static void checkDataNotNull(const IE::DataPtr& desc) {
    if (!desc) {
        THROW_IE_EXCEPTION << "Data is null";
    }
}

//------------------------------------------------------------------------------
InferDataAdapter::InferDataAdapter(const HddlUnite::WorkloadContext::Ptr& workloadContext,
    const InferenceEngine::ColorFormat colorFormat, const size_t numOutputs)
    : _workloadContext(workloadContext),
      _haveRemoteContext(workloadContext != nullptr),
      _needUnitePreProcessing(true),
      _graphColorFormat(colorFormat) {
    _auxBlob = {HddlUnite::Inference::AuxBlob::Type::TimeTaken};

    _inferDataPtr = HddlUnite::Inference::makeInferData(_auxBlob, _workloadContext, maxRoiNum, numOutputs);

    if (_inferDataPtr.get() == nullptr) {
        THROW_IE_EXCEPTION << "InferDataAdapter: "
                           << "Failed to create Unite inferData";
    }
}

void InferDataAdapter::setPreprocessFlag(const bool preprocessingRequired) {
    _needUnitePreProcessing = preprocessingRequired;
    if (_needUnitePreProcessing != preprocessingRequired) {
        _inferDataPtr->setPPFlag(_needUnitePreProcessing);
    }
}

void InferDataAdapter::prepareUniteInput(const IE::Blob::CPtr& blob, const IE::DataPtr& desc) {
    checkDataNotNull(desc);
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "Blob for input is null";
    }
    const std::string name = desc->getName();

    BlobDescriptor::Ptr blobDescriptorPtr;
    if (_haveRemoteContext) {
        blobDescriptorPtr = std::make_shared<RemoteBlobDescriptor>(desc, blob);
    } else {
        blobDescriptorPtr = std::make_shared<LocalBlobDescriptor>(desc, blob);
    }

    const bool isInput = true;
    auto blobDesc = blobDescriptorPtr->createUniteBlobDesc(isInput, _graphColorFormat);
    std::call_once(_onceFlagInputAllocations, [&] {
        if (!_inferDataPtr->createBlob(name, blobDesc, isInput)) {
            THROW_IE_EXCEPTION << "Error creating Unite Blob";
        }

        if ((_needUnitePreProcessing || blobDescriptorPtr->getROIPtr() != nullptr) && _haveRemoteContext) {
            auto nnBlobDesc = blobDescriptorPtr->createNNDesc();
            _inferDataPtr->setNNInputDesc(nnBlobDesc);
        }
    });

    // needUnitePreProcessing on Unite includes existence of roi currently
    _inferDataPtr->setPPFlag(_needUnitePreProcessing);

    blobDescriptorPtr->initUniteBlobDesc(blobDesc);
    if (!_inferDataPtr->getInputBlob(name)->updateBlob(blobDesc)) {
        THROW_IE_EXCEPTION << "Error updating Unite Blob";
    }
    _inputs[name] = blobDescriptorPtr;
}

void InferDataAdapter::prepareUniteOutput(const IE::DataPtr& desc) {
    checkDataNotNull(desc);
    const auto name = desc->getName();

    auto findIt = std::find(_onceFlagOutputAllocations.begin(), _onceFlagOutputAllocations.end(), name);
    if (findIt == _onceFlagOutputAllocations.end()) {
        _onceFlagOutputAllocations.push_back(name);

        BlobDescriptor::Ptr blobDescriptorPtr;
        if (_haveRemoteContext) {
            blobDescriptorPtr = std::make_shared<RemoteBlobDescriptor>(desc, nullptr);
        } else {
            blobDescriptorPtr = std::make_shared<LocalBlobDescriptor>(desc, nullptr);
        }

        const bool isInput = false;
        _inferDataPtr->createBlob(name, blobDescriptorPtr->createUniteBlobDesc(isInput, _graphColorFormat), isInput);

        _outputs[name] = blobDescriptorPtr;
    }
}

std::string InferDataAdapter::getOutputData(const std::string& outputName) {
    // TODO send roiIndex (second parameter)
    auto outputData = _inferDataPtr->getOutputData(outputName);
    if (outputData.empty()) {
        THROW_IE_EXCEPTION << "Failed to get blob from hddlUnite!";
    }
    _profileData = _inferDataPtr->getProfileData();
    return outputData;
}

void InferDataAdapter::waitInferDone() const {
    auto status = _inferDataPtr->waitInferDone(_asyncInferenceWaitTimeoutMs);
    if (status != HDDL_OK) {
        THROW_IE_EXCEPTION << "Failed to wait for inference result with error: " << status;
    }
}

std::map<std::string, IE::InferenceEngineProfileInfo> InferDataAdapter::getHDDLUnitePerfCounters() const {
    std::map<std::string, IE::InferenceEngineProfileInfo> perfCounts;
    IE::InferenceEngineProfileInfo info;
    info.status = IE::InferenceEngineProfileInfo::EXECUTED;
    info.cpu_uSec = 0;
    info.execution_index = 0;
    info.realTime_uSec = 0;

    info.realTime_uSec = static_cast<long long>(_profileData.infer.time);
    perfCounts["Total scoring time"] = info;

    info.realTime_uSec = static_cast<long long>(_profileData.nn.time);
    perfCounts["Total scoring time on inference"] = info;

    info.realTime_uSec = static_cast<long long>(_profileData.pp.time);
    perfCounts["Total scoring time on preprocess"] = info;
    return perfCounts;
}

}  // namespace HDDL2Plugin
}  // namespace vpu
