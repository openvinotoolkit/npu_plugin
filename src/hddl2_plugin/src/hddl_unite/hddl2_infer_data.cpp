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

#include <Inference.h>
#include <hddl_unite/hddl2_infer_data.h>
#include <ie_compound_blob.h>

#include <ie_preprocess_data.hpp>
#include <memory>
#include <string>

#include "hddl2_remote_blob.h"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static void checkData(const IE::DataPtr& desc) {
    if (!desc) {
        THROW_IE_EXCEPTION << "Data is null";
    }
}

//------------------------------------------------------------------------------
HddlUniteInferData::HddlUniteInferData(const bool& needUnitePreProcessing, const HDDL2RemoteContext::Ptr& remoteContext,
    const IE::ColorFormat& colorFormat)
    : _haveRemoteContext(remoteContext != nullptr),
      _needUnitePreProcessing(needUnitePreProcessing),
      _graphColorFormat(colorFormat) {
    _auxBlob = {HddlUnite::Inference::AuxBlob::Type::TimeTaken};

    if (_haveRemoteContext) {
        _workloadContext = remoteContext->getHddlUniteWorkloadContext();
        if (_workloadContext == nullptr) {
            THROW_IE_EXCEPTION << "Workload context is null!";
        }
    }

    _inferDataPtr = HddlUnite::Inference::makeInferData(_auxBlob, _workloadContext);

    if (_inferDataPtr.get() == nullptr) {
        THROW_IE_EXCEPTION << "Failed to create Unite inferData";
    }
}

void HddlUniteInferData::prepareUniteInput(const IE::Blob::Ptr& blob, const IE::InputInfo::Ptr& info) {
    if (!info) {
        THROW_IE_EXCEPTION << "Input blob info is null";
    }
    checkData(info->getInputData());

    const std::string name = info->getInputData()->getName();
    IE::DataPtr desc = info->getInputData();

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

void HddlUniteInferData::prepareUniteOutput(const IE::Blob::Ptr& blob, const IE::DataPtr& desc) {
    std::call_once(_onceFlagOutputAllocations, [&] {
        checkData(desc);

        const std::string name = desc->getName();

        BlobDescriptor::Ptr blobDescriptorPtr;
        if (_haveRemoteContext) {
            blobDescriptorPtr = std::make_shared<RemoteBlobDescriptor>(desc, blob);
        } else {
            blobDescriptorPtr = std::make_shared<LocalBlobDescriptor>(desc, blob);
        }

        const bool isInput = false;
        _inferDataPtr->createBlob(name, blobDescriptorPtr->createUniteBlobDesc(isInput, _graphColorFormat), isInput);

        _outputs[name] = blobDescriptorPtr;
    });
}

std::string HddlUniteInferData::getOutputData(const std::string& outputName) {
    // TODO: send roiIndex (second parameter)
    auto outputData = _inferDataPtr->getOutputData(outputName);
    if (outputData.empty()) {
        THROW_IE_EXCEPTION << "Failed to get blob from hddlUnite!";
    }
    _profileData = _inferDataPtr->getProfileData();
    return outputData;
}

void HddlUniteInferData::waitInferDone() const {
    auto status = _inferDataPtr->waitInferDone(_asyncInferenceWaitTimeoutMs);
    if (status != HDDL_OK) {
        THROW_IE_EXCEPTION << "Failed to wait for inference result with error: " << status;
    }
}

void HddlUniteInferData::getHddlUnitePerfCounters(
    std::map<std::string, IE::InferenceEngineProfileInfo>& retPerfCounters) {
    IE::InferenceEngineProfileInfo info;
    info.status = IE::InferenceEngineProfileInfo::EXECUTED;
    info.cpu_uSec = 0;
    info.execution_index = 0;
    info.realTime_uSec = 0;

    info.realTime_uSec = _profileData.infer.time;
    IE_ASSERT(info.realTime_uSec != 0);
    retPerfCounters["Total scoring time"] = info;

    info.realTime_uSec = _profileData.nn.time;
    IE_ASSERT(info.realTime_uSec != 0);
    retPerfCounters["Total scoring time on inference"] = info;

    info.realTime_uSec = _profileData.pp.time;
    retPerfCounters["Total scoring time on preprocess"] = info;
}
