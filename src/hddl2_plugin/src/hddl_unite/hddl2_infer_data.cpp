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
HddlUniteInferData::HddlUniteInferData(const bool& needPreProcessing, const HDDL2RemoteContext::Ptr& remoteContext)
    : _haveRemoteContext(remoteContext != nullptr), _needPreProcessing(needPreProcessing) {
    _auxBlob = {HddlUnite::Inference::AuxBlob::Type::TimeTaken};

    HddlUnite::WorkloadContext::Ptr workloadContext = nullptr;
    if (_haveRemoteContext) {
        workloadContext = remoteContext->getHddlUniteWorkloadContext();
        if (workloadContext == nullptr) {
            THROW_IE_EXCEPTION << "Workload context is null!";
        }
    }

    _inferDataPtr = HddlUnite::Inference::makeInferData(_auxBlob, workloadContext, needPreProcessing);

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
    auto blobDesc = blobDescriptorPtr->createUniteBlobDesc();
    const bool isInput = true;
    _inferDataPtr->createBlob(name, blobDesc, isInput);

    blobDescriptorPtr->initUniteBlobDesc(blobDesc);
    _inferDataPtr->getInputBlob(name)->updateBlob(blobDesc);
    _inputs[name] = blobDescriptorPtr;

    if (_needPreProcessing && _haveRemoteContext) {
        auto nnBlobDesc = blobDescriptorPtr->createNNDesc();
        _inferDataPtr->setNNInputDesc(nnBlobDesc);
    }
}

void HddlUniteInferData::prepareUniteOutput(const IE::Blob::Ptr& blob, const IE::DataPtr& desc) {
    checkData(desc);

    const std::string name = desc->getName();

    BlobDescriptor::Ptr blobDescriptorPtr;
    if (_haveRemoteContext) {
        blobDescriptorPtr = std::make_shared<RemoteBlobDescriptor>(desc, blob);
    } else {
        blobDescriptorPtr = std::make_shared<LocalBlobDescriptor>(desc, blob);
    }

    const bool isInput = false;
    _inferDataPtr->createBlob(name, blobDescriptorPtr->createUniteBlobDesc(), isInput);

    _outputs[name] = blobDescriptorPtr;
}

std::string HddlUniteInferData::getOutputData(const std::string& outputName) {
    const auto outputBlob = _inferDataPtr->getOutputBlob(outputName);
    if (outputBlob == nullptr) {
        THROW_IE_EXCEPTION << "Failed to get blob from hddlUnite!";
    }
    return outputBlob->getData();
}
