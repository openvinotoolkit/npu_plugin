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

// TODO [Workaround] Until we will be able to reset input blobs and call createBlob for same name again
//  It useful for if user set NV12 blob, run inference, and after call set blob with BGR blob, not NV
//  This will require recreating of BlobDesc due to different color format / size;
void InferDataAdapter::createInferData() {
    _inferDataPtr = HddlUnite::Inference::makeInferData(_auxBlob, _workloadContext, maxRoiNum,
                                                        _networkDescription->getDeviceOutputsInfo().size());
    if (_inferDataPtr.get() == nullptr) {
        THROW_IE_EXCEPTION << "InferDataAdapter: Failed to create Unite inferData";
    }

    for (const auto& deviceInput : _networkDescription->getDeviceInputsInfo()) {
        const auto& inputName = deviceInput.first;
        const auto& blobDesc = deviceInput.second;
        checkDataNotNull(blobDesc);

        const bool isInput = true;
        std::shared_ptr<BlobDescriptorAdapter> blobDescriptorPtr(
                new BlobDescriptorAdapter(getBlobType(_haveRemoteContext), blobDesc, _graphColorFormat, isInput));

        _inputs[inputName] = blobDescriptorPtr;
    }

    for (const auto& networkOutput : _networkDescription->getDeviceOutputsInfo()) {
        const auto& outputName = networkOutput.first;
        const auto& blobDesc = networkOutput.second;
        checkDataNotNull(blobDesc);

        const bool isInput = false;
        std::shared_ptr<BlobDescriptorAdapter> blobDescriptorPtr(
                new BlobDescriptorAdapter(getBlobType(_haveRemoteContext), blobDesc, _graphColorFormat, isInput));

        const auto HDDLUniteBlobDesc = blobDescriptorPtr->createUniteBlobDesc(isInput);
        _inferDataPtr->createBlob(outputName, HDDLUniteBlobDesc, isInput);

        _outputs[outputName] = blobDescriptorPtr;
    }
}
//------------------------------------------------------------------------------
InferDataAdapter::InferDataAdapter(const vpux::NetworkDescription::CPtr& networkDescription,
                                   const HddlUnite::WorkloadContext::Ptr& workloadContext,
                                   const InferenceEngine::ColorFormat colorFormat)
        : _networkDescription(networkDescription),
          _workloadContext(workloadContext),
          _graphColorFormat(colorFormat),
          _haveRemoteContext(workloadContext != nullptr),
          _needUnitePreProcessing(true) {
    _auxBlob = {HddlUnite::Inference::AuxBlob::Type::TimeTaken};
    if (networkDescription == nullptr) {
        THROW_IE_EXCEPTION << "InferDataAdapter: NetworkDescription is null";
    }
    createInferData();
}

void InferDataAdapter::setPreprocessFlag(const bool preprocessingRequired) {
    _needUnitePreProcessing = preprocessingRequired;
}

static bool isInputBlobDescAlreadyCreated(const HddlUnite::Inference::InferData::Ptr& inferDataPtr,
                                          const std::string& inputBlobName) {
    const auto& inputBlobs = inferDataPtr->getInBlobs();
    auto result =
            std::find_if(inputBlobs.begin(), inputBlobs.end(),
                         [&inputBlobName](const std::pair<std::string, HddlUnite::Inference::InBlob::Ptr>& element) {
                             return element.first == inputBlobName;
                         });
    return result != inputBlobs.end();
}

void InferDataAdapter::prepareUniteInput(const InferenceEngine::Blob::CPtr& blob, const std::string& inputName) {
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "InferDataAdapter: Blob for input is null";
    }
    if (_inputs.find(inputName) == _inputs.end()) {
        THROW_IE_EXCEPTION << "InferDataAdapter: Failed to find BlobDesc for: " << inputName;
    }

    auto blobDescriptorPtr = _inputs.at(inputName);

    // Check that created blob description is suitable for input blob
    const auto blobDescSuitable = blobDescriptorPtr->isBlobDescSuitableForBlob(blob);
    if (!blobDescSuitable) {
        const auto& deviceInputInfo = _networkDescription->getDeviceInputsInfo().at(inputName);
        std::shared_ptr<BlobDescriptorAdapter> newBlobDescriptorPtr(
                new BlobDescriptorAdapter(blob, _graphColorFormat, deviceInputInfo));

        _inputs[inputName] = newBlobDescriptorPtr;
        blobDescriptorPtr = newBlobDescriptorPtr;

        // TODO [Worklaround] !!! Check if blob already exists. If so, well, create inferRequest again
        if (isInputBlobDescAlreadyCreated(_inferDataPtr, inputName)) {
            createInferData();
        }
    }

    const bool isInput = true;
    const auto& HDDLUniteBlobDesc = blobDescriptorPtr->createUniteBlobDesc(isInput);

    // Postponed blob creation
    if (!isInputBlobDescAlreadyCreated(_inferDataPtr, inputName)) {
        const auto success = _inferDataPtr->createBlob(inputName, HDDLUniteBlobDesc, isInput);
        if (!success) {
            THROW_IE_EXCEPTION << "InferDataAdapter: Error creating HDDLUnite Blob";
        }
        /** setPPFlag should be called after inferData->createBLob call but before updateBlob **/
        _inferDataPtr->setPPFlag(_needUnitePreProcessing);

        if ((_needUnitePreProcessing || blobDescriptorPtr->isROIPreprocessingRequired()) && _haveRemoteContext) {
            auto nnBlobDesc = blobDescriptorPtr->createNNDesc();
            _inferDataPtr->setNNInputDesc(nnBlobDesc);
        }
    }

    const auto updatedHDDLUniteBlobDesc = blobDescriptorPtr->updateUniteBlobDesc(blob);
    if (!_inferDataPtr->getInputBlob(inputName)->updateBlob(updatedHDDLUniteBlobDesc)) {
        THROW_IE_EXCEPTION << "InferDataAdapter: Error updating Unite Blob";
    }
    // Strides alignment is supported only for PP case
    // Hantro video-codec needs by it (only for KMB EVM B0)
    const size_t strideAlignment = 256;
    if (!(updatedHDDLUniteBlobDesc.m_widthStride % strideAlignment) &&
        (updatedHDDLUniteBlobDesc.m_resWidth % strideAlignment) && !_needUnitePreProcessing) {
        THROW_IE_EXCEPTION << "InferDataAdapter: strides alignment is supported only for preprocessing.";
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
