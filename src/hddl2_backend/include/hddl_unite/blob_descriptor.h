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

#pragma once
// System
#include <memory>
// IE
#include "ie_blob.h"
#include "ie_data.h"
#include "ie_preprocess.hpp"
// Plugin
#include <vpux_params.hpp>
// Low-level
#include "Inference.h"

namespace vpu {
namespace HDDL2Plugin {

/** @brief Blobs for image workload have different was of creation */
enum class BlobDescType { VideoWorkload = 1, ImageWorkload = 2 };

/**  * @brief HDDL2 Blob descriptor in term of HddlUnite BlobDesc */
class BlobDescriptorAdapter final {
public:
    BlobDescriptorAdapter() = delete;
    BlobDescriptorAdapter(const BlobDescriptorAdapter&) = delete;
    BlobDescriptorAdapter(const BlobDescriptorAdapter&&) = delete;
    BlobDescriptorAdapter& operator=(const BlobDescriptorAdapter&) = delete;
    BlobDescriptorAdapter& operator=(const BlobDescriptorAdapter&&) = delete;
    explicit BlobDescriptorAdapter(
        BlobDescType typeOfBlob, const InferenceEngine::DataPtr& desc, const InferenceEngine::Blob::CPtr& blob);
    virtual ~BlobDescriptorAdapter() = default;

public:
    /** @brief Create BlobDesc in terms of HddlUnite. Will have information only  */
    HddlUnite::Inference::BlobDesc createUniteBlobDesc(
        const bool& isInput, const InferenceEngine::ColorFormat& colorFormat);
    void initUniteBlobDesc(HddlUnite::Inference::BlobDesc&);
    HddlUnite::Inference::NNInputDesc createNNDesc();

    std::shared_ptr<const InferenceEngine::ROI> getROIPtr() const { return _parsedBlobParamsPtr->getROIPtr(); }
    std::shared_ptr<const InferenceEngine::TensorDesc> getOriginalTensorDesc() const {
        return _parsedBlobParamsPtr->getOriginalTensorDesc();
    };

protected:
    const BlobDescType _blobType;
    bool _isNeedAllocation;

    InferenceEngine::Blob::CPtr _blobPtr = nullptr;
    std::shared_ptr<vpux::ParsedRemoteBlobParams> _parsedBlobParamsPtr = nullptr;

    InferenceEngine::DataPtr _desc = nullptr;

    void setImageFormatToDesc(HddlUnite::Inference::BlobDesc& blobDesc);
    // TODO [Workaround] Find suitable approach for IE::NV12 & HddlUnite::NV12 handling
    /**
     * @brief Workaround to provide to HddlUnite one sequence of raw data
     * (NV12 Blob can contains two pointer to data which are not sequential)
     */
    void createRepackedNV12Blob(const InferenceEngine::Blob::CPtr& blobPtr);
    InferenceEngine::Blob::Ptr _repackedBlob;  //!< Repacked NV12 Blob if specified

    // TODO To be removed
    bool _isOutput = true;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
