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

/**
 * @brief HDDL2 Blob descriptor in term of HddlUnite BlobDesc
 */
class BlobDescriptor {
public:
    using Ptr = std::shared_ptr<BlobDescriptor>;

    BlobDescriptor(const BlobDescriptor&) = delete;
    BlobDescriptor(const BlobDescriptor&&) = delete;
    BlobDescriptor& operator=(const BlobDescriptor&) = delete;
    BlobDescriptor& operator=(const BlobDescriptor&&) = delete;

    explicit BlobDescriptor(const InferenceEngine::DataPtr& desc, const InferenceEngine::Blob::CPtr& blob,
        bool createRemoteMemoryDescriptor, bool isNeedAllocation, bool isOutput);
    virtual ~BlobDescriptor() = default;

    virtual HddlUnite::Inference::BlobDesc createUniteBlobDesc(
        const bool& isInput, const InferenceEngine::ColorFormat& colorFormat);
    virtual void initUniteBlobDesc(HddlUnite::Inference::BlobDesc&);
    virtual HddlUnite::Inference::NNInputDesc createNNDesc();

    std::shared_ptr<const InferenceEngine::ROI> getROIPtr() const { return _parsedBlobParamsPtr->getROIPtr(); }
    std::shared_ptr<const InferenceEngine::TensorDesc> getOriginalTensorDesc() const {
        return _parsedBlobParamsPtr->getOriginalTensorDesc();
    };

protected:
    const bool _createRemoteMemoryDescriptor;
    const bool _isNeedAllocation = true;

    InferenceEngine::Blob::CPtr _blobPtr = nullptr;
    std::shared_ptr<vpux::ParsedRemoteBlobParams> _parsedBlobParamsPtr = nullptr;

    InferenceEngine::DataPtr _desc = nullptr;

    virtual void setImageFormatToDesc(HddlUnite::Inference::BlobDesc& blobDesc);
    // TODO [Workaround] Find suitable approach for IE::NV12 & HddlUnite::NV12 handling
    /**
     * @brief Workaround to provide to HddlUnite one sequence of raw data
     * (NV12 Blob can contains two pointer to data which are not sequential)
     */
    void createRepackedNV12Blob(const InferenceEngine::Blob::CPtr& blobPtr);
    InferenceEngine::Blob::Ptr _repackedBlob;  //!< Repacked NV12 Blob if specified

    // TODO To be removed
    const bool _isOutput = true;
};

//------------------------------------------------------------------------------
class LocalBlobDescriptor : public BlobDescriptor {
public:
    explicit LocalBlobDescriptor(const InferenceEngine::DataPtr& desc, const InferenceEngine::Blob::CPtr& blob);
};

//------------------------------------------------------------------------------
class RemoteBlobDescriptor : public BlobDescriptor {
public:
    explicit RemoteBlobDescriptor(const InferenceEngine::DataPtr& desc, const InferenceEngine::Blob::CPtr& blob);
};
}  // namespace HDDL2Plugin
}  // namespace vpu
