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
#include <memory>
#include "Inference.h"
#include "ie_blob.h"
#include "ie_preprocess.hpp"

namespace vpu {
namespace HDDL2Plugin {

/**
 * @brief HDDL2 Blob descriptor in term of HddlUnite BlobDesc
 */
class BlobDescriptor {
public:
    using Ptr = std::shared_ptr<BlobDescriptor>;

    explicit BlobDescriptor(
        const bool& isInput, const InferenceEngine::DataPtr& desc, const InferenceEngine::Blob::Ptr& blob);
    virtual ~BlobDescriptor() = default;

    virtual void setPreProcessing(const InferenceEngine::PreProcessInfo& preProcess);
    virtual HddlUnite::Inference::BlobDesc create();
    virtual HddlUnite::Inference::BlobDesc init() = 0;

protected:
    bool _isInput;
    bool _isRemoteMemory;
    bool _isNeedAllocation;

    InferenceEngine::Blob::Ptr _blobPtr = nullptr;
    std::shared_ptr<InferenceEngine::PreProcessInfo> _preProcessPtr = nullptr;

    // FIXME Do we really need this?
    InferenceEngine::DataPtr _desc = nullptr;

    HddlUnite::Inference::BlobDesc _blobDesc;

    virtual void setBlobFormat();
    // TODO [Workaround] Find suitable approach for IE::NV12 & HddlUnite::NV12 handling
    /**
     * @brief Workaround to provide to HddlUnite one sequence of raw data
     * (NV12 Blob can contains two pointer to data which are not sequential)
     */
    void createRepackedNV12Blob(const InferenceEngine::Blob::Ptr& blobPtr);
    InferenceEngine::Blob::Ptr _repackedBlob;  //!< Repacked NV12 Blob if specified
};

//------------------------------------------------------------------------------
class LocalBlobDescriptor : public BlobDescriptor {
public:
    explicit LocalBlobDescriptor(
        const bool& isInput, const InferenceEngine::DataPtr& desc, const InferenceEngine::Blob::Ptr& blob);

    HddlUnite::Inference::BlobDesc init() override;
};

//------------------------------------------------------------------------------
class RemoteBlobDescriptor : public BlobDescriptor {
public:
    explicit RemoteBlobDescriptor(
        const bool& isInput, const InferenceEngine::DataPtr& desc, const InferenceEngine::Blob::Ptr& blob);

    HddlUnite::Inference::BlobDesc init() override;
};
}  // namespace HDDL2Plugin
}  // namespace vpu
