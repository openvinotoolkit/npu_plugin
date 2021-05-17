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

namespace vpux {
namespace hddl2 {

/** @brief Blobs for image workload have different ways of creation */
enum class BlobDescType { VideoWorkload = 1, ImageWorkload = 2 };

/** @brief If we have workload context, we are working with VideoWorkload */
inline BlobDescType getBlobType(const bool haveRemoteContext) {
    return haveRemoteContext ? BlobDescType::VideoWorkload : BlobDescType::ImageWorkload;
}

//------------------------------------------------------------------------------
/** @brief Allocation information for BlobDesc */
struct AllocationInfo {
    explicit AllocationInfo(const BlobDescType typeOfBlob, const InferenceEngine::DataPtr& blobDesc,
                            const InferenceEngine::ColorFormat& graphColorFormat, const bool isInput);
    explicit AllocationInfo(const InferenceEngine::Blob::CPtr& blob,
                            const InferenceEngine::ColorFormat& graphColorFormat);
    bool operator==(const AllocationInfo& rhs) const;
    bool operator!=(const AllocationInfo& rhs) const;

    // Size
    const HddlUnite::Inference::Precision precision;
    const uint64_t dataSize;
    // Flags
    const bool isRemoteMemory;
    const bool isNeedAllocation;
    const bool isCompound;
    // Expected graph input color format
    const InferenceEngine::ColorFormat nnInputColorFormat;
};

/** @brief Source and meta information for BlobDesc */
struct SourceInfo {
    union {
        int remoteMemoryFd;
        void* localMemoryPtr;
    };
    HddlUnite::Inference::FourCC blobColorFormat;
    //    TODO Add resize algorithm information
    int32_t resWidth;
    int32_t resHeight;
    int32_t widthStride;
    int32_t planeStride;
    std::vector<HddlUnite::Inference::Rectangle> roiRectangles;
};

/** @brief Info for intermediate buffer for preprocessing result
 *  @details AllocationInfo should be created again if blob require preprocessing. NNInputInfo always the same */
struct NNInputInfo {
    NNInputInfo(const BlobDescType typeOfBlob, const InferenceEngine::DataPtr& blobDesc);
    // Size
    const HddlUnite::Inference::Precision precision;
    const uint64_t dataSize;
    // Flags
    const bool isRemoteMemory;
    const bool isNeedAllocation;
    // TODO For multiple roi case?
    const uint32_t batch;
};

//------------------------------------------------------------------------------
/**  @brief HDDL2 Blob descriptor in term of HddlUnite BlobDesc */
class BlobDescriptorAdapter final {
public:
    using Ptr = std::shared_ptr<BlobDescriptorAdapter>;

    BlobDescriptorAdapter() = delete;
    BlobDescriptorAdapter(const BlobDescriptorAdapter&) = delete;
    BlobDescriptorAdapter(const BlobDescriptorAdapter&&) = delete;
    BlobDescriptorAdapter& operator=(const BlobDescriptorAdapter&) = delete;
    BlobDescriptorAdapter& operator=(const BlobDescriptorAdapter&&) = delete;
    explicit BlobDescriptorAdapter(BlobDescType typeOfBlob, const InferenceEngine::DataPtr& blobDesc,
                                   const InferenceEngine::ColorFormat& graphColorFormat, const bool isInput);
    /** @brief If blob allocation data is different from networkDesc, recreation of blob desc required */
    explicit BlobDescriptorAdapter(const InferenceEngine::Blob::CPtr& blobPtr,
                                   const InferenceEngine::ColorFormat& graphColorFormat,
                                   const InferenceEngine::DataPtr& blobDesc);
    virtual ~BlobDescriptorAdapter() = default;

public:
    /** @brief Create BlobDesc in terms of HddlUnite. Will have information only about size*/
    const HddlUnite::Inference::BlobDesc& createUniteBlobDesc(const bool isInput);

    /** @brief Update HddlUnite::BlobDesc with source data, format and ROI
     * @return BlobDesc updated with information from blob */
    const HddlUnite::Inference::BlobDesc& updateUniteBlobDesc(
            const InferenceEngine::Blob::CPtr& blob,
            const InferenceEngine::ColorFormat colorFormat = InferenceEngine::ColorFormat::BGR);

    /** @brief Create intermediate buffer for preprocessing result / NN input. Will have information only about size */
    HddlUnite::Inference::NNInputDesc createNNDesc() const;

    /** @brief Allocation information may be suitable for blob.
     * @details Blob for preprocessing might have different size, so recreation of BlobDesc is required */
    bool isBlobDescSuitableForBlob(const InferenceEngine::Blob::CPtr& blob) const;

    bool isROIPreprocessingRequired() const;

private:
    const BlobDescType _blobType;
    const AllocationInfo _allocationInfo;
    SourceInfo _sourceInfo;
    const NNInputInfo _nnInputInfo;
    HddlUnite::Inference::BlobDesc _hddlUniteBlobDesc;

private:
    /** Fill SourceInfo stuct with all frame related information **/
    void prepareImageFormatInfo(const InferenceEngine::Blob::CPtr& blobPtr,
                                const InferenceEngine::ColorFormat colorFormat);

    /** Prepare ROI information **/
    void getRect(const InferenceEngine::Blob::CPtr& blobPtr, const vpux::ParsedRemoteBlobParams::CPtr& blobParams);
};

}  // namespace hddl2
}  // namespace vpux
