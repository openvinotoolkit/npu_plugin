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

// IE
#include <ie_compound_blob.h>
#include <ie_memcpy.h>

#include <ie_algorithm.hpp>
// Plugin
#include "blob_descriptor_adapter.h"
#include "converters.h"
#include "hddl2_helper.h"

using namespace vpu::HDDL2Plugin;
namespace vpu {
namespace HDDL2Plugin {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static void checkBlobIsValid(const IE::Blob::CPtr& blob) {
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "Blob is null";
    }
    if (blob->size() == 0) {
        THROW_IE_EXCEPTION << "Blob is empty";
    }
}

static void checkBlobCompatibility(const IE::Blob::CPtr& blob, const BlobDescType blobType) {
    checkBlobIsValid(blob);
    if (blob->is<IE::RemoteBlob>()) {
        if (blobType == BlobDescType::ImageWorkload) {
            THROW_IE_EXCEPTION << "Remote blob not supported for ImageWorkload! Context required.";
        }
    } else {
        if (blobType == BlobDescType::VideoWorkload) {
            THROW_IE_EXCEPTION << "Local blob is not supported for VideoWorkload! Please use only remote blobs!";
        }
    }
    if (blob->is<IE::RemoteBlob>() || blob->is<IE::MemoryBlob>() || blob->is<IE::NV12Blob>()) {
        return;
    }
    if (blob->is<IE::CompoundBlob>()) {
        THROW_IE_EXCEPTION << "CompoundBlob is not supported";
    }
    THROW_IE_EXCEPTION << "Blob type is unexpected";
}

static void checkDataIsValid(const IE::DataPtr& data) {
    if (data == nullptr) {
        THROW_IE_EXCEPTION << "Blob descriptor is null";
    }
}

static IE::SizeVector getNV12ImageDims(const IE::Blob::CPtr& blobPtr) {
    if (blobPtr->is<IE::NV12Blob>()) {
        auto nv12Ptr = blobPtr->as<IE::NV12Blob>();
        if (nv12Ptr == nullptr) {
            THROW_IE_EXCEPTION << "Failed to cast nv12 blob.";
        }
        auto yPlaneBlob = nv12Ptr->y();
        return yPlaneBlob->getTensorDesc().getDims();
    } else if (blobPtr->is<IE::RemoteBlob>()) {
        return blobPtr->getTensorDesc().getDims();
    }
    THROW_IE_EXCEPTION << "Unsupported blob format with NV12 Data";
}

using matchColorFormats_t = std::unordered_map<int, HddlUnite::Inference::FourCC>;
static HddlUnite::Inference::FourCC covertColorFormat(const IE::ColorFormat colorFormat) {
    static const matchColorFormats_t matchColorFormats = {
        {static_cast<int>(IE::ColorFormat::BGR), HddlUnite::Inference::FourCC::BGR},
        {static_cast<int>(IE::ColorFormat::RGB), HddlUnite::Inference::FourCC::RGB}};

    auto format = matchColorFormats.find(colorFormat);
    if (format == matchColorFormats.end()) {
        throw std::logic_error("Color format is not valid.");
    }

    return format->second;
}

static size_t getSizeFromTensor(const IE::TensorDesc& tensorDesc) {
    if (tensorDesc.getLayout() == IE::Layout::SCALAR) {
        return 1;
    }
    const auto& dims = tensorDesc.getDims();
    const auto elementSize = tensorDesc.getPrecision().size();
    const size_t size =
        elementSize * std::accumulate(std::begin(dims), std::end(dims), (size_t)1, std::multiplies<size_t>());
    return size;
}

static bool isBlobContainsNV12Data(
    const IE::Blob::CPtr& blobPtr, const std::shared_ptr<vpux::ParsedRemoteBlobParams>& blobParams) {
    if (blobPtr == nullptr || blobParams == nullptr) {
        THROW_IE_EXCEPTION << "Check for NV12 failed, variables is null!";
    }
    if (blobPtr->is<IE::NV12Blob>() || blobParams->getColorFormat() == IE::NV12) {
        return true;
    }
    return false;
}

static inline bool isRemoteBlob(const InferenceEngine::Blob::CPtr& blob) { return blob && blob->is<IE::RemoteBlob>(); }

//------------------------------------------------------------------------------
AllocationInfo::AllocationInfo(const BlobDescType typeOfBlob, const IE::DataPtr& blobDesc,
    const IE::ColorFormat& graphColorFormat, const bool isInput)
    : precision(Unite::convertFromIEPrecision(blobDesc->getPrecision())),
      dataSize(getSizeFromTensor(blobDesc->getTensorDesc())),
      isRemoteMemory(typeOfBlob == BlobDescType::VideoWorkload),
      // It should be not possible to set local blob in VideoWorkload case
      isNeedAllocation(!(typeOfBlob == BlobDescType::VideoWorkload && isInput)),
      // TODO No compound support now
      isCompound(false),
      nnInputColorFormat(graphColorFormat) {
    checkDataIsValid(blobDesc);
}

AllocationInfo::AllocationInfo(const IE::Blob::CPtr& blob, const IE::ColorFormat& graphColorFormat)
    : precision(Unite::convertFromIEPrecision(blob->getTensorDesc().getPrecision())),
      dataSize(blob->size()),
      isRemoteMemory(isRemoteBlob(blob)),
      isNeedAllocation(!isRemoteBlob(blob)),
      isCompound(false),
      nnInputColorFormat(graphColorFormat) {}

bool AllocationInfo::operator==(const AllocationInfo& rhs) const {
    return precision == rhs.precision && dataSize == rhs.dataSize && isRemoteMemory == rhs.isRemoteMemory &&
           isNeedAllocation == rhs.isNeedAllocation && isCompound == rhs.isCompound &&
           nnInputColorFormat == rhs.nnInputColorFormat;
}
bool AllocationInfo::operator!=(const AllocationInfo& rhs) const { return !(rhs == *this); }

static void validateAllocatorInfoFields(const AllocationInfo& allocationInfo) {
    if (allocationInfo.dataSize == 0) {
        THROW_IE_EXCEPTION << "BlobDescriptorAdapter: dataSize is zero!";
    }
}
//------------------------------------------------------------------------------
NNInputInfo::NNInputInfo(const BlobDescType typeOfBlob, const IE::DataPtr& blobDesc)
    : precision(Unite::convertFromIEPrecision(blobDesc->getPrecision())),
      dataSize(getSizeFromTensor(blobDesc->getTensorDesc())),
      isRemoteMemory(typeOfBlob == BlobDescType::VideoWorkload),
      // Not possible create intermediate buffer from plugin side
      isNeedAllocation(true),
      batch(1) {}

//------------------------------------------------------------------------------
BlobDescriptorAdapter::BlobDescriptorAdapter(BlobDescType typeOfBlob, const InferenceEngine::DataPtr& blobDesc,
    const InferenceEngine::ColorFormat& graphColorFormat, const bool isInput)
    : _blobType(typeOfBlob),
      _allocationInfo(typeOfBlob, blobDesc, graphColorFormat, isInput),
      _nnInputInfo(typeOfBlob, blobDesc) {}

BlobDescriptorAdapter::BlobDescriptorAdapter(const InferenceEngine::Blob::CPtr& blobPtr,
    const InferenceEngine::ColorFormat& graphColorFormat, const InferenceEngine::DataPtr& blobDesc)
    : _blobType(isRemoteBlob(blobPtr) ? BlobDescType::VideoWorkload : BlobDescType::ImageWorkload),
      _allocationInfo(blobPtr, graphColorFormat),
      _nnInputInfo(_blobType, blobDesc) {
    checkBlobCompatibility(blobPtr, _blobType);
    // TODO [Workaround] If it's NV12 Blob, use repacked memory instead
    if (blobPtr->is<IE::NV12Blob>()) {
        createRepackedNV12Blob(blobPtr);
        checkBlobIsValid(_repackedBlob);
        _allocationInfo.dataSize = _repackedBlob->byteSize();
    }
}

const HddlUnite::Inference::BlobDesc& BlobDescriptorAdapter::createUniteBlobDesc(const bool isInput) {
    validateAllocatorInfoFields(_allocationInfo);
    _hddlUniteBlobDesc = HddlUnite::Inference::BlobDesc(_allocationInfo.precision, _allocationInfo.isRemoteMemory,
        _allocationInfo.isNeedAllocation, _allocationInfo.dataSize, _allocationInfo.isCompound);

    if (isInput) _hddlUniteBlobDesc.m_nnInputFormat = covertColorFormat(_allocationInfo.nnInputColorFormat);
    return _hddlUniteBlobDesc;
}

const HddlUnite::Inference::BlobDesc& BlobDescriptorAdapter::updateUniteBlobDesc(const IE::Blob::CPtr& blobPtr) {
    checkBlobIsValid(blobPtr);
    validateAllocatorInfoFields(_allocationInfo);
    checkBlobCompatibility(blobPtr, _blobType);
    isBlobDescSuitableForBlob(blobPtr);  // This step might be excess, but to be sure

    auto parsedBlobParamsPtr = std::make_shared<vpux::ParsedRemoteBlobParams>();
    if (blobPtr->is<IE::RemoteBlob>()) {
        const auto remoteBlob = std::dynamic_pointer_cast<const InferenceEngine::RemoteBlob>(blobPtr);
        checkBlobIsValid(remoteBlob);
        parsedBlobParamsPtr->update(remoteBlob->getParams());
        const auto memoryDesc = vpux::HDDL2::getRemoteMemoryFromParams(remoteBlob->getParams());
        if (memoryDesc == nullptr) {
            THROW_IE_EXCEPTION << "BlobDescriptorAdapter: Memory desc is null";
        }
        _sourceInfo.remoteMemoryFd = memoryDesc->getDmaBufFd();
    } else {
        if (blobPtr->is<IE::NV12Blob>()) {
            // TODO Double repacked blob creation
            createRepackedNV12Blob(blobPtr);
            checkBlobIsValid(_repackedBlob);
            _sourceInfo.localMemoryPtr = _repackedBlob->cbuffer().as<void*>();
        } else {
            _sourceInfo.localMemoryPtr = blobPtr->cbuffer().as<void*>();
        }
    }

    prepareImageFormatInfo(blobPtr, parsedBlobParamsPtr);
    getRect(blobPtr, parsedBlobParamsPtr);

    // Updating blobDesc with gathered params
    if (_blobType == BlobDescType::VideoWorkload) {
        _hddlUniteBlobDesc.m_fd = _sourceInfo.remoteMemoryFd;
    } else {
        _hddlUniteBlobDesc.m_srcPtr = _sourceInfo.localMemoryPtr;
    }
    _hddlUniteBlobDesc.m_format = _sourceInfo.blobColorFormat;
    _hddlUniteBlobDesc.m_resWidth = _sourceInfo.resWidth;
    _hddlUniteBlobDesc.m_resHeight = _sourceInfo.resHeight;
    _hddlUniteBlobDesc.m_widthStride = _sourceInfo.widthStride;
    _hddlUniteBlobDesc.m_planeStride = _sourceInfo.planeStride;
    _hddlUniteBlobDesc.m_rect = _sourceInfo.roiRectangles;

    return _hddlUniteBlobDesc;
}

void BlobDescriptorAdapter::createRepackedNV12Blob(const IE::Blob::CPtr& blobPtr) {
    if (!blobPtr->is<IE::NV12Blob>()) THROW_IE_EXCEPTION << "Incorrect blob for repacking!";

    auto nv12Ptr = blobPtr->as<IE::NV12Blob>();
    if (nv12Ptr == nullptr) THROW_IE_EXCEPTION << "Failed to cast nv12 blob.";

    auto nv12Tensor = nv12Ptr->getTensorDesc();
    if (nv12Tensor.getPrecision() != IE::Precision::U8) THROW_IE_EXCEPTION << "Unsupported NV12 Blob precision.";

    IE::Blob::Ptr yPlane = nv12Ptr->y();
    IE::Blob::Ptr uvPlane = nv12Ptr->uv();
    checkBlobIsValid(yPlane);
    checkBlobIsValid(uvPlane);

    const size_t repackedBlobSize = yPlane->size() + uvPlane->size();
    IE::TensorDesc repackedTensor = IE::TensorDesc(IE::Precision::U8, {1, repackedBlobSize}, IE::Layout::NC);

    IE::Blob::Ptr repackedBlob = IE::make_shared_blob<uint8_t>(repackedTensor);
    repackedBlob->allocate();

    {
        IE::MemoryBlob::Ptr yPlaneMBlob = IE::as<IE::MemoryBlob>(yPlane);
        if (yPlaneMBlob == nullptr) THROW_IE_EXCEPTION << "Failed to cast yPlane blob to memory blob";

        auto yPlaneLockedMemory = yPlaneMBlob->rmap();
        auto yPlaneMemory = yPlaneLockedMemory.as<uint8_t*>();
        if (yPlaneMemory == nullptr) THROW_IE_EXCEPTION << "Null yPlane memory";

        IE::MemoryBlob::Ptr uvPlaneMBlob = IE::as<IE::MemoryBlob>(uvPlane);
        if (uvPlaneMBlob == nullptr) THROW_IE_EXCEPTION << "Failed to cast uvPlane blob to memory blob";

        auto uvPlaneLockedMemory = uvPlaneMBlob->rmap();
        auto uvPlaneMemory = uvPlaneLockedMemory.as<uint8_t*>();
        if (uvPlaneMemory == nullptr) THROW_IE_EXCEPTION << "Null uvPlane memory";

        IE::MemoryBlob::Ptr repackedMBlob = IE::as<IE::MemoryBlob>(repackedBlob);
        if (repackedMBlob == nullptr) THROW_IE_EXCEPTION << "Failed to cast blob to memory blob";

        auto repackedBlobLockedMemory = repackedMBlob->wmap();
        auto repackedBlobMemory = repackedBlobLockedMemory.as<uint8_t*>();
        if (repackedBlobMemory == nullptr) THROW_IE_EXCEPTION << "Failed to allocate memory for blob";

        ie_memcpy(repackedBlobMemory, yPlane->size(), yPlaneMemory, yPlane->size());
        ie_memcpy(repackedBlobMemory + yPlane->size(), uvPlane->size(), uvPlaneMemory, uvPlane->size());
    }
    _repackedBlob = repackedBlob;
}

void BlobDescriptorAdapter::prepareImageFormatInfo(
    const IE::Blob::CPtr& blobPtr, const std::shared_ptr<vpux::ParsedRemoteBlobParams>& blobParams) {
    if (isBlobContainsNV12Data(blobPtr, blobParams)) {
        _sourceInfo.blobColorFormat = HddlUnite::Inference::FourCC::NV12;
    } else {
        _sourceInfo.blobColorFormat = HddlUnite::Inference::FourCC::BGR;
    }

    // NN input dims
    IE::SizeVector dims = blobPtr->getTensorDesc().getDims();

    // If it's NV12 data, we should use preprocessing input dims
    if (isBlobContainsNV12Data(blobPtr, blobParams)) {
        dims = getNV12ImageDims(blobPtr);
    }

    if (dims.size() != 4) {
        THROW_IE_EXCEPTION << "BlobDescriptorAdapter: Format with dims != 4 not supported";
    }

    // Dims stored in NCHW format
    const int H_index = 2;
    const int W_index = 3;

    _sourceInfo.resHeight = dims[H_index];
    _sourceInfo.resWidth = _sourceInfo.widthStride = dims[W_index];
    _sourceInfo.planeStride = _sourceInfo.widthStride * _sourceInfo.resHeight;
}

void BlobDescriptorAdapter::getRect(
    const InferenceEngine::Blob::CPtr& blobPtr, const std::shared_ptr<vpux::ParsedRemoteBlobParams>& blobParams) {
    const auto& roiPtr = blobParams->getROIPtr();
    if (blobPtr->is<IE::NV12Blob>() || roiPtr != nullptr) {
        if (roiPtr != nullptr) {
            // TODO Only one ROI is supported
            HddlUnite::Inference::Rectangle roi0{static_cast<int32_t>(roiPtr->posX), static_cast<int32_t>(roiPtr->posY),
                static_cast<int32_t>(roiPtr->sizeX), static_cast<int32_t>(roiPtr->sizeY)};
            _sourceInfo.roiRectangles.push_back(roi0);
        }
    }
}

HddlUnite::Inference::NNInputDesc BlobDescriptorAdapter::createNNDesc() const {
    return HddlUnite::Inference::NNInputDesc(_nnInputInfo.precision, _nnInputInfo.isRemoteMemory,
        _nnInputInfo.isNeedAllocation, _nnInputInfo.dataSize, _nnInputInfo.batch);
}

bool BlobDescriptorAdapter::isBlobDescSuitableForBlob(const InferenceEngine::Blob::CPtr& blob) const {
    return AllocationInfo(blob, _allocationInfo.nnInputColorFormat) == _allocationInfo;
}
bool BlobDescriptorAdapter::isROIPreprocessingRequired() const { return _sourceInfo.roiRectangles.size() > 0; }

}  // namespace HDDL2Plugin
}  // namespace vpu
