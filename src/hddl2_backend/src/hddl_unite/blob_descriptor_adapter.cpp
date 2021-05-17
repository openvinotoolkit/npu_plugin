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
// Utils
#include "vpux/utils/IE/blob.hpp"

namespace vpux {
namespace hddl2 {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------

static void checkBlobIsValid(const IE::Blob::CPtr& blob) {
    if (blob == nullptr) {
        IE_THROW() << "Blob is null";
    }
    if (blob->size() == 0 && !blob->is<IE::NV12Blob>()) {
        IE_THROW() << "Blob is empty";
    }
    if (blob->is<IE::NV12Blob>()) {
        const auto yBlob = IE::as<IE::NV12Blob>(blob)->y();
        const auto uvBlob = IE::as<IE::NV12Blob>(blob)->uv();
        if (yBlob == nullptr || uvBlob == nullptr) {
            IE_THROW() << "NV12 y/uv plane is null";
        }
        if (yBlob->size() == 0 || uvBlob->size() == 0) {
            IE_THROW() << "NV12 y/uv plane is empty";
        }
    }
}

static VpuxRemoteMemoryFD getBlobRemoteMemoryFD(const IE::Blob::CPtr& blob) {
    if (!isRemoteAnyBlob(blob)) {
        IE_THROW() << "Get blob remote memory FD: this blob is not remote";
    }
    VpuxRemoteMemoryFD remoteMemoryFd = -1;
    if (isRemoteNV12Blob(blob)) {
        const auto remoteYBlob = IE::as<IE::RemoteBlob>(IE::as<IE::NV12Blob>(blob)->y());
        const auto remoteMemoryY = getRemoteMemoryFDFromParams(remoteYBlob->getParams());
        const auto remoteUVBlob = IE::as<IE::RemoteBlob>(IE::as<IE::NV12Blob>(blob)->uv());
        const auto remoteMemoryUV = getRemoteMemoryFDFromParams(remoteUVBlob->getParams());
        if (remoteMemoryY != remoteMemoryUV) {
            IE_THROW()
                    << "Get blob remote memory FD: NV12 remote blob must have common RemoteMemory FD for Y/UV planes";
        }
        remoteMemoryFd = remoteMemoryY;
    } else {
        const auto remoteBlob = IE::as<IE::RemoteBlob>(blob);
        remoteMemoryFd = getRemoteMemoryFDFromParams(remoteBlob->getParams());
    }

    return remoteMemoryFd;
}

static void checkBlobCompatibility(const IE::Blob::CPtr& blob, const BlobDescType blobType) {
    checkBlobIsValid(blob);
    if (isRemoteAnyBlob(blob)) {
        if (blobType == BlobDescType::ImageWorkload) {
            IE_THROW() << "Remote blob is not supported for ImageWorkload! Context required.";
        }
    } else {
        if (blobType == BlobDescType::VideoWorkload) {
            IE_THROW() << "Local blob is not supported for VideoWorkload! Please use only remote blobs!";
        }
    }

    if (isRemoteAnyBlob(blob)) {
        const auto remoteMemoryFd = getBlobRemoteMemoryFD(blob);
        if (remoteMemoryFd < 0) {
            IE_THROW() << "Remote blob has bad remote memory FD";
        }
        return;
    }

    if (isLocalNV12Blob(blob)) {
        const auto nv12Blob = IE::as<IE::NV12Blob>(blob);
        const auto yBlob = IE::as<IE::MemoryBlob>(nv12Blob->y());
        const auto uvBlob = IE::as<IE::MemoryBlob>(nv12Blob->uv());
        const auto yAddr = yBlob->rmap().as<const char*>();
        const auto uvAddr = uvBlob->rmap().as<const char*>();
        const auto offset = yBlob->byteSize();
        if (yAddr + offset != uvAddr) {
            IE_THROW() << "Local NV12 blob must be allocated in one continuous memory area";
        }
        return;
    }

    if (blob->is<IE::MemoryBlob>()) {
        return;
    }

    if (blob->is<IE::CompoundBlob>()) {
        IE_THROW() << "CompoundBlob is not supported";
    }

    IE_THROW() << "Blob type is unexpected";
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

static const IE::TensorDesc& getSuitableTensorDesc(const IE::Blob::CPtr& blob) {
    checkBlobIsValid(blob);
    if (!isNV12AnyBlob(blob)) {
        return blob->getTensorDesc();
    }
    const auto yBlob = IE::as<IE::NV12Blob>(blob)->y();
    return yBlob->getTensorDesc();
}

static size_t getSizeFromBlob(const IE::Blob::CPtr& blob) {
    size_t dataSize = 0;
    checkBlobIsValid(blob);
    if (isRemoteNV12Blob(blob)) {
        const auto nv12Blob = IE::as<IE::NV12Blob>(blob);
        const auto yBlob = IE::as<IE::RemoteBlob>(nv12Blob->y());
        auto&& remoteMemoryDesc = getRemoteMemoryFromParams(yBlob->getParams())->getMemoryDesc();
        dataSize = remoteMemoryDesc.getDataSize();
    } else if (isRemoteBlob(blob)) {
        const auto remoteBlob = IE::as<IE::RemoteBlob>(blob);
        auto&& remoteMemoryDesc = getRemoteMemoryFromParams(remoteBlob->getParams())->getMemoryDesc();
        dataSize = remoteMemoryDesc.getDataSize();
    } else {
        dataSize = getSizeFromTensor(getSuitableTensorDesc(blob));
        if (isLocalNV12Blob(blob)) {
            dataSize = dataSize * 3 / 2;
        }
    }

    return dataSize;
}

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
}

AllocationInfo::AllocationInfo(const IE::Blob::CPtr& blob, const IE::ColorFormat& graphColorFormat)
        : precision(Unite::convertFromIEPrecision(getSuitableTensorDesc(blob).getPrecision())),
          dataSize(getSizeFromBlob(blob)),
          isRemoteMemory(isRemoteAnyBlob(blob)),
          isNeedAllocation(!(isRemoteAnyBlob(blob))),
          isCompound(false),
          nnInputColorFormat(graphColorFormat) {
}

bool AllocationInfo::operator==(const AllocationInfo& rhs) const {
    return precision == rhs.precision && dataSize == rhs.dataSize && isRemoteMemory == rhs.isRemoteMemory &&
           isNeedAllocation == rhs.isNeedAllocation && isCompound == rhs.isCompound &&
           nnInputColorFormat == rhs.nnInputColorFormat;
}

bool AllocationInfo::operator!=(const AllocationInfo& rhs) const {
    return !(rhs == *this);
}

static void validateAllocatorInfoFields(const AllocationInfo& allocationInfo) {
    if (allocationInfo.dataSize == 0) {
        IE_THROW() << "BlobDescriptorAdapter: dataSize is zero!";
    }
}
//------------------------------------------------------------------------------
NNInputInfo::NNInputInfo(const BlobDescType typeOfBlob, const IE::DataPtr& blobDesc)
        : precision(Unite::convertFromIEPrecision(blobDesc->getPrecision())),
          dataSize(getSizeFromTensor(blobDesc->getTensorDesc())),
          isRemoteMemory(typeOfBlob == BlobDescType::VideoWorkload),
          // Not possible create intermediate buffer from plugin side
          isNeedAllocation(true),
          batch(1) {
}

//------------------------------------------------------------------------------
BlobDescriptorAdapter::BlobDescriptorAdapter(BlobDescType typeOfBlob, const IE::DataPtr& blobDesc,
                                             const IE::ColorFormat& graphColorFormat, const bool isInput)
        : _blobType(typeOfBlob),
          _allocationInfo(typeOfBlob, blobDesc, graphColorFormat, isInput),
          _nnInputInfo(typeOfBlob, blobDesc) {
}

BlobDescriptorAdapter::BlobDescriptorAdapter(const IE::Blob::CPtr& blobPtr, const IE::ColorFormat& graphColorFormat,
                                             const IE::DataPtr& blobDesc)
        : _blobType(isRemoteAnyBlob(blobPtr) ? BlobDescType::VideoWorkload : BlobDescType::ImageWorkload),
          _allocationInfo(blobPtr, graphColorFormat),
          _nnInputInfo(_blobType, blobDesc) {
    checkBlobCompatibility(blobPtr, _blobType);
}

const HddlUnite::Inference::BlobDesc& BlobDescriptorAdapter::createUniteBlobDesc(const bool isInput) {
    validateAllocatorInfoFields(_allocationInfo);
    _hddlUniteBlobDesc = HddlUnite::Inference::BlobDesc(_allocationInfo.precision, _allocationInfo.isRemoteMemory,
                                                        _allocationInfo.isNeedAllocation, _allocationInfo.dataSize,
                                                        _allocationInfo.isCompound);

    if (isInput)
        _hddlUniteBlobDesc.m_nnInputFormat = covertColorFormat(_allocationInfo.nnInputColorFormat);
    return _hddlUniteBlobDesc;
}

const HddlUnite::Inference::BlobDesc& BlobDescriptorAdapter::updateUniteBlobDesc(const IE::Blob::CPtr& blobPtr,
                                                                                 const IE::ColorFormat colorFormat) {
    checkBlobIsValid(blobPtr);
    validateAllocatorInfoFields(_allocationInfo);
    checkBlobCompatibility(blobPtr, _blobType);
    isBlobDescSuitableForBlob(blobPtr);  // This step might be excess, but to be sure

    auto parsedBlobParamsPtr = std::make_shared<vpux::ParsedRemoteBlobParams>();
    if (isRemoteBlob(blobPtr)) {
        _sourceInfo.remoteMemoryFd = getBlobRemoteMemoryFD(blobPtr);
        const auto remoteBlob = IE::as<IE::RemoteBlob>(blobPtr);
        parsedBlobParamsPtr->update(remoteBlob->getParams());
    } else if (isRemoteNV12Blob(blobPtr)) {
        _sourceInfo.remoteMemoryFd = getBlobRemoteMemoryFD(blobPtr);
        const auto remoteYBlob = IE::as<IE::RemoteBlob>(IE::as<IE::NV12Blob>(blobPtr)->y());
        parsedBlobParamsPtr->update(remoteYBlob->getParams());
    } else {
        if (blobPtr->is<IE::NV12Blob>()) {
            _sourceInfo.localMemoryPtr = IE::as<IE::NV12Blob>(blobPtr)->y()->cbuffer().as<void*>();
        } else {
            _sourceInfo.localMemoryPtr = blobPtr->cbuffer().as<void*>();
        }
    }

    prepareImageFormatInfo(blobPtr, colorFormat);
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

void BlobDescriptorAdapter::prepareImageFormatInfo(const IE::Blob::CPtr& blobPtr, const IE::ColorFormat colorFormat) {
    if (colorFormat == IE::ColorFormat::NV12) {
        if (!isNV12AnyBlob(blobPtr)) {
            IE_THROW() << "BlobDescriptorAdapter: blob type and color parameter are not compatible.";
        }
        _sourceInfo.blobColorFormat = HddlUnite::Inference::FourCC::NV12;
    } else {
        _sourceInfo.blobColorFormat = covertColorFormat(colorFormat);
    }

    // NN input dims
    const auto& tensorDesc = getSuitableTensorDesc(blobPtr);
    const auto& dims = tensorDesc.getDims();

    if (dims.size() != 4) {
        IE_THROW() << "BlobDescriptorAdapter: Formats with dims != 4 are not supported.";
    }

    if (!_allocationInfo.isRemoteMemory) {
        // Local memory - get information from TensorDesc dims and strides
        // Dims stored in NCHW format
        const int H_index = 2;
        const int W_index = 3;

        const auto& blockingDesc = tensorDesc.getBlockingDesc();
        const auto& strides = blockingDesc.getStrides();
        if (strides.empty()) {
            IE_THROW() << "Strides information is not provided.";
        }

        // Define strides and dimensions. Only NCHW/NHWC orders/layouts are supported. NV12 always has NHWC order
        const bool isNCHW = isNV12AnyBlob(blobPtr) ? false : (tensorDesc.getLayout() == IE::Layout::NCHW);
        _sourceInfo.widthStride = strides[isNCHW ? 2 : 1];
        _sourceInfo.planeStride = strides[isNCHW ? 1 : 0];
        _sourceInfo.resWidth = dims[W_index];
        _sourceInfo.resHeight = dims[H_index];
    } else {
        // Remote memory - get information from RemoteMemory HddlUnite object
        const auto remoteBlob = isRemoteNV12Blob(blobPtr) ? IE::as<IE::NV12Blob>(blobPtr)->y() : blobPtr;
        const auto& blobParams = IE::as<IE::RemoteBlob>(remoteBlob)->getParams();
        auto&& memoryDesc = getRemoteMemoryFromParams(blobParams)->getMemoryDesc();
        _sourceInfo.widthStride = memoryDesc.m_widthStride;
        _sourceInfo.planeStride = memoryDesc.m_widthStride * memoryDesc.m_heightStride;
        _sourceInfo.resWidth = memoryDesc.m_width;
        _sourceInfo.resHeight = memoryDesc.m_height;
    }
}

void BlobDescriptorAdapter::getRect(const IE::Blob::CPtr& blobPtr,
                                    const vpux::ParsedRemoteBlobParams::CPtr& blobParams) {
    const auto& roiPtr = blobParams->getROIPtr();
    _sourceInfo.roiRectangles.clear();
    if (blobPtr->is<IE::NV12Blob>() || roiPtr != nullptr) {
        if (roiPtr != nullptr) {
            // TODO Only one ROI is supported
            HddlUnite::Inference::Rectangle roi0{static_cast<int32_t>(roiPtr->posX), static_cast<int32_t>(roiPtr->posY),
                                                 static_cast<int32_t>(roiPtr->sizeX),
                                                 static_cast<int32_t>(roiPtr->sizeY)};
            _sourceInfo.roiRectangles.push_back(roi0);
        }
    } else {
        // Set default ROI
        HddlUnite::Inference::Rectangle defaultROI{static_cast<int32_t>(0), static_cast<int32_t>(0),
                                                   static_cast<int32_t>(_sourceInfo.resWidth),
                                                   static_cast<int32_t>(_sourceInfo.resHeight)};
        _sourceInfo.roiRectangles.push_back(defaultROI);
    }
}

HddlUnite::Inference::NNInputDesc BlobDescriptorAdapter::createNNDesc() const {
    return HddlUnite::Inference::NNInputDesc(_nnInputInfo.precision, _nnInputInfo.isRemoteMemory,
                                             _nnInputInfo.isNeedAllocation, _nnInputInfo.dataSize, _nnInputInfo.batch);
}

bool BlobDescriptorAdapter::isBlobDescSuitableForBlob(const IE::Blob::CPtr& blob) const {
    return AllocationInfo(blob, _allocationInfo.nnInputColorFormat) == _allocationInfo;
}
bool BlobDescriptorAdapter::isROIPreprocessingRequired() const {
    return _sourceInfo.roiRectangles.size() > 0;
}

}  // namespace hddl2
}  // namespace vpux
