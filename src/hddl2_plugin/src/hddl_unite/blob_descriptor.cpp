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

#include "blob_descriptor.h"

#include <hddl2_remote_blob.h>
#include <ie_compound_blob.h>

#include <ie_algorithm.hpp>
#include <memory>

#include "converters.h"
#include "hddl2_params.hpp"
#include "ie_memcpy.h"

using namespace vpu::HDDL2Plugin;
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

static void checkBlobCompatibility(const IE::Blob::CPtr& blob) {
    if (blob->is<HDDL2RemoteBlob>() || blob->is<IE::MemoryBlob>() || blob->is<IE::NV12Blob>()) {
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

static RemoteMemoryFD getFDFromRemoteBlob(const IE::Blob::CPtr& blob) {
    RemoteMemoryFD memoryFd = 0;
    try {
        HDDL2RemoteBlob::CPtr remoteBlobPtr = std::dynamic_pointer_cast<const HDDL2RemoteBlob>(blob);
        memoryFd = remoteBlobPtr->getRemoteMemoryFD();
    } catch (const std::exception& ex) {
        printf("Failed to get memory fd from remote blob! %s\n", ex.what());
    }
    return memoryFd;
}
static bool isBlobContainsNV12Data(const IE::Blob::CPtr& blobPtr) {
    if (blobPtr->is<IE::NV12Blob>()) {
        return true;
    } else if (blobPtr->is<HDDL2RemoteBlob>()) {
        auto remoteBlob = blobPtr->as<HDDL2RemoteBlob>();
        return remoteBlob->getColorFormat() == IE::ColorFormat::NV12;
    }
    return false;
}

static IE::SizeVector getNV12ImageDims(const IE::Blob::CPtr& blobPtr) {
    if (blobPtr->is<IE::NV12Blob>()) {
        auto nv12Ptr = blobPtr->as<IE::NV12Blob>();
        if (nv12Ptr == nullptr) {
            THROW_IE_EXCEPTION << "Failed to cast nv12 blob.";
        }
        auto yPlaneBlob = nv12Ptr->y();
        return yPlaneBlob->getTensorDesc().getDims();
    } else if (blobPtr->is<HDDL2RemoteBlob>()) {
        return blobPtr->getTensorDesc().getDims();
    }
    THROW_IE_EXCEPTION << "Unsupported blob format with NV12 Data";
}

static size_t calculateBlobSizeFromTensor(const IE::TensorDesc& tensorDesc) {
    if (tensorDesc.getLayout() == IE::Layout::SCALAR) return 1;
    return IE::details::product(tensorDesc.getDims().begin(), tensorDesc.getDims().end());
}

using matchColorFormats_t = std::unordered_map<int, HddlUnite::Inference::FourCC>;

static HddlUnite::Inference::FourCC getColorFormat(IE::ColorFormat colorFormat) {
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

//------------------------------------------------------------------------------
BlobDescriptor::BlobDescriptor(const IE::DataPtr& desc, const IE::Blob::CPtr& blob, bool createRemoteMemoryDescriptor,
    bool isNeedAllocation, bool isOutput)
    : _createRemoteMemoryDescriptor(createRemoteMemoryDescriptor),
      _isNeedAllocation(isNeedAllocation),
      _blobPtr(blob),
      _isOutput(isOutput) {
    /// For output use only desc information
    if (!isOutput) {
        checkBlobIsValid(blob);
        checkBlobCompatibility(blob);

        _isNV12Data = isBlobContainsNV12Data(blob);
    }

    checkDataIsValid(desc);
    _desc = desc;
}

HddlUnite::Inference::BlobDesc BlobDescriptor::createUniteBlobDesc(
    const bool& isInput, const IE::ColorFormat& colorFormat) {
    HddlUnite::Inference::BlobDesc blobDesc;
    HddlUnite::Inference::Precision precision = Unite::convertFromIEPrecision(_desc->getPrecision());

    size_t blobSize = 0;
    if (!_isOutput) {
        blobSize = _blobPtr->byteSize();

        // TODO [Workaround] If it's NV12 Blob, use repacked memory instead
        if (_blobPtr->is<IE::NV12Blob>()) {
            createRepackedNV12Blob(_blobPtr);
            checkBlobIsValid(_repackedBlob);
            blobSize = _repackedBlob->byteSize();
        }
    } else {
        blobSize = getSizeFromTensor(_desc->getTensorDesc());
    }

    blobDesc = HddlUnite::Inference::BlobDesc(precision, _createRemoteMemoryDescriptor, _isNeedAllocation, blobSize);
    if (isInput) blobDesc.m_nnInputFormat = getColorFormat(colorFormat);

    return blobDesc;
}

void BlobDescriptor::createRepackedNV12Blob(const IE::Blob::CPtr& blobPtr) {
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

void BlobDescriptor::setImageFormatToDesc(HddlUnite::Inference::BlobDesc& blobDesc) {
    blobDesc.m_format = HddlUnite::Inference::FourCC::BGR;
    if (_isNV12Data) {
        blobDesc.m_format = HddlUnite::Inference::FourCC::NV12;
    }
    // NN input dims
    IE::SizeVector dims = _desc->getDims();

    // If it's NV12 data, we should use preprocessing input dims
    if (_isNV12Data) {
        dims = getNV12ImageDims(_blobPtr);
    }

    // Dims stored in NCHW format
    const int H_index = 2;
    const int W_index = 3;

    blobDesc.m_resHeight = dims[H_index];
    blobDesc.m_resWidth = blobDesc.m_widthStride = dims[W_index];
    blobDesc.m_planeStride = blobDesc.m_widthStride * blobDesc.m_resHeight;

    if (_blobPtr->is<IE::NV12Blob>() || _roiPtr != nullptr) {
        if (_roiPtr != nullptr) {
            HddlUnite::Inference::Rectangle roi0{static_cast<int32_t>(_roiPtr->posX),
                static_cast<int32_t>(_roiPtr->posY), static_cast<int32_t>(_roiPtr->sizeX),
                static_cast<int32_t>(_roiPtr->sizeY)};
            blobDesc.m_rect.push_back(roi0);
        } else {
            HddlUnite::Inference::Rectangle rect0{0, 0, blobDesc.m_resWidth, blobDesc.m_resHeight};
            blobDesc.m_rect.push_back(rect0);
        }
    }
}

HddlUnite::Inference::NNInputDesc BlobDescriptor::createNNDesc() {
    HddlUnite::Inference::Precision precision = Unite::convertFromIEPrecision(_desc->getPrecision());
    const bool needAllocation = true;
    const size_t blobSize = calculateBlobSizeFromTensor(_desc->getTensorDesc());
    const int batch = 1;

    return HddlUnite::Inference::NNInputDesc(precision, _createRemoteMemoryDescriptor, needAllocation, blobSize, batch);
}

void BlobDescriptor::initUniteBlobDesc(HddlUnite::Inference::BlobDesc& blobDesc) {
    if (_blobPtr->is<HDDL2RemoteBlob>()) {
        blobDesc.m_fd = getFDFromRemoteBlob(_blobPtr);
    } else {
        // TODO Replace with rlock
        blobDesc.m_srcPtr = _blobPtr->cbuffer().as<void*>();

        if (_blobPtr->is<IE::NV12Blob>()) {
            if (_repackedBlob == nullptr) {
                THROW_IE_EXCEPTION << "Repacked nv12 blob is not created!";
            }
            blobDesc.m_srcPtr = _repackedBlob->buffer().as<void*>();
        }
    }
    setImageFormatToDesc(blobDesc);
}

//------------------------------------------------------------------------------
LocalBlobDescriptor::LocalBlobDescriptor(const IE::DataPtr& desc, const IE::Blob::CPtr& blob)
    : BlobDescriptor(desc, blob, false, true, blob == nullptr) {
    if (blob->is<HDDL2RemoteBlob>()) {
        THROW_IE_EXCEPTION << "Unable to create local blob descriptor from remote memory";
    }
}

//------------------------------------------------------------------------------
RemoteBlobDescriptor::RemoteBlobDescriptor(const IE::DataPtr& desc, const IE::Blob::CPtr& blob)
    : BlobDescriptor(desc, blob, true, blob ? !blob->is<HDDL2RemoteBlob>() : true, blob == nullptr) {
    if (_blobPtr->is<HDDL2RemoteBlob>()) {
        _roiPtr = _blobPtr->as<HDDL2RemoteBlob>()->getROIPtr();
    }
}
