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
#include <memory>

#include "converters.h"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static void checkBlobIsValid(const InferenceEngine::Blob::Ptr& blob) {
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "Blob is null";
    }
    if (blob->size() == 0) {
        THROW_IE_EXCEPTION << "Blob is empty";
    }
}

static void checkBlobCompatibility(const InferenceEngine::Blob::Ptr& blob) {
    if (blob->is<IE::CompoundBlob>()) {
        THROW_IE_EXCEPTION << "CompoundBlob is not supported";
    }
    if (blob->is<IE::NV12Blob>()) {
        THROW_IE_EXCEPTION << "NV12Blob is not supported";
    }
    if (blob->is<HDDL2RemoteBlob>() || blob->is<IE::MemoryBlob>()) {
        return;
    }
    THROW_IE_EXCEPTION << "Blob type is unexpected";
}

static void checkDataIsValid(const InferenceEngine::DataPtr& data) {
    if (data == nullptr) {
        THROW_IE_EXCEPTION << "Blob descriptor is null";
    }
}

static RemoteMemoryFD getFDFromRemoteBlob(const IE::Blob::Ptr& blob) {
    RemoteMemoryFD memoryFd = 0;
    try {
        HDDL2RemoteBlob::Ptr remoteBlobPtr = std::dynamic_pointer_cast<HDDL2RemoteBlob>(blob);
        memoryFd = remoteBlobPtr->getRemoteMemoryFD();
    } catch (const std::exception& ex) {
        printf("Failed to get memory fd from remote blob! %s\n", ex.what());
    }
    return memoryFd;
}

//------------------------------------------------------------------------------
BlobDescriptor::BlobDescriptor(
    const bool& isInput, const InferenceEngine::DataPtr& desc, const InferenceEngine::Blob::Ptr& blob) {
    _isInput = isInput;

    checkBlobIsValid(blob);
    checkBlobCompatibility(blob);

    _blobPtr = blob;

    checkDataIsValid(desc);
    _desc = desc;
}

HddlUnite::Inference::BlobDesc BlobDescriptor::create() {
    // TODO Why I can't get it from desc???
    const IE::TensorDesc tensorDesc = _blobPtr->getTensorDesc();
    HddlUnite::Inference::Precision precision = Unite::convertFromIEPrecision(tensorDesc.getPrecision());

    const size_t blobSize = _blobPtr->byteSize();
    _blobDesc = HddlUnite::Inference::BlobDesc(precision, _isRemoteMemory, _isNeedAllocation, blobSize);

    return _blobDesc;
}

void BlobDescriptor::createRepackedNV12Blob(const InferenceEngine::Blob::Ptr& blobPtr) {
    if (!blobPtr->is<InferenceEngine::NV12Blob>()) {
        THROW_IE_EXCEPTION << "Incorrect blob for repacking!";
    }
    auto nv12Ptr = blobPtr->as<IE::NV12Blob>();
    auto repackedBlob = InferenceEngine::make_shared_blob<uint8_t>(blobPtr->getTensorDesc());

    // FIXME throw exceptions instead of assert
    {
        auto yPlane = nv12Ptr->y();
        IE::Blob::Ptr blob;
        auto mblob = IE::as<IE::MemoryBlob>(blob);
        IE_ASSERT(!yPlane->is<IE::RemoteBlob>());
        auto yPlaneMemory = yPlane->buffer().as<uint8_t*>();

        auto uvPlane = nv12Ptr->uv();
        IE_ASSERT(!uvPlane->is<IE::RemoteBlob>());
        auto uvPlaneMemory = uvPlane->buffer().as<uint8_t*>();

        IE_ASSERT(repackedBlob->size() == (yPlane->size() + uvPlane->size()));
        auto memory = repackedBlob->buffer().as<uint8_t*>();
        memcpy(memory, yPlaneMemory, yPlane->size());
        memcpy(memory + yPlane->size(), uvPlaneMemory, uvPlane->size());
    }
    _repackedBlob = repackedBlob;
}

void BlobDescriptor::setBlobFormat() {
    _blobDesc.m_format = HddlUnite::Inference::FourCC::BGR;
    if (_blobPtr->is<IE::NV12Blob>()) {
        _blobDesc.m_format = HddlUnite::Inference::FourCC::NV12;
    }

    const IE::TensorDesc tensorDesc = _blobPtr->getTensorDesc();

    if (tensorDesc.getLayout() != IE::NCHW) {
        THROW_IE_EXCEPTION << "Failed to create blob description for " << tensorDesc.getLayout() << " layout.";
    }
    const uint H_index = 2;
    const uint W_index = 3;

    // FIXME Not sure about that
    _blobDesc.m_res_height = _blobDesc.m_plane_stride = tensorDesc.getDims()[H_index];
    _blobDesc.m_res_width = _blobDesc.m_width_stride = tensorDesc.getDims()[W_index];
}

void BlobDescriptor::setPreProcessing(const InferenceEngine::PreProcessInfo& preProcess) {
    _preProcessPtr = std::make_shared<InferenceEngine::PreProcessInfo>(preProcess);
}

//------------------------------------------------------------------------------
LocalBlobDescriptor::LocalBlobDescriptor(
    const bool& isInput, const InferenceEngine::DataPtr& desc, const InferenceEngine::Blob::Ptr& blob = nullptr)
    : BlobDescriptor(isInput, desc, blob) {
    _isRemoteMemory = false;
    // TODO For output memory buffer is not provided (for now)
    _isNeedAllocation = !_isInput;
}

HddlUnite::Inference::BlobDesc LocalBlobDescriptor::init() {
    _blobDesc.m_srcPtr = _blobPtr->buffer().as<void*>();

    // TODO [Workaround] If it's NV12 Blob, use repacked memory instead
    if (_blobPtr->is<InferenceEngine::NV12Blob>()) {
        createRepackedNV12Blob(_blobPtr);
        _blobDesc.m_srcPtr = _repackedBlob->buffer().as<void*>();
    }

    setBlobFormat();

    return _blobDesc;
}

//------------------------------------------------------------------------------
RemoteBlobDescriptor::RemoteBlobDescriptor(
    const bool& isInput, const InferenceEngine::DataPtr& desc, const InferenceEngine::Blob::Ptr& blob = nullptr)
    : BlobDescriptor(isInput, desc, blob) {
    _isRemoteMemory = true;
}

HddlUnite::Inference::BlobDesc RemoteBlobDescriptor::init() {
    if (_blobPtr->is<HDDL2RemoteBlob>()) {
        _isNeedAllocation = false;
        _blobDesc.m_fd = getFDFromRemoteBlob(_blobPtr);
    } else {
        _isNeedAllocation = true;
        _blobDesc.m_srcPtr = _blobPtr->buffer().as<void*>();

        // TODO [Workaround] If it's NV12 Blob, use repacked memory instead
        if (_blobPtr->is<InferenceEngine::NV12Blob>()) {
            createRepackedNV12Blob(_blobPtr);
            _blobDesc.m_srcPtr = _repackedBlob->buffer().as<void*>();
        }
    }

    setBlobFormat();

    return _blobDesc;
}
