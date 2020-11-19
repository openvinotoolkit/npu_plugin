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
// Plugin
#include "vpux_exceptions.h"
#include "vpux_params_private_options.h"
#include "vpux_remote_blob.h"

namespace vpux {
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
VPUXRemoteBlob::VPUXRemoteBlob(const IE::TensorDesc& tensorDesc, const VPUXRemoteContext::Ptr& contextPtr,
    const std::shared_ptr<Allocator>& allocator, const IE::ParamMap& params, const vpu::LogLevel logLevel)
    : RemoteBlob(tensorDesc),
      _remoteContextPtr(contextPtr),
      _allocatorPtr(allocator),
      _logger(std::make_shared<vpu::Logger>("VPUXRemoteBlob", logLevel, vpu::consoleOutput())),
      _originalTensorDesc(tensorDesc) {
    if (contextPtr == nullptr) {
        THROW_IE_EXCEPTION << CONTEXT_ERROR_str << "Remote context is null.";
    }
    _parsedParams.update(params);
    _logger->trace("VPUXRemoteBlob wrapping %d size\n", static_cast<int>(this->size()));

    auto updatedParams = IE::ParamMap(params);
    updatedParams.insert({{IE::KMB_PARAM_KEY(ALLOCATION_SIZE), this->size()}});
    // TODO since we can't use _allocatorPtr to wrap remote memory (instead, we are using input allocator)
    //  this shown design flaw in RemoteBlob + IE:Allocator concept
    _memoryHandle = allocator->wrapRemoteMemory(updatedParams);
    if (_memoryHandle == nullptr) {
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Allocation error";
    }
}

VPUXRemoteBlob::~VPUXRemoteBlob() {
    if (_allocatorPtr != nullptr) {
        _allocatorPtr->free(_memoryHandle);
    }
}

static std::shared_ptr<IE::ROI> makeROIOverROI(const std::shared_ptr<const IE::ROI>& origROIPtr,
    const IE::ROI& appliedROI, const size_t width, const size_t height) {
    std::shared_ptr<IE::ROI> resultROI = nullptr;
    if (origROIPtr) {
        IE::ROI newROI = appliedROI;
        newROI.posX += origROIPtr->posX;
        newROI.posY += origROIPtr->posY;
        resultROI = std::make_shared<IE::ROI>(newROI);
    } else {
        resultROI = std::make_shared<IE::ROI>(appliedROI);
    }

    if ((resultROI->posX + resultROI->sizeX > width) || (resultROI->posY + resultROI->sizeY > height)) {
        THROW_IE_EXCEPTION << "ROI out of blob bounds";
    }
    return resultROI;
}

VPUXRemoteBlob::VPUXRemoteBlob(const VPUXRemoteBlob& origBlob, const IE::ROI& regionOfInterest)
    : RemoteBlob(make_roi_desc(origBlob.getTensorDesc(), regionOfInterest, true)),
      _parsedParams(origBlob._parsedParams),
      _remoteContextPtr(origBlob._remoteContextPtr),
      _allocatorPtr(origBlob._allocatorPtr),
      _logger(std::make_shared<vpu::Logger>("VPUXRemoteBlob", origBlob._logger->level(), vpu::consoleOutput())),
      _originalTensorDesc(origBlob.getOriginalTensorDesc()) {
    if (_allocatorPtr == nullptr) {
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set allocator";
    }

    if (tensorDesc.getDims().size() != 4) {
        THROW_IE_EXCEPTION << "Unsupported layout for VPUXRemoteBlob";
    }
    const auto origBlobTensorDesc = origBlob.getOriginalTensorDesc();
    const auto orig_W = origBlobTensorDesc.getDims()[3];
    const auto orig_H = origBlobTensorDesc.getDims()[2];
    auto newROI = makeROIOverROI(_parsedParams.getROIPtr(), regionOfInterest, orig_W, orig_H);
    // With ROI param full tensor desc also should be stored to be able to get full frame information
    IE::ParamMap updatedROIPtrParam = {{IE::KMB_PARAM_KEY(ROI_PTR), newROI},
        {IE::KMB_PARAM_KEY(ORIGINAL_TENSOR_DESC), std::make_shared<IE::TensorDesc>(origBlob.getOriginalTensorDesc())}};
    _parsedParams.update(updatedROIPtrParam);

    // TODO Remove this cast
    const auto privateAllocator = std::static_pointer_cast<Allocator>(_allocatorPtr);
    IE::ParamMap params = {{IE::KMB_PARAM_KEY(BLOB_MEMORY_HANDLE), origBlob._memoryHandle},
        {IE::KMB_PARAM_KEY(ALLOCATION_SIZE), origBlob.size()}};

    try {
        auto origParams = origBlob.getParams();
        params.insert(origParams.begin(), origParams.end());
    } catch (std::exception& ex) {
        THROW_IE_EXCEPTION << "VPUXRemoteBlob: Failed to use original blob params" << ex.what();
    }

    _memoryHandle = privateAllocator->wrapRemoteMemory(params);
    if (_memoryHandle == nullptr) {
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to copy remote memory handle";
    }
}

IE::LockedMemory<void> VPUXRemoteBlob::buffer() noexcept {
    return IE::LockedMemory<void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

IE::LockedMemory<const void> VPUXRemoteBlob::cbuffer() const noexcept {
    return IE::LockedMemory<const void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

IE::LockedMemory<void> VPUXRemoteBlob::rwmap() noexcept {
    return IE::LockedMemory<void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

IE::LockedMemory<const void> VPUXRemoteBlob::rmap() const noexcept {
    return IE::LockedMemory<const void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

IE::LockedMemory<void> VPUXRemoteBlob::wmap() noexcept {
    return IE::LockedMemory<void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

std::string VPUXRemoteBlob::getDeviceName() const noexcept {
    auto remoteContext = _remoteContextPtr.lock();
    if (remoteContext == nullptr) {
        return "";
    }
    return remoteContext->getDeviceName();
}

std::shared_ptr<IE::RemoteContext> VPUXRemoteBlob::getContext() const noexcept { return _remoteContextPtr.lock(); }

const std::shared_ptr<IE::IAllocator>& VPUXRemoteBlob::getAllocator() const noexcept { return _allocatorPtr; }

size_t VPUXRemoteBlob::size() const noexcept {
    if (_parsedParams.getColorFormat() == IE::ColorFormat::NV12) {
        if (tensorDesc.getLayout() == IE::Layout::SCALAR) return 1;
        // FIXME It's a very bad solution
        const auto dims = tensorDesc.getDims();
        size_t height = dims.at(2);
        size_t width = dims.at(3);
        return (3 * width * height) / 2;
    } else {
        return MemoryBlob::size();
    }
}

size_t VPUXRemoteBlob::byteSize() const noexcept { return size() * element_size(); }

IE::Blob::Ptr VPUXRemoteBlob::createROI(const IE::ROI& regionOfInterest) const {
    return Blob::Ptr(new VPUXRemoteBlob(*this, regionOfInterest));
}
}  // namespace vpux
