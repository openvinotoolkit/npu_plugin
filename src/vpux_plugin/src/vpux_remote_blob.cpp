//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// System
#include <memory>
#include <string>
// Plugin
#include "vpux_exceptions.h"
#include "vpux_params_private_options.hpp"
#include "vpux_remote_blob.h"

namespace vpux {
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
VPUXRemoteBlob::VPUXRemoteBlob(const IE::TensorDesc& tensorDesc, const VPUXRemoteContext::Ptr& contextPtr,
                               const std::shared_ptr<Allocator>& allocator, const IE::ParamMap& params,
                               const LogLevel logLevel)
        : RemoteBlob(tensorDesc),
          _remoteContextPtr(contextPtr),
          _allocatorPtr(allocator),
          _logger("VPUXRemoteBlob", logLevel),
          _originalTensorDesc(tensorDesc) {
    if (contextPtr == nullptr) {
        IE_THROW() << CONTEXT_ERROR_str << "Remote context is null.";
    }
    if (_allocatorPtr == nullptr) {
        IE_THROW() << "Remote allocator is null.";
    }

    auto updatedParams = IE::ParamMap(params);
    updatedParams.insert(
            {{IE::VPUX_PARAM_KEY(ORIGINAL_TENSOR_DESC), std::make_shared<IE::TensorDesc>(getOriginalTensorDesc())},
             {IE::VPUX_PARAM_KEY(ALLOCATION_SIZE), this->size()}});

    const auto& contextParams = contextPtr->getParams();
    if (contextParams.find(IE::VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID)) != contextParams.end()) {
        uint64_t workloadId = contextParams.at(IE::VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID)).as<uint64_t>();
        updatedParams.insert({{IE::VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}});
    }

    _parsedParams.update(updatedParams);

    // TODO since we can't use _allocatorPtr to wrap remote memory (instead, we are using input allocator)
    //  this shown design flaw in RemoteBlob + IE:Allocator concept
    _memoryHandle = allocator->wrapRemoteMemory(updatedParams);
    if (_memoryHandle == nullptr) {
        IE_THROW(NotAllocated) << "Allocation error";
    }

    updatedParams.insert({{IE::VPUX_PARAM_KEY(MEM_HANDLE), _memoryHandle}});
    _parsedParams.update(updatedParams);
}

VPUXRemoteBlob::~VPUXRemoteBlob() {
// FIXME: there is possibility that of the following scenario on AARCH64 platform:
// 1. Remote blob created and virtual address is put into allocator
// 2. Memory used by remote blob is freed but user can still hold the remote blob
// 3. Local blob created with the same virtual address (this address is already free but remote blob is still alive)
// 4. Local blob replaces information about remote blob memory with its info using virtual address as a handle
// 5. Remote blob removed causing local blob to loose info about its memory
// To fix it properly we would need to fix h#14013533321, but it is not feasible at this point,
// so there was a decision to work around this by not calling free for remote blob on AARCH64,
// since it is not removed any way. Please check implementation vpusmm allocator free method
// TODO: think about local/remote and .isMemoryOwner contracts. It might help to avoid changing this place and
// move fix into vpusmm allocator
#ifndef __aarch64__
    if (_allocatorPtr != nullptr) {
        _allocatorPtr->free(_memoryHandle);
    }
#endif
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
        IE_THROW() << "ROI out of blob bounds";
    }
    return resultROI;
}

VPUXRemoteBlob::VPUXRemoteBlob(const VPUXRemoteBlob& origBlob, const std::vector<std::size_t>& begin,
                               const std::vector<std::size_t>& end)
        : RemoteBlob(make_roi_desc(origBlob.getTensorDesc(), begin, end, true)),
          _parsedParams(origBlob._parsedParams),
          _remoteContextPtr(origBlob._remoteContextPtr),
          _allocatorPtr(origBlob._allocatorPtr),
          _logger(origBlob._logger),
          _originalTensorDesc(origBlob.getOriginalTensorDesc()) {
    if (_allocatorPtr == nullptr) {
        IE_THROW(NotAllocated) << "Failed to set allocator";
    }

    if (tensorDesc.getDims().size() != 4) {
        IE_THROW() << "Unsupported layout for VPUXRemoteBlob";
    }
    const IE::ROI regionOfInterest = {0, begin[3], begin[2], end[3] - begin[3], end[2] - begin[2]};
    const auto origBlobTensorDesc = origBlob.getOriginalTensorDesc();
    const auto orig_W = origBlobTensorDesc.getDims()[3];
    const auto orig_H = origBlobTensorDesc.getDims()[2];
    auto newROI = makeROIOverROI(_parsedParams.getROIPtr(), regionOfInterest, orig_W, orig_H);
    const auto updatedROIPtrParam = IE::ParamMap{{IE::VPUX_PARAM_KEY(ROI_PTR), newROI}};
    _parsedParams.updateFull(updatedROIPtrParam);

    // TODO Remove this cast
    const auto privateAllocator = std::static_pointer_cast<Allocator>(_allocatorPtr);
    _memoryHandle = privateAllocator->wrapRemoteMemory(origBlob.getParams());
    if (_memoryHandle == nullptr) {
        IE_THROW(NotAllocated) << "Failed to copy remote memory handle";
    }
}

void VPUXRemoteBlob::updateColorFormat(const IE::ColorFormat colorFormat) {
    if (_parsedParams.getBlobColorFormat() == colorFormat) {
        return;
    }

    const auto updatedParams = IE::ParamMap{{IE::VPUX_PARAM_KEY(BLOB_COLOR_FORMAT), colorFormat}};
    _parsedParams.updateFull(updatedParams);

    // Update RemoteMemory information inside allocator
    const auto privateAllocator = std::static_pointer_cast<Allocator>(_allocatorPtr);
    void* newMemoryHandle = privateAllocator->wrapRemoteMemory(_parsedParams.getParamMap());
    if (newMemoryHandle != _memoryHandle) {
        const auto memHandleParams = IE::ParamMap{{IE::VPUX_PARAM_KEY(MEM_HANDLE), newMemoryHandle}};
        _parsedParams.updateFull(memHandleParams);
        _memoryHandle = newMemoryHandle;
    }
}

IE::LockedMemory<void> VPUXRemoteBlob::buffer() noexcept {
    return IE::LockedMemory<void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle,
                                  _parsedParams.getMemoryOffset());
}

IE::LockedMemory<const void> VPUXRemoteBlob::cbuffer() const noexcept {
    return IE::LockedMemory<const void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle,
                                        _parsedParams.getMemoryOffset());
}

IE::LockedMemory<void> VPUXRemoteBlob::rwmap() noexcept {
    return IE::LockedMemory<void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle,
                                  _parsedParams.getMemoryOffset());
}

IE::LockedMemory<const void> VPUXRemoteBlob::rmap() const noexcept {
    return IE::LockedMemory<const void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle,
                                        _parsedParams.getMemoryOffset());
}

IE::LockedMemory<void> VPUXRemoteBlob::wmap() noexcept {
    return IE::LockedMemory<void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle,
                                  _parsedParams.getMemoryOffset());
}

std::string VPUXRemoteBlob::getDeviceName() const noexcept {
    auto remoteContext = _remoteContextPtr.lock();
    if (remoteContext == nullptr) {
        return "";
    }
    return remoteContext->getDeviceName();
}

std::shared_ptr<IE::RemoteContext> VPUXRemoteBlob::getContext() const noexcept {
    return _remoteContextPtr.lock();
}

const std::shared_ptr<IE::IAllocator>& VPUXRemoteBlob::getAllocator() const noexcept {
    return _allocatorPtr;
}

IE::Blob::Ptr VPUXRemoteBlob::createROI(const IE::ROI& roi) const {
    const auto& dims = getTensorDesc().getDims();
    const std::vector<std::size_t> roiBegin = {0, 0, roi.posY, roi.posX};
    const std::vector<std::size_t> roiEnd = {dims[0], dims[1], roi.posY + roi.sizeY, roi.posX + roi.sizeX};
    return createROI(roiBegin, roiEnd);
}

IE::Blob::Ptr VPUXRemoteBlob::createROI(const std::vector<std::size_t>& begin,
                                        const std::vector<std::size_t>& end) const {
    return Blob::Ptr(new VPUXRemoteBlob(*this, begin, end));
}

}  // namespace vpux
