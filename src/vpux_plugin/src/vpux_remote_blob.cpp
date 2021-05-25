//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

// System
#include <memory>
#include <string>
// Plugin
#include "vpux_exceptions.h"
// [Track number: E#12122]
// TODO Remove this header after removing HDDL2 deprecated parameters in future releases
#include "hddl2/hddl2_params.hpp"
#include "vpux_params_private_options.hpp"
#include "vpux_remote_blob.h"

namespace vpux {
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
VPUXRemoteBlob::VPUXRemoteBlob(const IE::TensorDesc& tensorDesc, const VPUXRemoteContext::Ptr& contextPtr,
                               const std::shared_ptr<Allocator>& allocator, const IE::ParamMap& params,
                               const vpu::LogLevel logLevel)
        : RemoteBlob(tensorDesc),
          _remoteContextPtr(contextPtr),
          _allocatorPtr(allocator),
          _logger(std::make_shared<vpu::Logger>("VPUXRemoteBlob", logLevel, vpu::consoleOutput())),
          _originalTensorDesc(tensorDesc) {
    if (contextPtr == nullptr) {
        IE_THROW() << CONTEXT_ERROR_str << "Remote context is null.";
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

    // ******************************************
    // [Track number: E#12122]
    // TODO Remove this part after removing HDDL2 deprecated parameters in future releases
    if (contextParams.find(IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID)) != contextParams.end()) {
        uint64_t workloadId = contextParams.at(IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID)).as<uint64_t>();
        updatedParams.insert({{IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}});
    }
    // ******************************************

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
        IE_THROW() << "ROI out of blob bounds";
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
        IE_THROW(NotAllocated) << "Failed to set allocator";
    }

    if (tensorDesc.getDims().size() != 4) {
        IE_THROW() << "Unsupported layout for VPUXRemoteBlob";
    }
    const auto origBlobTensorDesc = origBlob.getOriginalTensorDesc();
    const auto orig_W = origBlobTensorDesc.getDims()[3];
    const auto orig_H = origBlobTensorDesc.getDims()[2];
    auto newROI = makeROIOverROI(_parsedParams.getROIPtr(), regionOfInterest, orig_W, orig_H);
    IE::ParamMap updatedROIPtrParam = {{IE::VPUX_PARAM_KEY(ROI_PTR), newROI}};
    _parsedParams.updateFull(updatedROIPtrParam);

    // TODO Remove this cast
    const auto privateAllocator = std::static_pointer_cast<Allocator>(_allocatorPtr);
    _memoryHandle = privateAllocator->wrapRemoteMemory(origBlob.getParams());
    if (_memoryHandle == nullptr) {
        IE_THROW(NotAllocated) << "Failed to copy remote memory handle";
    }
}

void VPUXRemoteBlob::updateColorFormat(const IE::ColorFormat colorFormat) {
    if (_parsedParams.getBlobColorFormat() == colorFormat)
        return;

    auto updatedParams = IE::ParamMap(_parsedParams.getParamMap());
    updatedParams.insert({{IE::VPUX_PARAM_KEY(BLOB_COLOR_FORMAT), colorFormat}});
    _parsedParams.updateFull(updatedParams);

    // Update RemoteMemory information inside allocator
    const auto privateAllocator = std::static_pointer_cast<Allocator>(_allocatorPtr);
    void* newMemoryHandle = privateAllocator->wrapRemoteMemory(updatedParams);
    if (newMemoryHandle != _memoryHandle) {
        IE_THROW() << "Memory handle changed suddenly.";
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

IE::Blob::Ptr VPUXRemoteBlob::createROI(const IE::ROI& regionOfInterest) const {
    return Blob::Ptr(new VPUXRemoteBlob(*this, regionOfInterest));
}
}  // namespace vpux
