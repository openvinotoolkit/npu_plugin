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

#include "hddl2_remote_blob.h"

#include <memory>
#include <string>

#include "hddl2_exceptions.h"
#include "hddl2_params.hpp"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

static void checkSupportedColorFormat(const IE::ColorFormat& colorFormat) {
    switch (colorFormat) {
    case InferenceEngine::NV12:
    case InferenceEngine::BGR:
        return;
    case InferenceEngine::RAW:
    case InferenceEngine::RGB:
    case InferenceEngine::RGBX:
    case InferenceEngine::BGRX:
    case InferenceEngine::I420:
        THROW_IE_EXCEPTION << "Unsupported color format.";
    }
}

//------------------------------------------------------------------------------
HDDL2BlobParams::HDDL2BlobParams(const InferenceEngine::ParamMap& params, const vpu::HDDL2Config& config)
    : _logger(std::make_shared<Logger>("HDDL2BlobParams", config.logLevel(), consoleOutput())) {
    if (params.empty()) {
        THROW_IE_EXCEPTION << CONFIG_ERROR_str << "Param map for blob is empty.";
    }

    // Check that it's really contains required params
    auto remote_memory_iter = params.find(IE::HDDL2_PARAM_KEY(REMOTE_MEMORY));
    if (remote_memory_iter == params.end()) {
        THROW_IE_EXCEPTION << CONFIG_ERROR_str
                           << "Param map does not contain remote memory file descriptor "
                              "information";
    }
    try {
        _remoteMemory = remote_memory_iter->second.as<HddlUnite::SMM::RemoteMemory::Ptr>();
    } catch (...) {
        THROW_IE_EXCEPTION << CONFIG_ERROR_str << "Remote memory param have incorrect type information";
    }

    auto color_format_iter = params.find(IE::HDDL2_PARAM_KEY(COLOR_FORMAT));
    if (color_format_iter == params.end()) {
        _logger->info("Color format information is not found. Default BGR will be used.");
        _colorFormat = IE::ColorFormat::BGR;
    } else {
        try {
            _colorFormat = color_format_iter->second.as<IE::ColorFormat>();
            checkSupportedColorFormat(_colorFormat);
        } catch (...) {
            THROW_IE_EXCEPTION << CONFIG_ERROR_str << "Color format param have incorrect type information";
        }
    }

    _paramMap = params;
}

//------------------------------------------------------------------------------
HDDL2RemoteBlob::HDDL2RemoteBlob(const InferenceEngine::TensorDesc& tensorDesc,
    const HDDL2RemoteContext::Ptr& contextPtr, const InferenceEngine::ParamMap& params, const vpu::HDDL2Config& config)
    : RemoteBlob(tensorDesc),
      _params(params, config),
      _remoteContextPtr(contextPtr),
      _config(config),
      _remoteMemory(_params.getRemoteMemory()),
      _colorFormat(_params.getColorFormat()),
      _roiPtr(nullptr),
      _logger(std::make_shared<Logger>("HDDL2RemoteBlob", config.logLevel(), consoleOutput())) {
    if (contextPtr == nullptr) {
        THROW_IE_EXCEPTION << CONTEXT_ERROR_str << "Remote context is null.";
    }
    if (contextPtr->getAllocator() == nullptr) {
        THROW_IE_EXCEPTION << CONTEXT_ERROR_str << "Remote context does not contain allocator.";
    }

    HDDL2RemoteAllocator::Ptr hddlAllocatorPtr = contextPtr->getAllocator();
    _logger->info("%s: HDDL2RemoteBlob wrapping %d size\n", __FUNCTION__, static_cast<int>(this->size()));

    _memoryHandle = hddlAllocatorPtr->wrapRemoteMemory(_remoteMemory, this->size());
    if (_memoryHandle == nullptr) {
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Allocation error";
    }

    _allocatorPtr = hddlAllocatorPtr;
}

HDDL2RemoteBlob::HDDL2RemoteBlob(const HDDL2RemoteBlob& origBlob, const InferenceEngine::ROI& regionOfInterest)
    : RemoteBlob(origBlob.getTensorDesc()),
      _params(origBlob._params),
      _remoteContextPtr(origBlob._remoteContextPtr),
      _allocatorPtr(origBlob._allocatorPtr),
      _config(origBlob._config),
      _remoteMemory(origBlob._remoteMemory),
      _colorFormat(origBlob._colorFormat),
      _logger(std::make_shared<Logger>("HDDL2RemoteBlob", origBlob._config.logLevel(), consoleOutput())) {
    if (_allocatorPtr == nullptr) {
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set allocator";
    }

    _memoryHandle = std::static_pointer_cast<HDDL2RemoteAllocator>(_allocatorPtr)
                        ->incrementRemoteMemoryCounter(origBlob._memoryHandle);
    if (_memoryHandle == nullptr) {
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to copy remote memory handle";
    }

    auto origROIPtr = origBlob.getROIPtr();
    if (origROIPtr) {
        InferenceEngine::ROI newROI = regionOfInterest;
        newROI.posX += origROIPtr->posX;
        newROI.posY += origROIPtr->posY;
        _roiPtr = std::make_shared<IE::ROI>(newROI);
    } else {
        _roiPtr = std::make_shared<IE::ROI>(regionOfInterest);
    }

    if (tensorDesc.getDims().size() < 4) {
        THROW_IE_EXCEPTION << "Unsupported layout for HDDL2RemoteBlob";
    }

    const auto W = tensorDesc.getDims()[3];
    const auto H = tensorDesc.getDims()[2];

    if ((_roiPtr->posX + _roiPtr->sizeX > W) || (_roiPtr->posY + _roiPtr->sizeY > H)) {
        THROW_IE_EXCEPTION << "ROI out of blob bounds";
    }
}

bool HDDL2RemoteBlob::deallocate() noexcept {
    if (_allocatorPtr == nullptr) {
        return false;
    }
    return _allocatorPtr->free(_memoryHandle);
}

InferenceEngine::LockedMemory<void> HDDL2RemoteBlob::buffer() noexcept {
    return IE::LockedMemory<void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

InferenceEngine::LockedMemory<const void> HDDL2RemoteBlob::cbuffer() const noexcept {
    return IE::LockedMemory<const void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

InferenceEngine::LockedMemory<void> HDDL2RemoteBlob::rwmap() noexcept {
    return IE::LockedMemory<void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

InferenceEngine::LockedMemory<const void> HDDL2RemoteBlob::rmap() const noexcept {
    return IE::LockedMemory<const void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

InferenceEngine::LockedMemory<void> HDDL2RemoteBlob::wmap() noexcept {
    return IE::LockedMemory<void>(reinterpret_cast<IE::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

std::string HDDL2RemoteBlob::getDeviceName() const noexcept {
    auto remoteContext = _remoteContextPtr.lock();
    if (remoteContext == nullptr) {
        return "";
    }
    return remoteContext->getDeviceName();
}

std::shared_ptr<InferenceEngine::RemoteContext> HDDL2RemoteBlob::getContext() const noexcept {
    return _remoteContextPtr.lock();
}

void* HDDL2RemoteBlob::getHandle() const noexcept { return _memoryHandle; }

const std::shared_ptr<InferenceEngine::IAllocator>& HDDL2RemoteBlob::getAllocator() const noexcept {
    return _allocatorPtr;
}

size_t HDDL2RemoteBlob::size() const noexcept {
    if (_colorFormat == IE::ColorFormat::NV12) {
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

size_t HDDL2RemoteBlob::byteSize() const noexcept { return size() * element_size(); }

InferenceEngine::Blob::Ptr HDDL2RemoteBlob::createROI(const InferenceEngine::ROI& regionOfInterest) const {
    return Blob::Ptr(new HDDL2RemoteBlob(*this, regionOfInterest));
}
