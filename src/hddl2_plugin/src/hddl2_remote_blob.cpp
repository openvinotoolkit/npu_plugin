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
//      class HDDL2BlobParams Implementation
//------------------------------------------------------------------------------
HDDL2BlobParams::HDDL2BlobParams(const InferenceEngine::ParamMap& params, const HDDL2Config& config)
    : _logger(std::make_shared<Logger>("HDDL2BlobParams", config.logLevel(), consoleOutput())) {
    if (params.empty()) {
        THROW_IE_EXCEPTION << CONFIG_ERROR_str << "Param map for blob is empty.";
    }

    // Check that it's really contains required params
    auto remote_memory_fd_iter = params.find(IE::HDDL2_PARAM_KEY(REMOTE_MEMORY_FD));
    if (remote_memory_fd_iter == params.end()) {
        THROW_IE_EXCEPTION << CONFIG_ERROR_str
                           << "Param map does not contain remote memory file descriptor "
                              "information";
    }
    try {
        _remoteMemoryFd = remote_memory_fd_iter->second.as<RemoteMemoryFD>();
    } catch (...) {
        THROW_IE_EXCEPTION << CONFIG_ERROR_str << "Remote memory fd param have incorrect type information";
    }

    auto color_format_iter = params.find(IE::HDDL2_PARAM_KEY(COLOR_FORMAT));
    if (color_format_iter == params.end()) {
        _logger->warning("Color format information is not found.");
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

InferenceEngine::ParamMap HDDL2BlobParams::getParamMap() const { return _paramMap; }

RemoteMemoryFD HDDL2BlobParams::getRemoteMemoryFD() const { return _remoteMemoryFd; }

InferenceEngine::ColorFormat HDDL2BlobParams::getColorFormat() const { return _colorFormat; }

//------------------------------------------------------------------------------
//      class HDDL2RemoteBlob Implementation
//------------------------------------------------------------------------------
HDDL2RemoteBlob::HDDL2RemoteBlob(const InferenceEngine::TensorDesc& tensorDesc,
    const HDDL2RemoteContext::Ptr& contextPtr, const InferenceEngine::ParamMap& params, const HDDL2Config& config)
    : RemoteBlob(tensorDesc),
      _params(params, config),
      _remoteContextPtr(contextPtr),
      _config(config),
      _logger(std::make_shared<Logger>("HDDL2RemoteBlob", config.logLevel(), consoleOutput())) {
    if (contextPtr == nullptr) {
        THROW_IE_EXCEPTION << CONTEXT_ERROR_str << "Remote context is null.";
    }
    if (contextPtr->getAllocator() == nullptr) {
        THROW_IE_EXCEPTION << CONTEXT_ERROR_str << "Remote context does not contain allocator.";
    }

    HDDL2RemoteAllocator::Ptr hddlAllocatorPtr = contextPtr->getAllocator();
    _logger->info("%s: HDDL2RemoteBlob wrapping %d size\n", __FUNCTION__, static_cast<int>(this->size()));

    _memoryHandle = hddlAllocatorPtr->wrapRemoteMemory(_params.getRemoteMemoryFD(), this->size());
    if (_memoryHandle == nullptr) {
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Allocation error";
    }

    _allocatorPtr = hddlAllocatorPtr;
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

InferenceEngine::ParamMap HDDL2RemoteBlob::getParams() const { return _params.getParamMap(); }

void* HDDL2RemoteBlob::getHandle() const noexcept { return _memoryHandle; }

const std::shared_ptr<InferenceEngine::IAllocator>& HDDL2RemoteBlob::getAllocator() const noexcept {
    return _allocatorPtr;
}

RemoteMemoryFD HDDL2RemoteBlob::getRemoteMemoryFD() const { return _params.getRemoteMemoryFD(); }

InferenceEngine::ColorFormat HDDL2RemoteBlob::getColorFormat() const { return _params.getColorFormat(); }

static void getImageSize(IE::TensorDesc tensorDesc, size_t& outWidth, size_t& outHeight) {
    const auto layout = tensorDesc.getLayout();
    const auto dims = tensorDesc.getDims();
    outHeight = 0;
    outWidth = 0;
    if (layout == IE::Layout::NCHW) {
        outHeight = dims.at(2);
        outWidth = dims.at(3);
    } else if (layout == IE::Layout::NHWC) {
        outHeight = dims.at(1);
        outWidth = dims.at(2);
    } else {
        THROW_IE_EXCEPTION << "Unsupported layout.";
    }
}

size_t HDDL2RemoteBlob::size() const noexcept {
    if (_params.getColorFormat() == IE::ColorFormat::NV12) {
        if (tensorDesc.getLayout() == IE::Layout::SCALAR) return 1;
        // FIXME It's a very bad solution
        size_t width, height;
        getImageSize(tensorDesc, width, height);
        return (3 * width * height) / 2;
    } else {
        return MemoryBlob::size();
    }
}

size_t HDDL2RemoteBlob::byteSize() const noexcept { return size() * element_size(); }
