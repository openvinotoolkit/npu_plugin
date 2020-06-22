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

#include "kmb_remote_blob.h"
#include "vpu/kmb_params.hpp"

#include <memory>
#include <string>

using namespace vpu::KmbPlugin;

KmbBlobParams::KmbBlobParams(const InferenceEngine::ParamMap& params, const KmbConfig& config)
    : _paramMap(params), _logger(std::make_shared<Logger>("KmbBlobParams", config.logLevel(), consoleOutput())) {
    if (params.empty()) {
        THROW_IE_EXCEPTION << "KmbBlobParams::KmbBlobParams: Param map for blob is empty.";
    }

    auto remoteMemoryFdIter = params.find(InferenceEngine::KMB_PARAM_KEY(REMOTE_MEMORY_FD));
    if (remoteMemoryFdIter == params.end()) {
        THROW_IE_EXCEPTION << "KmbBlobParams::KmbBlobParams: "
                           << "Param map does not contain remote memory file descriptor "
                              "information";
    }
    try {
        _remoteMemoryFd = remoteMemoryFdIter->second.as<KmbRemoteMemoryFD>();
    } catch (...) {
        THROW_IE_EXCEPTION << "KmbBlobParams::KmbBlobParams: Remote memory fd param has incorrect type information";
    }

    auto colorFormatIter = params.find(InferenceEngine::KMB_PARAM_KEY(COLOR_FORMAT));
    if (colorFormatIter == params.end()) {
        THROW_IE_EXCEPTION << "KmbBlobParams::KmbBlobParams: "
                           << "Param map does not contain color format information";
    }
    try {
        _colorFormat = colorFormatIter->second.as<InferenceEngine::ColorFormat>();
    } catch (...) {
        THROW_IE_EXCEPTION << "KmbBlobParams::KmbBlobParams: Color format param has incorrect type information";
    }
}

KmbRemoteBlob::KmbRemoteBlob(const InferenceEngine::TensorDesc& tensorDesc, const KmbRemoteContext::Ptr& contextPtr,
        const InferenceEngine::ParamMap& params, const KmbConfig& config)
    : RemoteBlob(tensorDesc),
      _params(params, config),
      _remoteContextPtr(contextPtr),
      _config(config),
      _remoteMemoryFd(_params.getRemoteMemoryFD()),
      _colorFormat(_params.getColorFormat()),
      _logger(std::make_shared<Logger>("KmbRemoteBlob", config.logLevel(), consoleOutput())) {
    if (contextPtr == nullptr) {
        THROW_IE_EXCEPTION << "Remote context is null.";
    }
    if (contextPtr->getAllocator() == nullptr) {
        THROW_IE_EXCEPTION << "Remote context does not contain allocator.";
    }

    KmbAllocator::Ptr kmbAllocatorPtr = contextPtr->getAllocator();
    _logger->info("%s: KmbRemoteBlob wrapping %d size\n", __FUNCTION__, static_cast<int>(this->size()));

    _memoryHandle = kmbAllocatorPtr->wrapRemoteMemory(_remoteMemoryFd, this->size());
    if (_memoryHandle == nullptr) {
        THROW_IE_EXCEPTION << "Allocation error";
    }

    _allocatorPtr = kmbAllocatorPtr;
}

bool KmbRemoteBlob::deallocate() noexcept {
    if (_allocatorPtr == nullptr) {
        return false;
    }
    return _allocatorPtr->free(_memoryHandle);
}

InferenceEngine::LockedMemory<void> KmbRemoteBlob::buffer() noexcept {
    return InferenceEngine::LockedMemory<void>(reinterpret_cast<InferenceEngine::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

InferenceEngine::LockedMemory<const void> KmbRemoteBlob::cbuffer() const noexcept {
    return InferenceEngine::LockedMemory<const void>(reinterpret_cast<InferenceEngine::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

InferenceEngine::LockedMemory<void> KmbRemoteBlob::rwmap() noexcept {
    return InferenceEngine::LockedMemory<void>(reinterpret_cast<InferenceEngine::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

InferenceEngine::LockedMemory<const void> KmbRemoteBlob::rmap() const noexcept {
    return InferenceEngine::LockedMemory<const void>(reinterpret_cast<InferenceEngine::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

InferenceEngine::LockedMemory<void> KmbRemoteBlob::wmap() noexcept {
    return InferenceEngine::LockedMemory<void>(reinterpret_cast<InferenceEngine::IAllocator*>(_allocatorPtr.get()), _memoryHandle, 0);
}

std::string KmbRemoteBlob::getDeviceName() const noexcept {
    auto remoteContext = _remoteContextPtr.lock();
    if (remoteContext == nullptr) {
        return "";
    }
    return remoteContext->getDeviceName();
}

std::shared_ptr<InferenceEngine::RemoteContext> KmbRemoteBlob::getContext() const noexcept {
    return _remoteContextPtr.lock();
}

void* KmbRemoteBlob::getHandle() const noexcept { return _memoryHandle; }

const std::shared_ptr<InferenceEngine::IAllocator>& KmbRemoteBlob::getAllocator() const noexcept {
    return _allocatorPtr;
}

static void getImageSize(InferenceEngine::TensorDesc tensorDesc, size_t& outWidth, size_t& outHeight) {
    const auto layout = tensorDesc.getLayout();
    const auto dims = tensorDesc.getDims();
    outHeight = 0;
    outWidth = 0;
    if (layout == InferenceEngine::Layout::NCHW) {
        outHeight = dims.at(2);
        outWidth = dims.at(3);
    } else if (layout == InferenceEngine::Layout::NHWC) {
        outHeight = dims.at(1);
        outWidth = dims.at(2);
    } else {
        THROW_IE_EXCEPTION << "Unsupported layout.";
    }
}

size_t KmbRemoteBlob::size() const noexcept {
    if (_colorFormat == InferenceEngine::ColorFormat::NV12) {
        if (tensorDesc.getLayout() == InferenceEngine::Layout::SCALAR) return 1;
        // FIXME It's a very bad solution
        size_t width, height;
        getImageSize(tensorDesc, width, height);
        return (3 * width * height) / 2;
    } else {
        return MemoryBlob::size();
    }
}

size_t KmbRemoteBlob::byteSize() const noexcept { return size() * element_size(); }
