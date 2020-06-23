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
        THROW_IE_EXCEPTION << "KmbBlobParams::KmbBlobParams: Remote memory fd param has incorrect type";
    }

    auto remoteMemoryHandleIter = params.find(InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE));
    if (remoteMemoryHandleIter == params.end()) {
        THROW_IE_EXCEPTION << "KmbBlobParams::KmbBlobParams: "
                           << "Param map does not contain remote memory handle information";
    }
    try {
        _remoteMemoryHandle = remoteMemoryHandleIter->second.as<KmbHandleParam>();
    } catch (...) {
        THROW_IE_EXCEPTION << "KmbBlobParams::KmbBlobParams: Remote memory handle param has incorrect type";
    }
}

KmbRemoteBlob::KmbRemoteBlob(const InferenceEngine::TensorDesc& tensorDesc, const KmbRemoteContext::Ptr& contextPtr,
        const InferenceEngine::ParamMap& params, const KmbConfig& config)
    : RemoteBlob(tensorDesc),
      _params(params, config),
      _remoteContextPtr(contextPtr),
      _config(config),
      _remoteMemoryFd(_params.getRemoteMemoryFD()),
      _logger(std::make_shared<Logger>("KmbRemoteBlob", config.logLevel(), consoleOutput())) {
    if (contextPtr == nullptr) {
        THROW_IE_EXCEPTION << "Remote context is null.";
    }
    if (contextPtr->getAllocator() == nullptr) {
        THROW_IE_EXCEPTION << "Remote context does not contain allocator.";
    }

    KmbAllocator::Ptr kmbAllocatorPtr = contextPtr->getAllocator();
    _logger->info("%s: KmbRemoteBlob wrapping %d size\n", __FUNCTION__, static_cast<int>(this->size()));

    _memoryHandle = kmbAllocatorPtr->wrapRemoteMemory(_remoteMemoryFd, this->size(), _params.getRemoteMemoryHandle());
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

size_t KmbRemoteBlob::size() const noexcept {
    return MemoryBlob::size();
}

size_t KmbRemoteBlob::byteSize() const noexcept { return size() * element_size(); }
