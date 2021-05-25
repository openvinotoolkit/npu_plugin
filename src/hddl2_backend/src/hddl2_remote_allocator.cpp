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
#include "vpux/vpux_plugin_params.hpp"
#include "vpux_params_private_options.hpp"
// Subplugin
#include "hddl2_helper.h"
#include "hddl2_remote_allocator.h"
// Low-level
#include "Inference.h"

namespace vpux {
namespace hddl2 {
namespace IE = InferenceEngine;

using namespace vpu;

//------------------------------------------------------------------------------
using matchColorFormats_t = std::unordered_map<int, HddlUnite::eRemoteMemoryFormat>;
static HddlUnite::eRemoteMemoryFormat covertColorFormat(const IE::ColorFormat colorFormat) {
    static const matchColorFormats_t matchColorFormats = {
            {static_cast<int>(IE::ColorFormat::BGR), HddlUnite::eRemoteMemoryFormat::BGR},
            {static_cast<int>(IE::ColorFormat::RGB), HddlUnite::eRemoteMemoryFormat::RGB},
            {static_cast<int>(IE::ColorFormat::NV12), HddlUnite::eRemoteMemoryFormat::NV12}};

    auto format = matchColorFormats.find(colorFormat);
    if (format == matchColorFormats.end()) {
        throw std::logic_error("Color format is not valid.");
    }

    return format->second;
}

static bool isValidRemoteMemory(const HddlUnite::RemoteMemory::Ptr& remoteMemory) {
    // Using local namespace because INVALID_DMABUFFD is macro (-1) and HddlUnite::(-1) is incorrect
    using namespace HddlUnite;
    return remoteMemory->getDmaBufFd() != INVALID_DMABUFFD;
}

static std::string lockOpToStr(const InferenceEngine::LockOp& lockOp) {
    switch (lockOp) {
    case InferenceEngine::LOCK_FOR_READ:
        return "LOCK_FOR_READ";
    case InferenceEngine::LOCK_FOR_WRITE:
        return "LOCK_FOR_WRITE (Read&Write)";
    default:
        return "Unknown Op Mode";
    }
}

HDDL2RemoteMemoryContainer::HDDL2RemoteMemoryContainer(const HddlUnite::RemoteMemory::Ptr& remoteMemory)
        : remoteMemory(remoteMemory) {
}

HDDL2RemoteAllocator::HDDL2RemoteAllocator(const HddlUnite::WorkloadContext::Ptr& contextPtr, const LogLevel logLevel)
        : _logger(std::make_shared<Logger>("RemoteAllocator", logLevel, consoleOutput())) {
    if (contextPtr == nullptr) {
        IE_THROW() << "Context pointer is null";
    }

    _contextPtr = contextPtr;
}

void* HDDL2RemoteAllocator::alloc(size_t size) noexcept {
    UNUSED(size);
    _logger->error("%s: not implemented!\n", __FUNCTION__);
    return nullptr;
}

void* HDDL2RemoteAllocator::wrapRemoteMemory(const InferenceEngine::ParamMap& map) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    // Get blob color format from params
    HddlUnite::eRemoteMemoryFormat colorFormat = HddlUnite::eRemoteMemoryFormat::BGR;
    if (map.find(IE::VPUX_PARAM_KEY(BLOB_COLOR_FORMAT)) != map.end()) {
        colorFormat = covertColorFormat(map.at(IE::VPUX_PARAM_KEY(BLOB_COLOR_FORMAT)));
    }

    // CreateROI case - if we've already wrapped memory with this handle
    // we need to find it and increase counter
    if (map.find(IE::VPUX_PARAM_KEY(MEM_HANDLE)) != map.end()) {
        const auto memoryHandle = static_cast<void*>(map.at(IE::VPUX_PARAM_KEY(MEM_HANDLE)));
        return incrementRemoteMemoryCounter(memoryHandle, colorFormat);
    }

    // Get remote memory FD from params
    VpuxRemoteMemoryFD remoteMemoryFD = -1;
    try {
        remoteMemoryFD = getRemoteMemoryFDFromParams(map);
    } catch (const std::exception& ex) {
        _logger->error("Failed to get remote memory FD! {}", ex.what());
        return nullptr;
    }

    if (remoteMemoryFD < 0) {
        return nullptr;
    }

    // Common case - if we've already wrapped memory with this FD
    // (for example - NV12 blob with common RemoteMemory + Y/UV offsets)
    // we need to find it and increase counter
    for (auto memoryIter = _memoryStorage.cbegin(); memoryIter != _memoryStorage.cend(); ++memoryIter) {
        if (remoteMemoryFD == memoryIter->second.remoteMemory->getDmaBufFd()) {
            return incrementRemoteMemoryCounter(memoryIter->first, colorFormat);
        }
    }

    // Common case - we haven't wrapped yet memory with this FD
    // We need to create RemoteMemory object with current FD
    std::shared_ptr<IE::TensorDesc> tensorDesc = nullptr;
    try {
        tensorDesc = getOriginalTensorDescFromParams(map);
    } catch (const std::exception& ex) {
        _logger->error("Failed to get original tensor descriptor! {}", ex.what());
        return nullptr;
    }

    HddlUnite::WorkloadContext::Ptr remoteContext = nullptr;
    try {
        WorkloadID workloadId = getWorkloadIDFromParams(map);
        remoteContext = HddlUnite::queryWorkloadContext(workloadId);
    } catch (const std::exception& ex) {
        _logger->error("Failed to get workload context! {}", ex.what());
        return nullptr;
    }

    if (remoteContext == nullptr) {
        return nullptr;
    }

    HddlUnite::RemoteMemory::Ptr remoteMemory = nullptr;
    const auto& strides = tensorDesc->getBlockingDesc().getStrides();
    const auto& dims = tensorDesc->getDims();
    if (dims.size() != 4) {
        _logger->error("Remote allocator - layouts with dims != 4 are not supported!");
        return nullptr;
    }
    if (strides.empty()) {
        const auto elementSize = tensorDesc->getPrecision().size();
        const size_t size =
                elementSize * std::accumulate(std::begin(dims), std::end(dims), (size_t)1, std::multiplies<size_t>());
        HddlUnite::RemoteMemoryDesc memoryDesc(size, 1, size, 1);
        remoteMemory = std::make_shared<HddlUnite::RemoteMemory>(*remoteContext, memoryDesc, remoteMemoryFD);
    } else {
        uint32_t mWidth = dims[3];
        uint32_t mHeight = dims[2];
        bool isNV12Blob = (colorFormat == HddlUnite::eRemoteMemoryFormat::NV12);
        const bool isNCHW = isNV12Blob ? false : (tensorDesc->getLayout() == IE::Layout::NCHW);
        uint32_t mWidthStride = strides[isNCHW ? 2 : 1];
        uint32_t mHeightStride = strides[isNCHW ? 1 : 0] / mWidthStride;
        HddlUnite::RemoteMemoryDesc memoryDesc(mWidth, mHeight, mWidthStride, mHeightStride, colorFormat);
        remoteMemory = std::make_shared<HddlUnite::RemoteMemory>(*remoteContext, memoryDesc, remoteMemoryFD);
    }

    if (!isValidRemoteMemory(remoteMemory)) {
        _logger->warning("%s: Incorrect memory fd!\n", __FUNCTION__);
        return nullptr;
    }

    try {
        // Use already allocated memory
        HDDL2RemoteMemoryContainer memoryContainer(remoteMemory);
        void* remMemHandle = static_cast<void*>(remoteMemory.get());
        _memoryStorage.emplace(remMemHandle, memoryContainer);
        ++_memoryHandleCounter[remMemHandle];

        _logger->info("%s: Wrapped memory of %lu size\n", __FUNCTION__, remoteMemory->getMemoryDesc().getDataSize());
        return static_cast<void*>(remoteMemory.get());
    } catch (const std::exception& ex) {
        _logger->error("%s: Failed to wrap memory. Error: %s\n", __FUNCTION__, ex.what());
        return nullptr;
    }
}

void* HDDL2RemoteAllocator::incrementRemoteMemoryCounter(void* remoteMemoryHandle,
                                                         const HddlUnite::eRemoteMemoryFormat format) noexcept {
    if (remoteMemoryHandle == nullptr) {
        _logger->warning("%s: Invalid address: %p \n", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    auto counter_it = _memoryHandleCounter.find(const_cast<void*>(remoteMemoryHandle));
    if (counter_it == _memoryHandleCounter.end()) {
        _logger->warning("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    HddlUnite::RemoteMemory* remoteMemory = static_cast<HddlUnite::RemoteMemory*>(remoteMemoryHandle);
    if (remoteMemory == nullptr) {
        _logger->warning("%s: Invalid cast to HddlUnite::RemoteMemory: %p \n", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    if (remoteMemory->getMemoryDesc().m_format != format) {
        HddlUnite::RemoteMemoryDesc updatedMemoryDesc = remoteMemory->getMemoryDesc();
        updatedMemoryDesc.m_format = format;

        auto status = remoteMemory->update(*remoteMemory->getWorkloadContext(), updatedMemoryDesc,
                                           remoteMemory->getDmaBufFd());
        if (status != HDDL_OK) {
            _logger->warning("%s: Updating remote memory error: %p \n", __FUNCTION__, remoteMemoryHandle);
            return nullptr;
        }
    }

    ++_memoryHandleCounter[const_cast<void*>(remoteMemoryHandle)];
    return const_cast<void*>(remoteMemoryHandle);
}

size_t HDDL2RemoteAllocator::decrementRemoteMemoryCounter(void* remoteMemoryHandle, bool& findMemoryHandle) noexcept {
    auto counter_it = _memoryHandleCounter.find(remoteMemoryHandle);
    if (counter_it == _memoryHandleCounter.end()) {
        findMemoryHandle = false;
        return 0;
    }

    if (!counter_it->second) {
        findMemoryHandle = false;
        return 0;
    }

    findMemoryHandle = true;
    auto ret_counter = --(counter_it->second);
    if (!ret_counter) {
        _memoryHandleCounter.erase(counter_it);
    }
    return ret_counter;
}

bool HDDL2RemoteAllocator::free(void* remoteMemoryHandle) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    if (remoteMemoryHandle == nullptr) {
        _logger->warning("%s: Invalid address: %p \n", __FUNCTION__, remoteMemoryHandle);
        return false;
    }
    auto iterator = _memoryStorage.find(remoteMemoryHandle);
    if (iterator == _memoryStorage.end()) {
        _logger->warning("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return false;
    }

    auto memory = &iterator->second;
    if (memory->isLocked) {
        _logger->warning("%s: Memory %p is locked!\n", __FUNCTION__, remoteMemoryHandle);
        return false;
    }

    bool findMemoryHandle;
    auto handle_counter = decrementRemoteMemoryCounter(remoteMemoryHandle, findMemoryHandle);
    if (!findMemoryHandle) {
        _logger->warning("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return false;
    }

    if (handle_counter) {
        _logger->info("%s: Memory %p found, remaining references = %lu\n", __FUNCTION__, remoteMemoryHandle,
                      handle_counter);
        return true;
    }

    _logger->info("%s: Memory %p found, removing element\n", __FUNCTION__, remoteMemoryHandle);
    _memoryStorage.erase(iterator);
    return true;
}

// TODO LOCK_FOR_READ behavior when we will have lock for read-write
/**
 * LOCK_FOR_READ - do not sync to device on this call
 * LOCK_FOR_WRITE - default behavior - read&write option
 */
void* HDDL2RemoteAllocator::lock(void* remoteMemoryHandle, InferenceEngine::LockOp lockOp) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    auto iterator = _memoryStorage.find(remoteMemoryHandle);
    if (iterator == _memoryStorage.end()) {
        _logger->warning("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    _logger->info("%s: Locking memory %p \n", __FUNCTION__, remoteMemoryHandle);

    auto memory = &iterator->second;

    if (memory->isLocked) {
        _logger->warning("%s: Memory %p is already locked!\n", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    memory->isLocked = true;
    memory->lockOp = lockOp;

    const size_t dmaBufSize = memory->remoteMemory->getMemoryDesc().getDataSize();
    memory->localMemory.resize(dmaBufSize);

    if (dmaBufSize != memory->localMemory.size()) {
        _logger->info("%s: dmaBufSize(%d) != memory->size(%d)\n", __FUNCTION__, static_cast<int>(dmaBufSize),
                      static_cast<int>(memory->localMemory.size()));
        return nullptr;
    }

    _logger->info("%s: LockOp: %s\n", __FUNCTION__, lockOpToStr(lockOp).c_str());

    // TODO Do this step only on R+W and R operations, not for Write
    _logger->info("%s: Sync %d memory from device, remoteMemoryHandle %p, fd %d\n", __FUNCTION__,
                  static_cast<int>(memory->localMemory.size()), remoteMemoryHandle,
                  memory->remoteMemory->getDmaBufFd());

    HddlStatusCode statusCode =
            memory->remoteMemory->syncFromDevice(memory->localMemory.data(), memory->localMemory.size());
    if (statusCode != HDDL_OK) {
        memory->isLocked = false;
        return nullptr;
    }

    return memory->localMemory.data();
}

void HDDL2RemoteAllocator::unlock(void* remoteMemoryHandle) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    auto iterator = _memoryStorage.find(remoteMemoryHandle);
    if (iterator == _memoryStorage.end() || !iterator->second.isLocked) {
        _logger->warning("%s: Memory %p is not found!\n", __FUNCTION__, remoteMemoryHandle);
        return;
    }
    auto memory = &iterator->second;

    if (memory->lockOp == InferenceEngine::LOCK_FOR_WRITE) {
        // Sync memory to device
        _logger->info("%s: Sync %d memory to device, remoteMemoryHandle %p\n", __FUNCTION__,
                      static_cast<int>(memory->localMemory.size()), remoteMemoryHandle);
        memory->remoteMemory->syncToDevice(memory->localMemory.data(), memory->localMemory.size());
    } else {
        _logger->warning("%s: LOCK_FOR_READ, Memory %d will NOT be synced, remoteMemoryHandle %p\n", __FUNCTION__,
                         static_cast<int>(memory->localMemory.size()), remoteMemoryHandle);
    }

    memory->isLocked = false;
}

void* HDDL2RemoteAllocator::wrapRemoteMemoryHandle(const int& /*remoteMemoryFd*/, const size_t /*size*/,
                                                   void* /*memHandle*/) noexcept {
    _logger->error("Not implemented");
    return nullptr;
}

void* HDDL2RemoteAllocator::wrapRemoteMemoryOffset(const int& /*remoteMemoryFd*/, const size_t /*size*/,
                                                   const size_t& /*memOffset*/) noexcept {
    _logger->error("Not implemented");
    return nullptr;
}

unsigned long HDDL2RemoteAllocator::getPhysicalAddress(void* handle) noexcept {
    UNUSED(handle);
    return 0;
}

}  // namespace hddl2
}  // namespace vpux
