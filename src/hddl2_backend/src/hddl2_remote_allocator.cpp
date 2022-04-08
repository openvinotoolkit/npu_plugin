//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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

//------------------------------------------------------------------------------
using matchColorFormats_t = std::unordered_map<int, HddlUnite::eRemoteMemoryFormat>;
static HddlUnite::eRemoteMemoryFormat convertColorFormat(const IE::ColorFormat colorFormat) {
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

static bool checkDims(const IE::SizeVector& dims, Logger logger) {
    if (dims.size() != 4) {
        logger.error("{0}: Layouts with dims != 4 are not supported!", __FUNCTION__);
        return false;
    }

    for (const auto& dim : dims) {
        if (!dim) {
            logger.error("{0}: Zero dimension", __FUNCTION__);
            return false;
        } else if (dim > std::numeric_limits<uint32_t>::max()) {
            logger.trace("{0}: Enormous dimension. Preventing overflow", __FUNCTION__);
            return false;
        }
    }

    return true;
}

static bool checkStrides(const IE::SizeVector& strides, Logger logger) {
    if (strides.empty()) {
        return true;
    }
    if (strides.size() != 4) {
        logger.error("{0}: Strides with dims != 4 are not supported!", __FUNCTION__);
        return false;
    }

    for (auto strideIt = strides.cbegin(); strideIt != strides.cend(); ++strideIt) {
        const auto curStride = *strideIt;
        if (!curStride) {
            logger.error("{0}: Zero stride", __FUNCTION__);
            return false;
        } else if (curStride > std::numeric_limits<uint32_t>::max()) {
            logger.error("{0}: Enormous stride. Preventing overflow", __FUNCTION__);
            return false;
        } else if (strideIt + 1 != strides.cend()) {
            const auto nextStride = *(strideIt + 1);
            if (curStride < nextStride) {
                logger.error("{0}: Incorrect strides values", __FUNCTION__);
                return false;
            }
        }
    }

    return true;
}

HDDL2RemoteMemoryContainer::HDDL2RemoteMemoryContainer(const HddlUnite::RemoteMemory::Ptr& remoteMemory)
        : _remoteMemory(remoteMemory), _updatedMemoryHandle(remoteMemory.get()) {
}

HDDL2RemoteAllocator::HDDL2RemoteAllocator(const HddlUnite::WorkloadContext::Ptr& contextPtr, const LogLevel logLevel)
        : _logger("RemoteAllocator", logLevel) {
    if (contextPtr == nullptr) {
        IE_THROW() << "Context pointer is null";
    }

    _contextPtr = contextPtr;
}

void* HDDL2RemoteAllocator::alloc(size_t /*size*/) noexcept {
    _logger.error("{0}: Not implemented!", __FUNCTION__);
    return nullptr;
}

// TODO Simplify this method - add some small helpers for every logic block
// [Track number: E#15988]
void* HDDL2RemoteAllocator::wrapRemoteMemory(const InferenceEngine::ParamMap& map) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    // We are having the following workflow for VPUXRemoteBlob:
    // 1. Create Remote Blob
    // 2. Create ROI Remote Blob (optional) from Remote Blob
    // 3. Create Infer Request and set preprocessing for Remote Blob (optional)
    // 4. Update color format for Remote Blob (if PreprocessInfo is set)
    const auto updateColorFormat = map.find(IE::VPUX_PARAM_KEY(BLOB_COLOR_FORMAT)) != map.end();
    const auto isMemHandleDefined = map.find(IE::VPUX_PARAM_KEY(MEM_HANDLE)) != map.end();
    void* memoryHandle = isMemHandleDefined ? static_cast<void*>(map.at(IE::VPUX_PARAM_KEY(MEM_HANDLE))) : nullptr;
    const auto createROI = memoryHandle && !updateColorFormat;

    // Create ROI case - if we've already wrapped memory with this handle
    // we need to find it and increase the counter
    if (createROI) {
        return incrementRemoteMemoryCounter(memoryHandle);
    }

    // Get blob color format from params
    const auto colorFormat = updateColorFormat ? convertColorFormat(map.at(IE::VPUX_PARAM_KEY(BLOB_COLOR_FORMAT)))
                                               : HddlUnite::eRemoteMemoryFormat::BGR;

    // Color format updating can be done only for the blobs with wrapped remote memory
    if (updateColorFormat && !memoryHandle) {
        _logger.error("{0}: Updating color format with null memory pointer", __FUNCTION__);
        return nullptr;
    }

    // Update color format - check if color format is the same, nothing is required
    if (updateColorFormat) {
        const auto remoteMem = static_cast<HddlUnite::RemoteMemory*>(memoryHandle);
        if (!remoteMem) {
            _logger.error("{0}: Failed to cast pointer to RemoteMemory object", __FUNCTION__);
            return nullptr;
        }
        const auto currentColorFormat = remoteMem->getMemoryDesc().m_format;
        if (currentColorFormat == colorFormat) {
            return memoryHandle;
        }
    }

    // Get remote memory FD from params
    VpuxRemoteMemoryFD remoteMemoryFD = -1;
    try {
        remoteMemoryFD = getRemoteMemoryFDFromParams(map);
    } catch (const std::exception& ex) {
        _logger.error("Failed to get remote memory FD! {0}", ex.what());
        return nullptr;
    }

    if (remoteMemoryFD < 0) {
        _logger.error("{0}: Invalid remote memory FD - {1}", __FUNCTION__, remoteMemoryFD);
        return nullptr;
    }

    // Common case - if we've already wrapped memory with this FD
    // (for example - NV12 blob with common RemoteMemory + Y/UV offsets)
    // we need to find it and increase counter
    if (!updateColorFormat) {
        const auto memoryIter = std::find_if(
                _memoryStorage.cbegin(), _memoryStorage.cend(),
                [remoteMemoryFD](const std::pair<void*, vpux::hddl2::HDDL2RemoteMemoryContainer>& elem) -> bool {
                    return (remoteMemoryFD == elem.second._remoteMemory->getDmaBufFd());
                });
        if (memoryIter != _memoryStorage.end()) {
            return incrementRemoteMemoryCounter(memoryIter->first);
        }
    }

    // Else - first wrapping or re-wrapping (update color format)

    // Get TensorDesc from remote blob params
    std::shared_ptr<IE::TensorDesc> tensorDesc = nullptr;
    try {
        tensorDesc = getOriginalTensorDescFromParams(map);
    } catch (const std::exception& ex) {
        _logger.error("Failed to get original tensor descriptor! {0}", ex.what());
        return nullptr;
    }

    // Get remote context from remote blob params
    HddlUnite::WorkloadContext::Ptr remoteContext = nullptr;
    try {
        WorkloadID workloadId = getWorkloadIDFromParams(map);
        remoteContext = HddlUnite::queryWorkloadContext(workloadId);
    } catch (const std::exception& ex) {
        _logger.error("Failed to get workload context! {0}", ex.what());
        return nullptr;
    }

    if (remoteContext == nullptr) {
        _logger.error("{0}: Workload context null pointer!", __FUNCTION__);
        return nullptr;
    }

    if (updateColorFormat) {
        // Check if we already created RemoteMemory with updated color format
        // (for example, Y blob, and now it's UV blob)
        // In this case we should use existing RemoteMemory
        const auto findRemMemIt =
                std::find_if(_memoryStorage.cbegin(), _memoryStorage.cend(),
                             [&](const std::pair<void*, vpux::hddl2::HDDL2RemoteMemoryContainer>& elem) -> bool {
                                 return (elem.first == memoryHandle && elem.first != elem.second._updatedMemoryHandle);
                             });
        if (findRemMemIt != _memoryStorage.end()) {
            const auto currentHandle = findRemMemIt->first;
            const auto updatedHandle = findRemMemIt->second._updatedMemoryHandle;
            // Remove previous handle
            if (!freeMemory(currentHandle)) {
                return nullptr;
            }
            // Use updated handle
            return incrementRemoteMemoryCounter(updatedHandle);
        }
    }

    // We need to create RemoteMemory object with provided FD
    HddlUnite::RemoteMemory::Ptr remoteMemory = nullptr;
    const auto& strides = tensorDesc->getBlockingDesc().getStrides();
    const auto& dims = tensorDesc->getDims();
    if (!checkDims(dims, _logger) || !checkStrides(strides, _logger)) {
        return nullptr;
    }

    if (strides.empty()) {
        const auto elementSize = tensorDesc->getPrecision().size();
        const size_t size = elementSize * std::accumulate(dims.cbegin(), dims.cend(), static_cast<size_t>(1),
                                                          std::multiplies<size_t>());
        if (size > std::numeric_limits<uint32_t>::max()) {
            _logger.error("{0}: Enormous blob size. Preventing overflow", __FUNCTION__);
            return nullptr;
        }
        const uint32_t uniteSize = static_cast<uint32_t>(size);
        HddlUnite::RemoteMemoryDesc memoryDesc(uniteSize, 1, uniteSize, 1);
        remoteMemory = std::make_shared<HddlUnite::RemoteMemory>(*remoteContext, memoryDesc, remoteMemoryFD);
    } else {
        const uint32_t mWidth = static_cast<uint32_t>(dims[3]);
        const uint32_t mHeight = static_cast<uint32_t>(dims[2]);
        const bool isNV12Blob = colorFormat == HddlUnite::eRemoteMemoryFormat::NV12;
        const bool isNCHW = isNV12Blob ? false : tensorDesc->getLayout() == IE::Layout::NCHW;
        const uint32_t mWidthStride = isNCHW ? strides[2] : strides[1] / strides[2];
        const uint32_t mHeightStride = strides[isNCHW ? 1 : 0] / mWidthStride;
        HddlUnite::RemoteMemoryDesc memoryDesc(mWidth, mHeight, mWidthStride, mHeightStride, colorFormat);
        remoteMemory = std::make_shared<HddlUnite::RemoteMemory>(*remoteContext, memoryDesc, remoteMemoryFD);
    }

    if (!isValidRemoteMemory(remoteMemory)) {
        _logger.warning("{0}: Incorrect memory fd!", __FUNCTION__);
        return nullptr;
    }

    try {
        // Use already allocated memory
        HDDL2RemoteMemoryContainer memoryContainer(remoteMemory);
        void* remMemHandle = static_cast<void*>(remoteMemory.get());
        _memoryStorage.emplace(remMemHandle, memoryContainer);
        ++_memoryHandleCounter[remMemHandle];

        _logger.info("{0}: Wrapped memory of {1} size", __FUNCTION__, remoteMemory->getMemoryDesc().getDataSize());
        if (updateColorFormat) {
            if (memoryHandle == nullptr) {
                _logger.error("{0}: Null pointer to remote memory", __FUNCTION__);
                return nullptr;
            }
            // We have to remove previous remote memory in updating color format case
            if (!freeMemory(memoryHandle)) {
                return nullptr;
            }
            // We have to update memory storage information - to mark updated handle for all remaining blobs
            // which are still using previous handle (with old color format)
            auto memStorageIt = _memoryStorage.find(memoryHandle);
            if (memStorageIt != _memoryStorage.end()) {
                memStorageIt->second._updatedMemoryHandle = remMemHandle;
            }
        }
        return static_cast<void*>(remoteMemory.get());
    } catch (const std::exception& ex) {
        _logger.error("{0}: Failed to wrap memory. Error: {1}", __FUNCTION__, ex.what());
        return nullptr;
    }
}

void* HDDL2RemoteAllocator::incrementRemoteMemoryCounter(void* remoteMemoryHandle) noexcept {
    if (remoteMemoryHandle == nullptr) {
        _logger.warning("{0}: Invalid address: {1}", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    auto counter_it = _memoryHandleCounter.find(const_cast<void*>(remoteMemoryHandle));
    if (counter_it == _memoryHandleCounter.end()) {
        _logger.warning("{0}: Memory {1} is not found!", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    HddlUnite::RemoteMemory* remoteMemory = static_cast<HddlUnite::RemoteMemory*>(remoteMemoryHandle);
    if (remoteMemory == nullptr) {
        _logger.warning("{0}: Invalid cast to HddlUnite::RemoteMemory: {1}", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
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
        _logger.info("{0}: Remove memory {1}", __FUNCTION__, remoteMemoryHandle);
        _memoryHandleCounter.erase(counter_it);
    }
    return ret_counter;
}

bool HDDL2RemoteAllocator::freeMemory(void* remoteMemoryHandle) noexcept {
    if (remoteMemoryHandle == nullptr) {
        _logger.warning("{0}: Invalid address: {1}", __FUNCTION__, remoteMemoryHandle);
        return false;
    }
    auto iterator = _memoryStorage.find(remoteMemoryHandle);
    if (iterator == _memoryStorage.end()) {
        _logger.warning("{0}: Memory {1} is not found!", __FUNCTION__, remoteMemoryHandle);
        return false;
    }

    auto memory = &iterator->second;
    if (memory->_isLocked) {
        _logger.warning("{0}: Memory {1} is locked!", __FUNCTION__, remoteMemoryHandle);
        return false;
    }

    bool findMemoryHandle;
    auto handle_counter = decrementRemoteMemoryCounter(remoteMemoryHandle, findMemoryHandle);
    if (!findMemoryHandle) {
        _logger.warning("{0}: Memory {1} is not found!", __FUNCTION__, remoteMemoryHandle);
        return false;
    }

    if (handle_counter) {
        _logger.info("{0}: Memory {1} found, remaining references = {2}", __FUNCTION__, remoteMemoryHandle,
                     handle_counter);
        return true;
    }

    _logger.info("{0}: Memory {1} found, removing element", __FUNCTION__, remoteMemoryHandle);
    _memoryStorage.erase(iterator);
    return true;
}

bool HDDL2RemoteAllocator::free(void* remoteMemoryHandle) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);
    return freeMemory(remoteMemoryHandle);
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
        _logger.warning("{0}: Memory {1} is not found!", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    _logger.info("{0}: Locking memory {1}", __FUNCTION__, remoteMemoryHandle);

    auto memory = &iterator->second;

    if (memory->_isLocked) {
        _logger.warning("{0}: Memory {1} is already locked!", __FUNCTION__, remoteMemoryHandle);
        return nullptr;
    }

    memory->_isLocked = true;
    memory->_lockOp = lockOp;

    const size_t dmaBufSize = memory->_remoteMemory->getMemoryDesc().getDataSize();
    memory->_localMemory.resize(dmaBufSize);

    if (dmaBufSize != memory->_localMemory.size()) {
        _logger.info("{0}: dmaBufSize({1}) != memory->size({2})", __FUNCTION__, dmaBufSize,
                     memory->_localMemory.size());
        return nullptr;
    }

    _logger.info("{0}: LockOp: {1}", __FUNCTION__, lockOpToStr(lockOp));

    // TODO Do this step only on R+W and R operations, not for Write
    _logger.info("{0}: Sync {1} memory from device, remoteMemoryHandle {2}, fd {3}", __FUNCTION__,
                 memory->_localMemory.size(), remoteMemoryHandle, memory->_remoteMemory->getDmaBufFd());

    HddlStatusCode statusCode =
            memory->_remoteMemory->syncFromDevice(memory->_localMemory.data(), memory->_localMemory.size());
    if (statusCode != HDDL_OK) {
        memory->_isLocked = false;
        return nullptr;
    }

    return memory->_localMemory.data();
}

void HDDL2RemoteAllocator::unlock(void* remoteMemoryHandle) noexcept {
    std::lock_guard<std::mutex> lock(memStorageMutex);

    auto iterator = _memoryStorage.find(remoteMemoryHandle);
    if (iterator == _memoryStorage.end() || !iterator->second._isLocked) {
        _logger.warning("{0}: Memory {1} is not found!", __FUNCTION__, remoteMemoryHandle);
        return;
    }
    auto memory = &iterator->second;

    if (memory->_lockOp == InferenceEngine::LOCK_FOR_WRITE) {
        // Sync memory to device
        _logger.info("{0}: Sync {1} memory to device, remoteMemoryHandle {2}", __FUNCTION__,
                     memory->_localMemory.size(), remoteMemoryHandle);
        memory->_remoteMemory->syncToDevice(memory->_localMemory.data(), memory->_localMemory.size());
    } else {
        _logger.warning("{0}: LOCK_FOR_READ, Memory {1} will NOT be synced, remoteMemoryHandle {2}", __FUNCTION__,
                        memory->_localMemory.size(), remoteMemoryHandle);
    }

    memory->_isLocked = false;
}

void* HDDL2RemoteAllocator::wrapRemoteMemoryHandle(const int& /*remoteMemoryFd*/, const size_t /*size*/,
                                                   void* /*memHandle*/) noexcept {
    _logger.error("{0}: Not implemented!", __FUNCTION__);
    return nullptr;
}

void* HDDL2RemoteAllocator::wrapRemoteMemoryOffset(const int& /*remoteMemoryFd*/, const size_t /*size*/,
                                                   const size_t& /*memOffset*/) noexcept {
    _logger.error("{0}: Not implemented!", __FUNCTION__);
    return nullptr;
}

unsigned long HDDL2RemoteAllocator::getPhysicalAddress(void* handle) noexcept {
    UNUSED(handle);
    return 0;
}

}  // namespace hddl2
}  // namespace vpux
