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
#pragma once

#include <ie_allocator.hpp>
#include <unordered_set>
#include <vpux.hpp>

#include "ze_api.h"

namespace vpux {
class ZeroAllocator : public Allocator {
    ze_driver_handle_t driver_handle = nullptr;
    const static size_t alignment = 4096;

    static std::unordered_set<const void*> our_pointers;

public:
    explicit ZeroAllocator(ze_driver_handle_t driver): driver_handle(driver) {
    }

    /**
     * @brief Maps handle to heap memory accessible by any memory manipulation routines.
     *
     * @param handle Handle to the allocated memory to be locked
     * @param op Operation to lock memory for
     * @return Generic pointer to memory
     */
    void* lock(void* handle, InferenceEngine::LockOp) noexcept override {
        return handle;
    }
    /**
     * @brief Unmaps memory by handle with multiple sequential mappings of the same handle.
     *
     * The multiple sequential mappings of the same handle are suppose to get the same
     * result while there isn't a ref counter supported.
     *
     * @param handle Handle to the locked memory to unlock
     */
    void unlock(void*) noexcept override {
    }
    /**
     * @brief Allocates memory
     *
     * @param size The size in bytes to allocate
     * @return Handle to the allocated resource
     */
    void* alloc(size_t size) noexcept override;
    /**
     * @brief Releases the handle and all associated memory resources which invalidates the handle.
     * @param handle The handle to free
     * @return `false` if handle cannot be released, otherwise - `true`.
     */
    bool free(void* handle) noexcept override;

    // TODO: need update methods to remove Kmb from parameters
    void* wrapRemoteMemoryHandle(const int&, const size_t, void*) noexcept override {
        return 0;
    }
    void* wrapRemoteMemoryOffset(const int&, const size_t, const size_t&) noexcept override {
        return 0;
    }

    // FIXME: temporary exposed to allow executor to use vpux::Allocator
    unsigned long getPhysicalAddress(void* /*handle*/) noexcept override {
        return 0;
    }

    static bool isZeroPtr(const void*);

protected:
    ZeroAllocator(const ZeroAllocator&) = default;
    ZeroAllocator& operator=(const ZeroAllocator&) = default;
};

}  // namespace vpux
