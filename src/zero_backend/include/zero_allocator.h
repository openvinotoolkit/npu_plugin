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
    explicit ZeroAllocator(ze_driver_handle_t driver): driver_handle(driver) {}

    /**
     * @brief Maps handle to heap memory accessible by any memory manipulation routines.
     *
     * @param handle Handle to the allocated memory to be locked
     * @param op Operation to lock memory for
     * @return Generic pointer to memory
     */
    void* lock(void* handle, InferenceEngine::LockOp) noexcept override { return handle; }
    /**
     * @brief Unmaps memory by handle with multiple sequential mappings of the same handle.
     *
     * The multiple sequential mappings of the same handle are suppose to get the same
     * result while there isn't a ref counter supported.
     *
     * @param handle Handle to the locked memory to unlock
     */
    void unlock(void*) noexcept override {}
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
    void* wrapRemoteMemoryHandle(const int&, const size_t, void*) noexcept override { return 0; }
    void* wrapRemoteMemoryOffset(const int&, const size_t, const size_t&) noexcept override { return 0; }

    // FIXME: temporary exposed to allow executor to use vpux::Allocator
    unsigned long getPhysicalAddress(void* handle) noexcept override { return 0; }

    static bool isZeroPtr(const void*);

protected:
    ZeroAllocator(const ZeroAllocator&) = default;
    ZeroAllocator& operator=(const ZeroAllocator&) = default;
};

}  // namespace vpux
