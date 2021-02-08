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

#include "zero_allocator.h"

using namespace vpux;

/**
 * @brief Allocates memory
 *
 * @param size The size in bytes to allocate
 * @return Handle to the allocated resource
 */
void* ZeroAllocator::alloc(size_t size) noexcept {
    void* mem = new char[size];
    our_pointers.insert(mem);
    return mem;
}

/**
 * @brief Releases the handle and all associated memory resources which invalidates the handle.
 * @param handle The handle to free
 * @return `false` if handle cannot be released, otherwise - `true`.
 */
bool ZeroAllocator::free(void* handle) noexcept {
    if (handle) {
        our_pointers.erase(handle);
        delete [] handle;
    }
    return true;
}

bool ZeroAllocator::isZeroPtr(const void* ptr) { return our_pointers.count(ptr); }

std::unordered_set<const void*> ZeroAllocator::our_pointers;
