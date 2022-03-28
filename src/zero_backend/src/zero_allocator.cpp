//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "zero_allocator.h"

using namespace vpux;

/**
 * @brief Allocates memory
 *
 * @param size The size in bytes to allocate
 * @return Handle to the allocated resource
 */
void* ZeroAllocator::alloc(std::size_t size) noexcept {
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
        delete[] static_cast<char*>(handle);
    }
    return true;
}

bool ZeroAllocator::isZeroPtr(const void* ptr) {
    return our_pointers.count(ptr);
}

std::unordered_set<const void*> ZeroAllocator::our_pointers;
