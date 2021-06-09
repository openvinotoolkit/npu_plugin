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
        delete[] static_cast<char*>(handle);
    }
    return true;
}

bool ZeroAllocator::isZeroPtr(const void* ptr) {
    return our_pointers.count(ptr);
}

std::unordered_set<const void*> ZeroAllocator::our_pointers;
