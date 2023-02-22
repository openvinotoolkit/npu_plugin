//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <map>
#include <string>

#include "ze_api.h"
#include "ze_graph_ext.h"

namespace vpux {
namespace zeroMemory {
struct HostMem {
    HostMem() = default;
    /* flag = {} (default 0) - default behavior may use implicit driver-based heuristics */
    HostMem(const ze_context_handle_t context, const std::size_t size, ze_host_mem_alloc_flag_t flag = {});
    HostMem(const HostMem&) = delete;
    HostMem(HostMem&& other): _size(other._size), _data(other._data), _context(other._context) {
        other._size = 0;
        other._data = nullptr;
    }
    HostMem& operator=(const HostMem&) = delete;
    HostMem& operator=(HostMem&& other);

    const void* data() const {
        return _data;
    }
    void* data() {
        return _data;
    }
    std::size_t size() const {
        return _size;
    }
    void free();
    ~HostMem();

private:
    std::size_t _size = 0;
    void* _data = nullptr;
    ze_context_handle_t _context = nullptr;
    const static std::size_t _alignment = 4096;
};

struct DeviceMem {
    DeviceMem() = default;
    DeviceMem(const ze_device_handle_t device_handle, const ze_context_handle_t context, const std::size_t size);
    DeviceMem(const DeviceMem&) = delete;
    DeviceMem(DeviceMem&& other): _size(other._size), _data(other._data), _context(other._context) {
        other._size = 0;
        other._data = nullptr;
    }
    DeviceMem& operator=(const DeviceMem&) = delete;
    DeviceMem& operator=(DeviceMem&& other);

    const void* data() const {
        return _data;
    }
    void* data() {
        return _data;
    }
    std::size_t size() const {
        return _size;
    }
    void free();
    ~DeviceMem();

private:
    std::size_t _size = 0;
    void* _data = nullptr;
    ze_context_handle_t _context = nullptr;
    const static std::size_t _alignment = 4096;
};

// For graph arguments (inputs and outputs) memory should be located on a host side. For discrete HW
// generation the arguments has to be moved to device side to make it accessible.
// MemoryManagementUnit allow to keeps host or host and device allocations.
// Usage: we should append graph arguments with corresponding names with `appendArgument` call
// to prepare size statistics and lookup table. To commit memory allocation we should call `allocate`
struct MemoryManagementUnit {
    MemoryManagementUnit() = default;

    void appendArgument(const std::string& name, const ze_graph_argument_properties_t& argument);
    /* Allocate only Host memory */
    void allocate(const ze_context_handle_t context, ze_host_mem_alloc_flag_t flag = {});
    /* Allocate Host and Device memories */
    void allocate(const ze_device_handle_t device_handle, const ze_context_handle_t context);

    std::size_t getSize() const;
    const void* getHostMemRegion() const;
    const void* getDeviceMemRegion() const;
    void* getHostMemRegion();
    void* getDeviceMemRegion();

    void* getHostPtr(const std::string& name);
    void* getDevicePtr(const std::string& name);

    bool checkHostPtr(const void* ptr) const;

private:
    std::size_t _size = 0;

    HostMem _host;
    DeviceMem _device;
    std::map<std::string, std::size_t> _offsets;

    const static std::size_t alignment = 4096;
};
}  // namespace zeroMemory
}  // namespace vpux
