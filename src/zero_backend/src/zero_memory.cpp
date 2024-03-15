//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "zero_memory.h"
#include "zero_utils.h"

namespace vpux {
namespace zeroMemory {
HostMem::HostMem(const ze_context_handle_t context, const std::size_t size, ze_host_mem_alloc_flag_t flag)
        : _size(size), _context(context), _log(Logger::global().nest("HostMem", 0)) {
    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
                                     static_cast<ze_host_mem_alloc_flags_t>(flag)};
    zeroUtils::throwOnFail("zeMemAllocHost", zeMemAllocHost(_context, &desc, _size, _alignment, &_data));
}
HostMem& HostMem::operator=(HostMem&& other) {
    if (this == &other)
        return *this;
    free();
    _size = other._size;
    _data = other._data;
    _context = other._context;
    other._size = 0;
    other._data = nullptr;
    return *this;
}
void HostMem::free() {
    if (_size != 0) {
        _size = 0;
        zeroUtils::throwOnFail("zeMemFree HostMem", zeMemFree(_context, _data));
        _data = nullptr;
    }
}
HostMem::~HostMem() {
    try {
        free();
    } catch (const std::exception& e) {
        _log.error("Caught when freeing memory: {0}", e.what());
    }
}

DeviceMem::DeviceMem(const ze_device_handle_t device_handle, ze_context_handle_t context, const std::size_t size)
        : _size(size), _context(context), _log(Logger::global().nest("DeviceMem", 0)) {
    ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};

    zeroUtils::throwOnFail("zeMemAllocDevice",
                           zeMemAllocDevice(_context, &desc, _size, _alignment, device_handle, &_data));
}
DeviceMem& DeviceMem::operator=(DeviceMem&& other) {
    if (this == &other)
        return *this;
    free();
    _size = other._size;
    _data = other._data;
    _context = other._context;
    other._size = 0;
    other._data = nullptr;
    return *this;
}
void DeviceMem::free() {
    if (_size != 0) {
        _size = 0;
        zeroUtils::throwOnFail("zeMemFree DeviceMem", zeMemFree(_context, _data));
        _data = nullptr;
    }
}
DeviceMem::~DeviceMem() {
    try {
        free();
    } catch (const std::exception& e) {
        _log.error("Caught when freeing memory: {0}", e.what());
    }
}

void MemoryManagementUnit::appendArgument(const std::string& name, const ze_graph_argument_properties_t& argument) {
    _offsets.emplace(std::make_pair(name, _size));

    const std::size_t argSize = zeroUtils::getSizeIOBytes(argument);
    _size += argSize + alignment -
             (argSize % alignment);  // is this really necessary? if 0==argSize%alignment -> add 1 * alignment
}

void MemoryManagementUnit::allocate(const ze_context_handle_t context, ze_host_mem_alloc_flag_t flag) {
    if (_host && _host->size() != 0) {
        OPENVINO_THROW("Memory already allocated");
    }
    if (_size == 0) {
        OPENVINO_THROW("Can't allocate empty buffer");
    }

    _host = std::make_unique<HostMem>(context, _size, flag);
}

void MemoryManagementUnit::allocate(const ze_device_handle_t device_handle, const ze_context_handle_t context) {
    if (_host && _host->size() != 0) {
        OPENVINO_THROW("Memory already allocated");
    }
    if (_size == 0) {
        OPENVINO_THROW("Can't allocate empty buffer");
    }

    _host = std::make_unique<HostMem>(context, _size);
    _device = std::make_unique<DeviceMem>(device_handle, context, _size);
}
std::size_t MemoryManagementUnit::getSize() const {
    return _size;
}
const void* MemoryManagementUnit::getHostMemRegion() const {
    return _host ? _host->data() : nullptr;
}
const void* MemoryManagementUnit::getDeviceMemRegion() const {
    return _device ? _device->data() : nullptr;
}
void* MemoryManagementUnit::getHostMemRegion() {
    return _host ? _host->data() : nullptr;
}
void* MemoryManagementUnit::getDeviceMemRegion() {
    return _device ? _device->data() : nullptr;
}
void* MemoryManagementUnit::getHostPtr(const std::string& name) {
    uint8_t* from = static_cast<uint8_t*>(_host ? _host->data() : nullptr);
    if (from == nullptr) {
        OPENVINO_THROW("Host memory not allocated yet");
    }
    if (!_offsets.count(name)) {
        OPENVINO_THROW("Invalid memory offset key: ", name);
    }

    return _offsets.at(name) + from;
}
void* MemoryManagementUnit::getDevicePtr(const std::string& name) {
    uint8_t* from = static_cast<uint8_t*>(_device ? _device->data() : nullptr);
    if (from == nullptr) {
        OPENVINO_THROW("Device memory not allocated yet");
    }
    if (!_offsets.count(name)) {
        OPENVINO_THROW("Invalid memory offset key: ", name);
    }

    return _offsets.at(name) + from;
}
bool MemoryManagementUnit::checkHostPtr(const void* ptr) const {
    const uint8_t* from = static_cast<const uint8_t*>(_host ? _host->data() : nullptr);
    return (ptr >= from && (from + _size) > ptr);
}
}  // namespace zeroMemory
}  // namespace vpux
