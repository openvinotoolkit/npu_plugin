//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "zero_memory.h"
#include "zero_utils.h"

namespace vpux {
namespace zeroMemory {
HostMem::HostMem(const ze_context_handle_t context, const std::size_t size): _size(size), _context(context) {
    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
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
    if (0 != _size) {
        _size = 0;
        zeroUtils::throwOnFail("zeMemFree HostMem", zeMemFree(_context, _data));
        _data = nullptr;
    }
}
HostMem::~HostMem() {
    free();
}

DeviceMem::DeviceMem(const ze_device_handle_t device_handle, ze_context_handle_t context, const std::size_t size)
        : _size(size), _context(context) {
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
    if (0 != _size) {
        _size = 0;
        zeroUtils::throwOnFail("zeMemFree DeviceMem", zeMemFree(_context, _data));
        _data = nullptr;
    }
}
DeviceMem::~DeviceMem() {
    free();
}

void MemoryManagementUnit::appendArgument(const std::string& name, const ze_graph_argument_properties_t& argument) {
    _offsets.emplace(std::make_pair(name, _size));

    const std::size_t argSize = zeroUtils::getSizeIOBytes(argument);
    _size += argSize + alignment -
             (argSize % alignment);  // is this really necessary? if 0==argSize%alignment -> add 1 * alignment
}

void MemoryManagementUnit::allocate(const ze_device_handle_t device_handle, const ze_context_handle_t context) {
    if (0 != _host.size())
        IE_THROW() << "Memory already allocated";
    if (0 == _size)
        IE_THROW() << "Can't allocate empty buffer";

    _host = HostMem(context, _size);
    _device = DeviceMem(device_handle, context, _size);
}
std::size_t MemoryManagementUnit::getSize() const {
    return _size;
}
const void* MemoryManagementUnit::getHostMemRegion() const {
    return _host.data();
}
const void* MemoryManagementUnit::getDeviceMemRegion() const {
    return _device.data();
}
void* MemoryManagementUnit::getHostMemRegion() {
    return _host.data();
}
void* MemoryManagementUnit::getDeviceMemRegion() {
    return _device.data();
}
void* MemoryManagementUnit::getHostPtr(const std::string& name) {
    uint8_t* from = static_cast<uint8_t*>(_host.data());
    if (nullptr == from)
        IE_THROW() << "Host memory not allocated yet";

    return zeroUtils::mapArguments(_offsets, name) + from;
}
void* MemoryManagementUnit::getDevicePtr(const std::string& name) {
    uint8_t* from = static_cast<uint8_t*>(_device.data());
    if (nullptr == from)
        IE_THROW() << "Device memory not allocated yet";

    return zeroUtils::mapArguments(_offsets, name) + from;
}
bool MemoryManagementUnit::checkHostPtr(const void* ptr) const {
    const uint8_t* from = static_cast<const uint8_t*>(_host.data());
    return (ptr >= from && (from + _size) > ptr);
}
}  // namespace zeroMemory
}  // namespace vpux
