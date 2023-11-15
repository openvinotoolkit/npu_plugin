//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/IE/itt.hpp"
#include "ze_api.h"

#include "zero_device.h"

#include "zero_allocator.h"
#include "zero_executor.h"
#include "zero_infer_request.h"

using namespace vpux;
static size_t get_cpu_ram_size();

ZeroDevice::ZeroDevice(ze_driver_handle_t driver, ze_device_handle_t device, ze_context_handle_t context,
                       ze_graph_dditable_ext_t* graph_ddi_table_ext,
                       ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext)
        : _driver_handle(driver),
          _device_handle(device),
          _context(context),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _graph_profiling_ddi_table_ext(graph_profiling_ddi_table_ext),
          log(Logger::global().nest("ZeroDevice", 0)) {
    ze_device_properties_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties", zeDeviceGetProperties(_device_handle, &properties));

    fullDeviceName = properties.name;

    if (properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) {
        // Return host memory size when device is integrated
        totalMemSize = get_cpu_ram_size();
    } else {
        // Query driver when device is discrete
        try {
            uint32_t device_memory_properties_count = 0;
            zeroUtils::throwOnFail(
                    "zeDeviceGetMemoryProperties",
                    zeDeviceGetMemoryProperties(_device_handle, &device_memory_properties_count, nullptr));
            VPUX_THROW_UNLESS(device_memory_properties_count == 1, "Unexpected count of memory properties");
            std::vector<ze_device_memory_properties_t> device_memory_properties;
            device_memory_properties.resize(device_memory_properties_count);
            zeroUtils::throwOnFail("zeDeviceGetMemoryProperties",
                                   zeDeviceGetMemoryProperties(_device_handle, &device_memory_properties_count,
                                                               device_memory_properties.data()));
            VPUX_THROW_UNLESS(strcmp(device_memory_properties[0].name, "DDR") == 0,
                              "Unexpected name of memory property");

            totalMemSize = device_memory_properties[0].totalSize;
        } catch (const std::exception& e) {
            log.debug("Can not obtain device memory properties: {0}",
                      e.what());  // todo: E#78609, upgrade driver version to 31.0.12+ and then remove try block
        }
    }
    std::vector<ze_command_queue_group_properties_t> command_group_properties;
    uint32_t command_queue_group_count = 0;
    // Discover all command queue groups
    zeroUtils::throwOnFail("zeDeviceGetCommandQueueGroupProperties",
                           zeDeviceGetCommandQueueGroupProperties(_device_handle, &command_queue_group_count, nullptr));
    command_group_properties.resize(command_queue_group_count);
    zeroUtils::throwOnFail("zeDeviceGetCommandQueueGroupProperties",
                           zeDeviceGetCommandQueueGroupProperties(_device_handle, &command_queue_group_count,
                                                                  command_group_properties.data()));

    // Find the corespondinng command queue group.
    _group_ordinal = zeroUtils::findGroupOrdinal(command_group_properties, properties);
}

std::shared_ptr<Allocator> ZeroDevice::getAllocator() const {
    std::shared_ptr<Allocator> result = std::make_shared<ZeroAllocator>(_driver_handle);
    return result;
}

std::shared_ptr<Executor> ZeroDevice::createExecutor(const NetworkDescription::Ptr& networkDescription,
                                                     const Config& config) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Device::createExecutor");
    return std::make_shared<ZeroExecutor>(_driver_handle, _device_handle, _context, _graph_ddi_table_ext,
                                          _graph_profiling_ddi_table_ext, networkDescription, config, _group_ordinal);
}

std::string ZeroDevice::getName() const {
    ze_device_properties_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties", zeDeviceGetProperties(_device_handle, &properties));

//    KMD is setting usDeviceID from VpuFamilyID.h
#define VPU_2700_DEVICE_ID 0x6200
#define VPU_3700_DEVICE_ID 0x6240
#define VPU_3720_P_DEVICE_ID 0x7D1D
#define VPU_3720_S_DEVICE_ID 0xAD1D

    std::string name;
    switch (properties.deviceId) {
    case VPU_2700_DEVICE_ID:
        name = "2700";
        break;
    case VPU_3700_DEVICE_ID:
        name = "3700";
        break;
    case VPU_3720_P_DEVICE_ID:
    case VPU_3720_S_DEVICE_ID:
        name = "3720";
        break;
    default:
        name = "AUTO_DETECT";
    }

    return name + ".0";
}

std::string ZeroDevice::getFullDeviceName() const {
    return fullDeviceName;
}

Uuid ZeroDevice::getUuid() const {
    ze_device_properties_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties", zeDeviceGetProperties(_device_handle, &properties));

    Uuid uuid{};
    static_assert(sizeof(properties.uuid.id) == uuid.uuid.size(),
                  "ze_device_uuid_t::id size doesn't match vpux::Uuid::uuid size");

    std::copy(std::begin(properties.uuid.id), std::end(properties.uuid.id), std::begin(uuid.uuid));

    return uuid;
}

uint32_t ZeroDevice::getDriverVersion() const {
    ze_driver_properties_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties", zeDriverGetProperties(_driver_handle, &properties));

    return properties.driverVersion;
}

uint64_t ZeroDevice::getTotalMemSize() const {
    return totalMemSize;
}

IInferRequest::Ptr ZeroDevice::createInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                                  const InferenceEngine::OutputsDataMap& networkOutputs,
                                                  const Executor::Ptr& executor, const Config& config,
                                                  const std::string& netName,
                                                  const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                                  const std::vector<std::shared_ptr<const ov::Node>>& results,
                                                  const vpux::NetworkIOVector& networkStatesInfo,
                                                  const std::shared_ptr<InferenceEngine::IAllocator>& allocator) {
    return std::make_shared<ZeroInferRequest>(networkInputs, networkOutputs, executor, config, netName, parameters,
                                              results, networkStatesInfo, allocator);
}

#if defined(_WIN32) && !defined(__GNUC__)
#include "windows.h"

static size_t get_cpu_ram_size() {
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    GlobalMemoryStatusEx(&s);
    return s.ullTotalPhys;
}

#else
#include <sys/sysinfo.h>

static size_t get_cpu_ram_size() {
    struct sysinfo s {};
    sysinfo(&s);
    return s.totalram;
}
#endif
