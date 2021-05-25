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

#include "hddl2_helper.h"

// Utils
#include <device_helpers.hpp>

#include "converters.h"
// [Track number: E#12122]
// TODO Remove this header after removing HDDL2 deprecated parameters in future releases
#include "hddl2/hddl2_params.hpp"
#include "hddl2_exceptions.h"
#include "vpux/vpux_plugin_params.hpp"
#include "vpux_params_private_options.hpp"

namespace vpux {
namespace hddl2 {

namespace IE = InferenceEngine;
HddlUnite::RemoteMemory* getRemoteMemoryFromParams(const IE::ParamMap& params) {
    if (params.empty()) {
        IE_THROW() << PARAMS_ERROR_str << "Parameter map for allocator is empty.";
    }
    // Check that it really contains required params
    const auto remoteMemoryIter = params.find(IE::VPUX_PARAM_KEY(MEM_HANDLE));
    if (remoteMemoryIter == params.end()) {
        IE_THROW() << PARAMS_ERROR_str
                   << "Parameter map for allocator does not contain remote memory object information";
    }

    VpuxHandleParam vpuxMemHandle;
    try {
        vpuxMemHandle = remoteMemoryIter->second.as<VpuxHandleParam>();
    } catch (...) {
        IE_THROW() << CONFIG_ERROR_str << "Remote memory parameter has incorrect type information";
    }

    if (vpuxMemHandle == nullptr) {
        IE_THROW() << CONFIG_ERROR_str << "Remote memory parameter has incorrect data";
    }

    return static_cast<HddlUnite::RemoteMemory*>(vpuxMemHandle);
}

int32_t getRemoteMemoryFDFromParams(const IE::ParamMap& params) {
    if (params.empty()) {
        IE_THROW() << PARAMS_ERROR_str << "Parameter map is empty.";
    }
    // Check that it really contains required params
    auto remoteMemoryIter = params.find(IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD));
    if (remoteMemoryIter == params.end()) {
        // ******************************************
        // [Track number: E#12122]
        // TODO Remove this part after removing HDDL2 deprecated parameters in future releases
        remoteMemoryIter = params.find(IE::HDDL2_PARAM_KEY(REMOTE_MEMORY));
        if (remoteMemoryIter != params.end()) {
            HddlUnite::RemoteMemory::Ptr vpuxRemoteMemoryPtr = nullptr;
            try {
                vpuxRemoteMemoryPtr = remoteMemoryIter->second.as<HddlUnite::RemoteMemory::Ptr>();
            } catch (...) {
                IE_THROW() << CONFIG_ERROR_str << "Remote memory parameter has incorrect type information";
            }

            if (vpuxRemoteMemoryPtr == nullptr) {
                IE_THROW() << CONFIG_ERROR_str << "Remote memory parameter has incorrect data";
            }

            return vpuxRemoteMemoryPtr->getDmaBufFd();
        }
        // ******************************************

        IE_THROW() << PARAMS_ERROR_str << "Parameter map does not contain remote memory file descriptor information";
    }

    VpuxRemoteMemoryFD vpuxRemoteMemoryFD;
    try {
        vpuxRemoteMemoryFD = remoteMemoryIter->second.as<VpuxRemoteMemoryFD>();
    } catch (...) {
        IE_THROW() << CONFIG_ERROR_str << "Remote memory FD parameter has incorrect type information";
    }

    if (vpuxRemoteMemoryFD < 0) {
        IE_THROW() << CONFIG_ERROR_str << "Remote memory FD parameter has incorrect data";
    }

    return vpuxRemoteMemoryFD;
}

std::shared_ptr<IE::TensorDesc> getOriginalTensorDescFromParams(const InferenceEngine::ParamMap& params) {
    if (params.empty()) {
        IE_THROW() << PARAMS_ERROR_str << "Parameter map is empty.";
    }
    // Check that it really contains required params
    const auto remoteMemoryIter = params.find(IE::VPUX_PARAM_KEY(ORIGINAL_TENSOR_DESC));
    if (remoteMemoryIter == params.end()) {
        IE_THROW() << PARAMS_ERROR_str << "Parameter map does not contain original tensor descriptor information";
    }

    std::shared_ptr<IE::TensorDesc> originalTensorDesc = nullptr;
    try {
        originalTensorDesc = remoteMemoryIter->second.as<std::shared_ptr<IE::TensorDesc>>();
    } catch (...) {
        IE_THROW() << CONFIG_ERROR_str << "Original tensor descriptor parameter has incorrect type information";
    }

    if (originalTensorDesc == nullptr) {
        IE_THROW() << CONFIG_ERROR_str << "Original tensor descriptor parameter has incorrect data";
    }

    return originalTensorDesc;
}

WorkloadID getWorkloadIDFromParams(const InferenceEngine::ParamMap& params) {
    if (params.empty()) {
        IE_THROW() << PARAMS_ERROR_str << "Parameter map is empty.";
    }
    // Check that it really contains required params
    auto workloadIdIter = params.find(IE::VPUX_PARAM_KEY(WORKLOAD_CONTEXT_ID));
    if (workloadIdIter == params.end()) {
        // ******************************************
        // TODO Remove this part after removing HDDL2 deprecated parameters in future releases
        workloadIdIter = params.find(IE::HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID));
        if (workloadIdIter != params.end()) {
            WorkloadID workloadIdHddl2;
            try {
                workloadIdHddl2 = workloadIdIter->second.as<WorkloadID>();
            } catch (...) {
                IE_THROW() << CONFIG_ERROR_str << "Workload ID parameter has incorrect type information";
            }

            return workloadIdHddl2;
        }
        // ******************************************

        IE_THROW() << PARAMS_ERROR_str << "Parameter map does not contain workload ID information";
    }

    WorkloadID workloadId;
    try {
        workloadId = workloadIdIter->second.as<WorkloadID>();
    } catch (...) {
        IE_THROW() << CONFIG_ERROR_str << "Workload ID parameter has incorrect type information";
    }

    return workloadId;
}

void setUniteLogLevel(const vpu::LogLevel logLevel, const vpu::Logger::Ptr logger) {
    const auto status = HddlUnite::setClientLogLevel(Unite::convertIELogLevelToUnite(logLevel));
    if (status != HddlStatusCode::HDDL_OK) {
        if (logger != nullptr) {
            logger->warning("Failed to set client log level for HddlUnite");
        } else {
            std::cerr << "Failed to set client log level for HddlUnite" << std::endl;
        }
    }
}

std::map<uint32_t, std::string> getSwDeviceIdNameMap() {
    std::vector<HddlUnite::Device> devices;
    auto status = getAvailableDevices(devices);
    if (status != HDDL_OK) {
        IE_THROW() << "Failed to get devices sw IDs!";
    }

    std::map<uint32_t, std::string> swIdNameMap;
    for (const auto& device : devices) {
        const auto swDevId = device.getSwDeviceId();
        const auto devName = utils::getDeviceNameBySwDeviceId(swDevId);
        swIdNameMap.insert({swDevId, devName});
    }

    return swIdNameMap;
}

std::string getSwDeviceIdFromName(const std::string& devName) {
    const auto devMap = getSwDeviceIdNameMap();

    // Firstly check new naming approach with platform.slice_id
    for (const auto& dev : devMap) {
        if (dev.second == devName) {
            return std::to_string(dev.first);
        }
    }

    // Secondly check old naming approach with swDeviceId
    try {
        std::stol(devName);
    } catch (...) {
        // Some unknown name - return empty string - scheduler is responsible for scheduling the device
        return "";
    }

    return devName;
}

}  // namespace hddl2
}  // namespace vpux
