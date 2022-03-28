//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

// Plugin
#include "hddl2_exceptions.h"
#include "hddl2_executor.h"
// Subplugin
#include "image_workload_device.h"

namespace IE = InferenceEngine;

namespace vpux {
namespace hddl2 {

//------------------------------------------------------------------------------
ImageWorkloadDevice::ImageWorkloadDevice(const std::string& name): _name(name) {
}

vpux::Executor::Ptr ImageWorkloadDevice::createExecutor(const NetworkDescription::Ptr& networkDescription,
                                                        const Config& config) {
    return HDDL2Executor::prepareExecutor(networkDescription, config);
}
std::shared_ptr<Allocator> ImageWorkloadDevice::getAllocator(const InferenceEngine::ParamMap& paramMap) const {
    if (paramMap.empty())
        return _allocatorPtr;
    IE_THROW() << "ImageWorkloadDevice: ImageWorkload device doesn't have allocators for any params.";
}
}  // namespace hddl2
}  // namespace vpux
