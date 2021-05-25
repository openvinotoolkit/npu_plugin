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
                                                        const VPUXConfig& config) {
    return HDDL2Executor::prepareExecutor(networkDescription, config);
}
std::shared_ptr<Allocator> ImageWorkloadDevice::getAllocator(const InferenceEngine::ParamMap& paramMap) const {
    if (paramMap.empty())
        return _allocatorPtr;
    IE_THROW() << "ImageWorkloadDevice: ImageWorkload device doesn't have allocators for any params.";
}
}  // namespace hddl2
}  // namespace vpux
