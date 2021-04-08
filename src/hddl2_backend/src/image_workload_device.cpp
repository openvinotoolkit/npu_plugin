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
