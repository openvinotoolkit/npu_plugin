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

#include "zero_device.h"

#include "zero_allocator.h"
#include "zero_executor.h"

using namespace vpux;

std::shared_ptr<Allocator> ZeroDevice::getAllocator() const {
    std::shared_ptr<Allocator> result = InferenceEngine::details::shared_from_irelease(new ZeroAllocator(_driver_handle));
    return result;
}

std::shared_ptr<Executor> ZeroDevice::createExecutor(
    const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) {
    std::shared_ptr<Executor> result =
        std::make_shared<ZeroExecutor>(_driver_handle, _device_handle, networkDescription, config);
    return result;
}

std::string ZeroDevice::getName() const { return std::string("VPU-0"); }
