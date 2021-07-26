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

#include "zero_device.h"

#include "zero_allocator.h"
#include "zero_executor.h"

using namespace vpux;

std::shared_ptr<Allocator> ZeroDevice::getAllocator() const {
    std::shared_ptr<Allocator> result = std::make_shared<ZeroAllocator>(_driver_handle);
    return result;
}

std::shared_ptr<Executor> ZeroDevice::createExecutor(const NetworkDescription::Ptr& networkDescription,
                                                     const VPUXConfig& config) {
    return std::make_shared<ZeroExecutor>(_driver_handle, _device_handle, _context, _graph_ddi_table_ext,
                                          networkDescription, config);
}

std::string ZeroDevice::getName() const {
    return std::string("3700.0");
}
