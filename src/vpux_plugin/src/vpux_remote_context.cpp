//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_remote_context.h"

// System
#include <memory>
#include <string>

// Plugin
#include "vpux_remote_blob.h"

namespace vpux {
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
VPUXRemoteContext::VPUXRemoteContext(const std::shared_ptr<Device>& device, const IE::ParamMap& paramMap,
                                     LogLevel logLvl)
        : _devicePtr(device), _logger("VPUXRemoteContext", logLvl), _contextParams(paramMap) {
}

IE::RemoteBlob::Ptr VPUXRemoteContext::CreateBlob(const IE::TensorDesc& tensorDesc,
                                                  const IE::ParamMap& blobParams) noexcept {
    try {
        auto smart_this = shared_from_this();
    } catch (...) {
        _logger.warning("Please use smart ptr to context instead of instance of class");
        return nullptr;
    }
    try {
        auto allocator = _devicePtr->getAllocator(blobParams);
        return std::make_shared<VPUXRemoteBlob>(tensorDesc,
                                                std::dynamic_pointer_cast<VPUXRemoteContext>(shared_from_this()),
                                                allocator, blobParams, _logger.level());
    } catch (const std::exception& ex) {
        _logger.warning("Incorrect parameters for CreateBlob call. "
                        "Please make sure remote memory is correct. Error: {0}",
                        ex.what());
        return nullptr;
    }
}
}  // namespace vpux
