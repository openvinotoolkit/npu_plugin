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

// System
#include <memory>
#include <string>
// Plugin
#include "vpux_remote_blob.h"
#include "vpux_remote_context.h"

namespace vpux {
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
VPUXRemoteContext::VPUXRemoteContext(const std::shared_ptr<Device>& device, const IE::ParamMap& paramMap,
                                     const VPUXConfig& config)
        : _devicePtr(device),
          _config(config),
          _logger(std::make_shared<vpu::Logger>("VPUXRemoteContext", config.logLevel(), vpu::consoleOutput())),
          _paramMap(paramMap) {
}

IE::RemoteBlob::Ptr VPUXRemoteContext::CreateBlob(const IE::TensorDesc& tensorDesc,
                                                  const IE::ParamMap& params) noexcept {
    try {
        auto smart_this = shared_from_this();
    } catch (...) {
        _logger->warning("Please use smart ptr to context instead of instance of class\n");
        return nullptr;
    }
    try {
        auto allocator = _devicePtr->getAllocator();
        return std::make_shared<VPUXRemoteBlob>(tensorDesc, shared_from_this(), allocator, params, _config.logLevel());
    } catch (const std::exception& ex) {
        _logger->warning("Incorrect parameters for CreateBlob call.\n"
                         "Please make sure remote memory is correct.\nError: %s\n",
                         ex.what());
        return nullptr;
    }
}
}  // namespace vpux
