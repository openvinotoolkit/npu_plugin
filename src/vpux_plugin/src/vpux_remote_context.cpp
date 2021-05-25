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
          _contextParams(paramMap) {
}

IE::RemoteBlob::Ptr VPUXRemoteContext::CreateBlob(const IE::TensorDesc& tensorDesc,
                                                  const IE::ParamMap& blobParams) noexcept {
    try {
        auto smart_this = shared_from_this();
    } catch (...) {
        _logger->warning("Please use smart ptr to context instead of instance of class\n");
        return nullptr;
    }
    try {
        auto allocator = _devicePtr->getAllocator(blobParams);
        return std::make_shared<VPUXRemoteBlob>(tensorDesc, shared_from_this(), allocator, blobParams,
                                                _config.logLevel());
    } catch (const std::exception& ex) {
        _logger->warning("Incorrect parameters for CreateBlob call.\n"
                         "Please make sure remote memory is correct.\nError: %s\n",
                         ex.what());
        return nullptr;
    }
}
}  // namespace vpux
