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

#pragma once

// System
#include <memory>
#include <string>
// Inference-Engine
#include "ie_blob.h"
#include "ie_remote_context.hpp"
// Plugin
#include <vpux_config.hpp>
// Subplugin
#include "vpux.hpp"
namespace vpux {

class VPUXRemoteContext :
        public InferenceEngine::RemoteContext,
        public std::enable_shared_from_this<VPUXRemoteContext> {
public:
    using Ptr = std::shared_ptr<VPUXRemoteContext>;
    using CPtr = std::shared_ptr<const VPUXRemoteContext>;

    explicit VPUXRemoteContext(const std::shared_ptr<Device>& device, const InferenceEngine::ParamMap& paramMap,
                               const VPUXConfig& config = {});

    InferenceEngine::RemoteBlob::Ptr CreateBlob(const InferenceEngine::TensorDesc& tensorDesc,
                                                const InferenceEngine::ParamMap& blobParams) noexcept override;
    std::shared_ptr<Device> getDevice() const {
        return _devicePtr;
    }

    /** @brief Provide device name attached to current context.
     * Format: {plugin prefix}.{device name} */
    std::string getDeviceName() const noexcept override {
        return "VPUX." + _devicePtr->getName();
    }
    InferenceEngine::ParamMap getParams() const override {
        return _contextParams;
    }

protected:
    std::shared_ptr<Device> _devicePtr = nullptr;
    const VPUXConfig _config;
    const vpu::Logger::Ptr _logger;
    const InferenceEngine::ParamMap _contextParams;
};

}  // namespace vpux
