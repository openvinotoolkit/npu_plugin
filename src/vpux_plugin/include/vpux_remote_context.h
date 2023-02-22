//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// System
#include <memory>
#include <string>

// Inference-Engine
#include <ie_blob.h>
#include <ie_remote_context.hpp>

// Subplugin
#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {

class VPUXRemoteContext : public InferenceEngine::RemoteContext {
public:
    using Ptr = std::shared_ptr<VPUXRemoteContext>;
    using CPtr = std::shared_ptr<const VPUXRemoteContext>;

    explicit VPUXRemoteContext(const std::shared_ptr<Device>& device, const InferenceEngine::ParamMap& paramMap,
                               LogLevel logLvl = LogLevel::None);

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
    Logger _logger;
    const InferenceEngine::ParamMap _contextParams;
};

}  // namespace vpux
