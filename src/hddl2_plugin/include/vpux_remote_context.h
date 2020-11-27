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
                                                const InferenceEngine::ParamMap& params) noexcept override;
    std::shared_ptr<Device> getDevice() const {
        return _devicePtr;
    }

    /** @brief Provide device name attached to current context.
     * Format: {plugin prefix}.{device name} */
    std::string getDeviceName() const noexcept override {
        return "VPUX." + _devicePtr->getName();
    }
    InferenceEngine::ParamMap getParams() const override {
        return _paramMap;
    }

protected:
    std::shared_ptr<Device> _devicePtr = nullptr;
    const VPUXConfig _config;
    const vpu::Logger::Ptr _logger;
    const InferenceEngine::ParamMap _paramMap;
};

}  // namespace vpux
