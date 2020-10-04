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
#include "hddl2_config.h"
// Subplugin
#include "vpux.hpp"
namespace vpu {
namespace HDDL2Plugin {

class HDDL2RemoteContext :
    public InferenceEngine::RemoteContext,
    public std::enable_shared_from_this<HDDL2RemoteContext> {
public:
    using Ptr = std::shared_ptr<HDDL2RemoteContext>;
    using CPtr = std::shared_ptr<const HDDL2RemoteContext>;

    explicit HDDL2RemoteContext(const InferenceEngine::ParamMap& paramMap, const vpu::HDDL2Config& config);

    InferenceEngine::RemoteBlob::Ptr CreateBlob(
        const InferenceEngine::TensorDesc& tensorDesc, const InferenceEngine::ParamMap& params) noexcept override;
    // TODO replace with Device::Ptr?
    std::shared_ptr<vpux::IDevice> getDevice() const;
    /** @brief Provide device name attached to current context.
     * Format: {plugin prefix}.{device name} */
    std::string getDeviceName() const noexcept override;

    InferenceEngine::ParamMap getParams() const override;

protected:
    std::shared_ptr<vpux::IDevice> _devicePtr = nullptr;
    const HDDL2Config& _config;
    const Logger::Ptr _logger;
    const InferenceEngine::ParamMap _paramMap;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
