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

#include <ie_blob.h>

#include <ie_allocator.hpp>
#include <ie_remote_context.hpp>
#include <memory>
#include <string>
#include <vpux.hpp>

#include "kmb_config.h"

namespace vpu {
namespace KmbPlugin {
//------------------------------------------------------------------------------
//      class KmbRemoteContext
//------------------------------------------------------------------------------
class KmbRemoteContext : public InferenceEngine::RemoteContext, public std::enable_shared_from_this<KmbRemoteContext> {
public:
    using Ptr = std::shared_ptr<KmbRemoteContext>;
    using CPtr = std::shared_ptr<const KmbRemoteContext>;

    explicit KmbRemoteContext(const InferenceEngine::ParamMap& paramMap, const KmbConfig& config,
        const std::shared_ptr<vpux::Device>& device);

    InferenceEngine::RemoteBlob::Ptr CreateBlob(
        const InferenceEngine::TensorDesc& tensorDesc, const InferenceEngine::ParamMap& params) noexcept override;

    std::string getDeviceName() const noexcept override;

    InferenceEngine::ParamMap getParams() const override;

protected:
    const KmbConfig& _config;
    InferenceEngine::ParamMap _params;

    const Logger::Ptr _logger;
    std::shared_ptr<vpux::Device> _device = nullptr;
};

}  // namespace KmbPlugin
}  // namespace vpu
