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

#include <memory>
#include <string>

#include "ie_remote_context.hpp"
#include "kmb_allocator.h"
#include "kmb_config.h"

namespace vpu {
namespace KmbPlugin {
//------------------------------------------------------------------------------
//      class KmbContextParams
//------------------------------------------------------------------------------
class KmbContextParams {
public:
    explicit KmbContextParams(const InferenceEngine::ParamMap& paramMap);

    InferenceEngine::ParamMap getParamMap() const;
    int getDeviceId() const;
    std::string getDeviceIdStr() const;

protected:
    InferenceEngine::ParamMap _paramMap;
    int _deviceId;
    std::string _deviceIdStr;
};

//------------------------------------------------------------------------------
//      class KmbRemoteContext
//------------------------------------------------------------------------------
class KmbRemoteContext :
    public InferenceEngine::RemoteContext,
    public std::enable_shared_from_this<KmbRemoteContext> {
public:
    using Ptr = std::shared_ptr<KmbRemoteContext>;
    using CPtr = std::shared_ptr<const KmbRemoteContext>;

    explicit KmbRemoteContext(const InferenceEngine::ParamMap& paramMap, const KmbConfig& config);

    InferenceEngine::RemoteBlob::Ptr CreateBlob(
        const InferenceEngine::TensorDesc& tensorDesc, const InferenceEngine::ParamMap& params) noexcept override;

    std::string getDeviceName() const noexcept override;
    int getDeviceId() const noexcept;

    InferenceEngine::ParamMap getParams() const override;
    KmbAllocator::Ptr getAllocator();
    KmbContextParams getContextParams() const;

protected:
    const KmbConfig _config;
    KmbContextParams _contextParams;
    KmbAllocator::Ptr _allocatorPtr = nullptr;

    const Logger::Ptr _logger;
    int _deviceId;
};

}  // namespace KmbPlugin
}  // namespace vpu
