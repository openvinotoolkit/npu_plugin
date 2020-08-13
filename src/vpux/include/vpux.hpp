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

#include <vector>
// clang-format off
#include <cstdint>
// clang-format on
#include <memory>
#include <map>
#include <set>

#include <ie_blob.h>
#include <ie_common.h>
#include <ie_remote_context.hpp>
#include <ie_icnn_network.hpp>

#include "vpux_compiler.hpp"

namespace vpux {

class Executor;

class SubPlugin : public InferenceEngine::details::IRelease {
public:
    virtual std::shared_ptr<InferenceEngine::IAllocator> getAllocator() = 0;
    virtual std::shared_ptr<Executor> createExecutor(
        const std::shared_ptr<NetworkDescription>& network, const InferenceEngine::ParamMap& params) = 0;
};

class Executor {
public:
    using Ptr = std::shared_ptr<Executor>;

    virtual void setup(const InferenceEngine::ParamMap& params) = 0;

    virtual void push(const InferenceEngine::BlobMap& inputs) = 0;
    virtual void pull(InferenceEngine::BlobMap& outputs) = 0;

    virtual bool isPreProcessingSupported(const InferenceEngine::PreProcessInfo& preProcessInfo) const = 0;
    virtual std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() = 0;
    virtual InferenceEngine::Parameter getParameter(const std::string& paramName) const = 0;

    virtual ~Executor() = default;
};

class SubPluginManager {
public:
    std::shared_ptr<SubPlugin> findSubPlugin(const InferenceEngine::ParamMap& params);

private:
    SubPluginManager();
};

}  // namespace vpux
