//
// Copyright 2019 Intel Corporation.
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

// clang-format off
// Can get compile error, if the order of the headers will be changed.

#include <ie_metric_helpers.hpp>
#include "inference_engine.hpp"
#include "description_buffer.hpp"
#include "kmb_executable_network.h"
#include "kmb_metrics.h"
#include <memory>
#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include "kmb_config.h"
#include "kmb_remote_context.h"

// clang-format on

namespace vpu {
namespace KmbPlugin {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    Engine();

    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
        const InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string>& config) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string>& config,
        InferenceEngine::QueryNetworkResult& res) const override;

    using ie::InferencePluginInternal::ImportNetwork;

    InferenceEngine::IExecutableNetwork::Ptr ImportNetwork(
        const std::string& modelFileName, const std::map<std::string, std::string>& config) override;

    InferenceEngine::ExecutableNetwork ImportNetworkImpl(
        std::istream& networkModel, const std::map<std::string, std::string>& config) override;

    InferenceEngine::ExecutableNetwork ImportNetworkImpl(std::istream& networkModel, const RemoteContext::Ptr& context,
        const std::map<std::string, std::string>& config) override;

    InferenceEngine::Parameter GetMetric(
        const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    RemoteContext::Ptr CreateContext(const ParamMap& map) override;
    RemoteContext::Ptr GetDefaultContext() override;

private:
    KmbConfig _parsedConfig;
    KmbMetrics _metrics;
    KmbRemoteContext::Ptr _defaultContext;
    std::mutex _contextCreateMutex;
};

}  // namespace KmbPlugin
}  // namespace vpu
