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

// System
#include <map>
#include <string>
// IE
#include "cpp_interfaces/impl/ie_plugin_internal.hpp"
#include "inference_engine.hpp"
// Plugin
#include "hddl2_metrics.h"
#include "vpux.hpp"
#include "vpux_backends.h"
#include "vpux_compiler.hpp"

namespace vpux {
namespace HDDL2 {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    Engine();

    ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
        const ICNNNetwork& network, const std::map<std::string, std::string>& config) override;

    ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
        const ICNNNetwork& network, RemoteContext::Ptr ptr, const std::map<std::string, std::string>& map) override;

    using InferenceEngine::InferencePluginInternal::ImportNetwork;

    IExecutableNetwork::Ptr ImportNetwork(
        const std::string& modelFileName, const std::map<std::string, std::string>& config) override;

    InferenceEngine::ExecutableNetwork ImportNetworkImpl(
        std::istream& networkModel, const std::map<std::string, std::string>& config) override;

    InferenceEngine::ExecutableNetwork ImportNetworkImpl(std::istream& networkModel, const RemoteContext::Ptr& context,
        const std::map<std::string, std::string>& config) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;

    void QueryNetwork(const InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string>& config,
        InferenceEngine::QueryNetworkResult& res) const override;

    InferenceEngine::Parameter GetMetric(
        const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    /**
     * @brief Create context form param map. Will reuse already created workloadContext (workload
     * context id should be specified in param map)
     * @note Params can be found in hddl2_params.h file
     */
    RemoteContext::Ptr CreateContext(const ParamMap& map) override;

private:
    ExecutableNetworkInternal::Ptr LoadExeNetwork(
        const ICNNNetwork& network, std::shared_ptr<vpux::IDevice>& device, const VPUXConfig& networkConfig);

private:
    VPUXConfig _parsedConfig;
    VPUXBackends::CPtr _backends;
    vpu::HDDL2Plugin::HDDL2Metrics _metrics;
    vpux::Compiler::Ptr _compiler;
};

}  // namespace HDDL2
}  // namespace vpux
