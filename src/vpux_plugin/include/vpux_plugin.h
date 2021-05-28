//
// Copyright 2021 Intel Corporation.
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
#include "vpux.hpp"
#include "vpux_backends.h"
#include "vpux_compiler.hpp"
#include "vpux_metrics.h"
#if defined(__arm__) || defined(__aarch64__)
#include "vpux_encryption.h"
#endif

namespace vpux {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    Engine();
    virtual ~Engine() = default;

    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
            const InferenceEngine::CNNNetwork& network, const std::map<std::string, std::string>& config) override;

    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
            const InferenceEngine::CNNNetwork& network, InferenceEngine::RemoteContext::Ptr ptr,
            const std::map<std::string, std::string>& map) override;

    using InferenceEngine::InferencePluginInternal::ImportNetwork;

    InferenceEngine::ExecutableNetwork ImportNetwork(const std::string& modelFileName,
                                                     const std::map<std::string, std::string>& config) override;

    InferenceEngine::ExecutableNetwork ImportNetworkImpl(std::istream& networkModel,
                                                         const std::map<std::string, std::string>& config) override;

    InferenceEngine::ExecutableNetwork ImportNetworkImpl(std::istream& networkModel,
                                                         const InferenceEngine::RemoteContext::Ptr& context,
                                                         const std::map<std::string, std::string>& config) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;

    InferenceEngine::Parameter GetConfig(
            const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::Parameter GetMetric(
            const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    /**
     * @brief Create context form param map. Will reuse already created workloadContext (workload
     * context id should be specified in param map)
     */
    InferenceEngine::RemoteContext::Ptr CreateContext(const InferenceEngine::ParamMap& map) override;

private:
    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetwork(const InferenceEngine::CNNNetwork& network,
                                                                   std::shared_ptr<Device>& device,
                                                                   const VPUXConfig& networkConfig);

private:
    VPUXConfig _parsedConfig;
    VPUXBackends::Ptr _backends;
    Metrics _metrics;
    vpu::Logger _logger;
#if defined(__arm__) || defined(__aarch64__)
    Encryption _encryptionModel;
#endif
};

}  // namespace vpux
