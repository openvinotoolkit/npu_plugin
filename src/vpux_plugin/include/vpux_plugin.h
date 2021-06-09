//
// Copyright 2021 Intel Corporation.
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
#include <map>
#include <string>
// IE
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
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

class Engine : public InferenceEngine::IInferencePlugin {
public:
    Engine();

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
            const InferenceEngine::CNNNetwork& network, const std::map<std::string, std::string>& config) override;

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
            const InferenceEngine::CNNNetwork& network, const std::shared_ptr<InferenceEngine::RemoteContext>& ptr,
            const std::map<std::string, std::string>& map) override;

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
            const std::string& modelFileName, const std::map<std::string, std::string>& config) override;

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
            std::istream& networkModel, const std::map<std::string, std::string>& config) override;

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
            std::istream& networkModel, const InferenceEngine::RemoteContext::Ptr& context,
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
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetwork(const InferenceEngine::CNNNetwork& network,
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
