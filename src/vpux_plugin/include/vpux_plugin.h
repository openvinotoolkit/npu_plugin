//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// System
#include <map>
#include <memory>
#include <string>

// IE
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <inference_engine.hpp>

// Plugin
#include "vpux.hpp"
#include "vpux_backends.h"
#include "vpux_compiler.hpp"
#include "vpux_metrics.h"

#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/core/logger.hpp"

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
            std::istream& networkModel, const std::shared_ptr<InferenceEngine::RemoteContext>& context,
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
    std::shared_ptr<InferenceEngine::RemoteContext> CreateContext(const InferenceEngine::ParamMap& map) override;

private:
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetwork(const InferenceEngine::CNNNetwork& network,
                                                                    std::shared_ptr<Device>& device,
                                                                    const Config& networkConfig);

private:
    std::shared_ptr<OptionsDesc> _options;
    Config _globalConfig;
    VPUXBackends::Ptr _backends;
    std::unique_ptr<Metrics> _metrics;
    Logger _logger;
};

}  // namespace vpux
