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
    Engine(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine&& operator=(Engine&&) = delete;
    virtual ~Engine() = default;

    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
            const InferenceEngine::CNNNetwork& network, const std::map<std::string, std::string>& config) override;

    // Virtual method implemetation throws an error as RemoteContext is deprecated
    InferenceEngine::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
            const InferenceEngine::CNNNetwork&, const std::shared_ptr<InferenceEngine::RemoteContext>&,
            const std::map<std::string, std::string>&) override;

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
            const std::string& modelFileName, const std::map<std::string, std::string>& config) override;

    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
            std::istream& networkModel, const std::map<std::string, std::string>& config) override;

    // Virtual method implemetation throws an error as RemoteContext is deprecated
    InferenceEngine::IExecutableNetworkInternal::Ptr ImportNetwork(
            std::istream&, const std::shared_ptr<InferenceEngine::RemoteContext>&,
            const std::map<std::string, std::string>&) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override;

    InferenceEngine::Parameter GetConfig(
            const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    InferenceEngine::Parameter GetMetric(
            const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    /**
     * @brief Virtual method implemetation throws an error as RemoteContext is deprecated
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
    // properties map OV1.0: {name -> [supported, eval function]}
    std::map<std::string, std::tuple<bool, std::function<InferenceEngine::Parameter(const Config&)>>> propertiesOv1;
    std::vector<ov::PropertyName> supportedProperties0v1;
    // properties map: {name -> [supported, mutable, eval function]}
    std::map<std::string,
             std::tuple<bool, ov::PropertyMutability, std::function<InferenceEngine::Parameter(const Config&)>>>
            propertiesOv2;
    std::vector<ov::PropertyName> supportedProperties0v2;
    Logger _logger;
};

}  // namespace vpux
