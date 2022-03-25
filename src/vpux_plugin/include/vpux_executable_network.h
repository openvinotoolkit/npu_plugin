//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// System
#include <memory>
#include <queue>
#include <string>
#include <vector>

// IE
#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"

// Plugin
#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {

class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<ExecutableNetwork>;

    /**
     * @brief Executable network constructor
     * @param network InferenceEngine neural network object
     * @param device pointer to device object
     * @param config config object connecting configuration with which network is compiled
     * @note properties supplied through config parameter, are mutable during the creation
     * of ExecutableNetwork (i.e. until network is compiled) and after ExecutableNetwork is created,
     * all of the supplied properties are switched to read-only mode.
     */
    explicit ExecutableNetwork(const InferenceEngine::CNNNetwork& network, const Device::Ptr& device,
                               const Config& config);

    /**
     * @brief Executable network constructor, imports network from file
     * @param networkModel input stream, to import network from
     * @param device pointer to device object
     * @param config config object connecting configuration with which network is imported
     * @note properties supplied through config parameter, are mutable during the creation
     * of ExecutableNetwork (i.e. until network is compiled) and after ExecutableNetwork is created,
     * all of the supplied properties are switched to read-only mode.
     */
    explicit ExecutableNetwork(std::istream& networkModel, const Device::Ptr& device, const Config& config);

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
            const InferenceEngine::InputsDataMap networkInputs,
            const InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;

    void Export(std::ostream& model) override;
    void Export(const std::string& modelFileName) override;

    /**
     * @brief Returns values of configs options and metrics.
     * Unlike prior to ExecutableNetwork creation, options recieved via this method
     * are read-only. Do not try to set any of the supported options.
     */
    InferenceEngine::Parameter GetMetric(const std::string& name) const override;

private:
    explicit ExecutableNetwork(const Config& config, const Device::Ptr& device);
    Executor::Ptr createExecutor(const NetworkDescription::Ptr& network, const Config& config,
                                 const Device::Ptr& device);

private:
    void ConfigureStreamsExecutor(const std::string& networkName);
    InferenceEngine::ITaskExecutor::Ptr getNextTaskExecutor();

    const Config _config;
    Logger _logger;
    const Device::Ptr _device;
    std::string _networkName;

    Compiler::Ptr _compiler = nullptr;
    NetworkDescription::Ptr _networkPtr = nullptr;
    Executor::Ptr _executorPtr;
    std::vector<std::string> _supportedMetrics;

    static std::atomic<int> loadBlobCounter;
    std::queue<std::string> _taskExecutorGetResultIds;
};

}  //  namespace vpux
