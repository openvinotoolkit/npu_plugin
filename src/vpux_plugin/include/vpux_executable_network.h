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
// System
#include <memory>
#include <queue>
#include <string>
#include <vector>
// IE
#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
// Plugin
#include "vpux.hpp"
#include "vpux_config.hpp"

namespace vpux {

class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<ExecutableNetwork>;

    explicit ExecutableNetwork(InferenceEngine::CNNNetwork& network, const Device::Ptr& device,
                               const VPUXConfig& config);
    explicit ExecutableNetwork(std::istream& networkModel, const Device::Ptr& device, const VPUXConfig& config);
    ~ExecutableNetwork() override = default;

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
            const InferenceEngine::InputsDataMap networkInputs,
            const InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;

    using InferenceEngine::ExecutableNetworkInternal::Export;
    void ExportImpl(std::ostream& model) override;
    void Export(std::ostream& networkModel) override {
        ExportImpl(networkModel);
    }
    void Export(const std::string& modelFileName) override;

    InferenceEngine::Parameter GetMetric(const std::string& name) const override;

private:
    explicit ExecutableNetwork(const VPUXConfig& config, const Device::Ptr& device);
    Executor::Ptr createExecutor(const NetworkDescription::Ptr& network, const VPUXConfig& config,
                                 const Device::Ptr& device);

private:
    void ConfigureStreamsExecutor(const std::string& networkName);
    InferenceEngine::ITaskExecutor::Ptr getNextTaskExecutor();

    const VPUXConfig _config;
    const vpu::Logger::Ptr _logger;
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
