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
#include <string>
#include <vector>
// IE
// #include <ie_common.h>
#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
// Plugin
#include "hddl2_config.h"
#include "vpux.hpp"
// #include <Inference.h>

namespace vpu {
namespace HDDL2Plugin {
namespace ie = InferenceEngine;

class ExecutableNetwork : public ie::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<ExecutableNetwork>;

    explicit ExecutableNetwork(
        InferenceEngine::ICNNNetwork& network, std::shared_ptr<vpux::IDevice>& device, const vpux::VPUXConfig& config);

    explicit ExecutableNetwork(
        std::istream& networkModel, std::shared_ptr<vpux::IDevice>& device, const vpux::VPUXConfig& config);
    ~ExecutableNetwork() override = default;

    ie::InferRequestInternal::Ptr CreateInferRequestImpl(
        const ie::InputsDataMap networkInputs, const ie::OutputsDataMap networkOutputs) override;
    void ExportImpl(std::ostream& model) override;
    void Export(std::ostream& networkModel) override { ExportImpl(networkModel); }

    using ie::ExecutableNetworkInternal::Export;
    void Export(const std::string& modelFileName) override;

    InferenceEngine::IInferRequest::Ptr CreateInferRequest() override;
    InferenceEngine::Parameter GetMetric(const std::string& name) const override;

    void GetMetric(const std::string& name, ie::Parameter& result, ie::ResponseDesc* resp) const override;

private:
    explicit ExecutableNetwork(const vpux::VPUXConfig& config);

private:
    const vpux::VPUXConfig _config;
    const Logger::Ptr _logger;

    vpux::Compiler::Ptr _compiler = nullptr;
    vpux::NetworkDescription::Ptr _networkPtr = nullptr;
    vpux::Executor::Ptr _executorPtr;
    std::vector<std::string> _supportedMetrics;
};

}  //  namespace HDDL2Plugin
}  //  namespace vpu
