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
#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
// Plugin
#include "hddl2_config.h"
#include "vpux.hpp"

namespace vpu {
namespace HDDL2Plugin {

class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<ExecutableNetwork>;

    explicit ExecutableNetwork(
        InferenceEngine::ICNNNetwork& network, std::shared_ptr<vpux::IDevice>& device, const vpux::VPUXConfig& config);

    explicit ExecutableNetwork(
        std::istream& networkModel, std::shared_ptr<vpux::IDevice>& device, const vpux::VPUXConfig& config);
    ~ExecutableNetwork() override = default;

    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(
        const InferenceEngine::InputsDataMap networkInputs,
        const InferenceEngine::OutputsDataMap networkOutputs) override;
    void ExportImpl(std::ostream& model) override;
    void Export(std::ostream& networkModel) override { ExportImpl(networkModel); }

    using InferenceEngine::ExecutableNetworkInternal::Export;
    void Export(const std::string& modelFileName) override;

    InferenceEngine::IInferRequest::Ptr CreateInferRequest() override;

private:
    explicit ExecutableNetwork(const vpux::VPUXConfig& config);

private:
    const vpux::VPUXConfig _config;
    const Logger::Ptr _logger;

    vpux::Compiler::Ptr _compiler = nullptr;
    vpux::NetworkDescription::Ptr _networkPtr = nullptr;
    vpux::Executor::Ptr _executorPtr;
};

}  //  namespace HDDL2Plugin
}  //  namespace vpu
