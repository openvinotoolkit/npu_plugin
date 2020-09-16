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

#include <ie_common.h>

#include <description_buffer.hpp>
#include <mcm_config.hpp>
#include <vpux_compiler.hpp>

class MCMCompiler final : public vpux::ICompiler {
public:
    std::shared_ptr<vpux::INetworkDescription> compile(
        InferenceEngine::ICNNNetwork& network, const vpux::VPUXConfig& config) override;

    std::shared_ptr<vpux::INetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
        const std::string& netName, const InferenceEngine::InputsDataMap& inputsInfo,
        const InferenceEngine::OutputsDataMap& outputsInfo, const vpux::VPUXConfig& config) override;

    std::shared_ptr<vpux::INetworkDescription> parse(
        const std::vector<char>& network, const vpux::VPUXConfig& config, const std::string& graphName = "") override;

    std::set<std::string> getSupportedLayers(InferenceEngine::ICNNNetwork& network) override;
    std::unordered_set<std::string> getSupportedOptions() override;

private:
    const vpu::MCMConfig _config = {};
};
