//
// Copyright 2020 Intel Corporation.
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

#include <ie_common.h>

#include <description_buffer.hpp>
#include <mcm_config.hpp>
#include <memory>
#include <vpux_compiler.hpp>

class MCMCompiler final : public vpux::ICompiler {
public:
    std::shared_ptr<vpux::INetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
                                                       const std::string& netName,
                                                       const InferenceEngine::InputsDataMap& inputsInfo,
                                                       const InferenceEngine::OutputsDataMap& outputsInfo,
                                                       const vpux::VPUXConfig& config) override;
    InferenceEngine::QueryNetworkResult query(const InferenceEngine::CNNNetwork& network,
                                              const vpux::VPUXConfig& config) override;

    std::shared_ptr<vpux::INetworkDescription> parse(const std::vector<char>& network, const vpux::VPUXConfig& config,
                                                     const std::string& graphName = "") override;

    std::unordered_set<std::string> getSupportedOptions() override;

private:
    const vpu::MCMConfig _config = {};
    const std::unique_ptr<vpu::Logger> _logger = std::unique_ptr<vpu::Logger>(
            new vpu::Logger("MCMCompiler", vpu::LogLevel::Debug /*_config.logLevel()*/, vpu::consoleOutput()));
};
