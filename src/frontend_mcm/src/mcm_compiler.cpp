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

#include <emulator_network_description.hpp>
#include <mcm_adapter.hpp>
#include <mcm_compiler.hpp>
#include <mcm_network_description.hpp>
#include <ngraph_mcm_frontend/frontend.hpp>

std::shared_ptr<vpux::INetworkDescription> MCMCompiler::compile(const std::shared_ptr<ngraph::Function>& func,
                                                                const std::string& netName,
                                                                const ie::InputsDataMap& inputsInfo,
                                                                const ie::OutputsDataMap& outputsInfo,
                                                                const vpux::VPUXConfig& config) {
    auto copy = _config;
    copy.parseFrom(config);
    std::string errMsg;
    std::unique_ptr<mv::CompilationUnit> compilationUnit =
            compileNGraphIntoCompilationUnit(func, netName, inputsInfo, outputsInfo, copy, errMsg);
    if (!compilationUnit)
        throw std::runtime_error(errMsg);
    if (config.deviceId() == "EMULATOR")
        return std::make_shared<vpu::MCMAdapter::EmulatorNetworkDescription>(std::move(compilationUnit), copy, netName);
    std::vector<char> compiledNetwork = serializeCompilationUnit(compilationUnit, errMsg);
    if (compiledNetwork.empty())
        throw std::runtime_error(errMsg);
    return std::make_shared<vpu::MCMAdapter::MCMNetworkDescription>(std::move(compiledNetwork), copy, netName);
}

InferenceEngine::QueryNetworkResult MCMCompiler::query(const InferenceEngine::CNNNetwork& /*network*/,
                                                       const vpux::VPUXConfig& /*config*/) {
    InferenceEngine::QueryNetworkResult result;
    return result;
}

std::shared_ptr<vpux::INetworkDescription> MCMCompiler::parse(const std::vector<char>& compiledNetwork,
                                                              const vpux::VPUXConfig& config,
                                                              const std::string& graphName) {
    auto copy = _config;
    copy.parseFrom(config);

    return std::make_shared<vpu::MCMAdapter::MCMNetworkDescription>(compiledNetwork, copy, graphName);
}

std::unordered_set<std::string> MCMCompiler::getSupportedOptions() {
    return _config.getCompileOptions();
}

INFERENCE_PLUGIN_API(void)
CreateVPUXCompiler(std::shared_ptr<vpux::ICompiler>& compiler) {
    compiler = std::make_shared<MCMCompiler>();
}
