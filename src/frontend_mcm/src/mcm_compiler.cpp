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

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode)
CreateVPUXCompiler(vpux::ICompiler*& compiler, InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        compiler = new MCMCompiler();
        return InferenceEngine::StatusCode::OK;
    } catch (std::exception& ex) {
        return InferenceEngine::DescriptionBuffer(InferenceEngine::StatusCode::GENERAL_ERROR, resp) << ex.what();
    }
}
