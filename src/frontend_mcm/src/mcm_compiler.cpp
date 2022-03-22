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

#include "mcm_compiler.hpp"

#include "emulator_network_description.hpp"
#include "mcm_network_description.hpp"
#include "ngraph_mcm_frontend/frontend.hpp"

#include "vpux/al/config/common.hpp"

using namespace vpux;
using namespace ov::intel_vpux;
using namespace InferenceEngine::VPUXConfigParams;

std::shared_ptr<INetworkDescription> MCMCompiler::compile(const std::shared_ptr<ngraph::Function>& func,
                                                          const std::string& netName,
                                                          const ie::InputsDataMap& inputsInfo,
                                                          const ie::OutputsDataMap& outputsInfo, const Config& config) {
    std::string errMsg;
    std::unique_ptr<mv::CompilationUnit> compilationUnit =
            compileNGraphIntoCompilationUnit(func, netName, inputsInfo, outputsInfo, config, errMsg);
    if (!compilationUnit)
        throw std::runtime_error(errMsg);

    std::vector<char> compiledNetwork = serializeCompilationUnit(compilationUnit, errMsg);
    if (compiledNetwork.empty())
        throw std::runtime_error(errMsg);

    if (config.get<PLATFORM>() == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR)
        return std::make_shared<vpu::MCMAdapter::EmulatorNetworkDescription>(std::move(compiledNetwork), config,
                                                                             netName);

    return std::make_shared<vpu::MCMAdapter::MCMNetworkDescription>(std::move(compiledNetwork), config, netName);
}

ie::QueryNetworkResult MCMCompiler::query(const ie::CNNNetwork& network, const Config& config) {
    ie::QueryNetworkResult result;
    const std::string plugin_name = "VPUX";

    auto supportedLayers = getSupportedLayers(network, config);

    for (auto&& layerName : *(supportedLayers.get())) {
        result.supportedLayersMap.emplace(layerName, plugin_name);
    }

    return result;
}

std::shared_ptr<INetworkDescription> MCMCompiler::parse(const std::vector<char>& compiledNetwork, const Config& config,
                                                        const std::string& graphName) {
    if (config.get<PLATFORM>() == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR)
        return std::make_shared<vpu::MCMAdapter::EmulatorNetworkDescription>(compiledNetwork, config, graphName);
    return std::make_shared<vpu::MCMAdapter::MCMNetworkDescription>(compiledNetwork, config, graphName);
}

INFERENCE_PLUGIN_API(void) CreateVPUXCompiler(std::shared_ptr<ICompiler>& obj) {
    obj = std::make_shared<MCMCompiler>();
}
