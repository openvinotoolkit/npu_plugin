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

#include <mcm_adapter.hpp>
#include <mcm_compiler.hpp>
#include <mcm_network_description.hpp>
#include <ngraph_mcm_frontend/frontend.hpp>

std::shared_ptr<vpux::NetworkDescription> MCMCompiler::compile(
    InferenceEngine::ICNNNetwork& network, const vpux::VPUXConfig& config) {
    std::vector<char> compiledNetwork;
    auto copy = _config;
    copy.parseFrom(config);

    vpu::MCMAdapter::compileNetwork(network, copy, compiledNetwork);
    return std::make_shared<vpu::MCMAdapter::MCMNetworkDescription>(compiledNetwork, copy, network.getName());
}

std::shared_ptr<vpux::NetworkDescription> MCMCompiler::compile(const std::shared_ptr<ngraph::Function>& func,
    const std::string& netName, const ie::InputsDataMap& inputsInfo, const ie::OutputsDataMap& outputsInfo,
    const vpux::VPUXConfig& config) {
    auto copy = _config;
    copy.parseFrom(config);

    auto compiledNetwork = compileNGraph(func, netName, inputsInfo, outputsInfo, copy);
    return std::make_shared<vpu::MCMAdapter::MCMNetworkDescription>(compiledNetwork, copy);
}

std::shared_ptr<vpux::NetworkDescription> MCMCompiler::parse(
    const std::vector<char>& compiledNetwork, const vpux::VPUXConfig& config, const std::string& graphName) {
    auto copy = _config;
    copy.parseFrom(config);

    return std::make_shared<vpu::MCMAdapter::MCMNetworkDescription>(compiledNetwork, copy, graphName);
}

std::set<std::string> MCMCompiler::getSupportedLayers(InferenceEngine::ICNNNetwork& network) {
    return vpu::MCMAdapter::getSupportedLayers(network, _config);
}

std::unordered_set<std::string> MCMCompiler::getSupportedOptions() { return _config.getCompileOptions(); }

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode)
CreateVPUXCompiler(vpux::ICompiler*& compiler, InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        compiler = new MCMCompiler();
        return InferenceEngine::StatusCode::OK;
    } catch (std::exception& ex) {
        return InferenceEngine::DescriptionBuffer(InferenceEngine::StatusCode::GENERAL_ERROR, resp) << ex.what();
    }
}
