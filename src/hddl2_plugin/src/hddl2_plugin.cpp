//
// Copyright 2019 Intel Corporation.
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

// System include
#include <map>
#include <memory>
#include <string>

// Inference Engine include
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <details/ie_irelease.hpp>
#include <fstream>
#include <ie_icore.hpp>
#include <cnn_network_ngraph_impl.hpp>
#include <ie_util_internal.hpp>

// Plugin include
#include "hddl2_executable_network.h"
#include "hddl2_helpers.h"
#include "hddl2_plugin.h"
#include "hddl2_remote_context.h"

using namespace vpu::HDDL2Plugin;

Engine::Engine() { _pluginName = "HDDL2"; }

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
    const ICore* core, const ICNNNetwork& network, const std::map<std::string, std::string>& config) {
    UNUSED(core);
    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    std::shared_ptr<ICNNNetwork> clonedNetwork(nullptr);

    if (auto networkNGraph = dynamic_cast<const CNNNetworkNGraphImpl*>(&network)) {
        auto nGraphNetwork = networkNGraph->cloneNGraphImpl();
        clonedNetwork = nGraphNetwork->getCNNNetwork();
    } else {
        clonedNetwork = cloneNet(network);
    }

    ConstTransformer transformator(clonedNetwork.get());
    transformator.fullTrim();

    return std::make_shared<HDDL2Plugin::ExecutableNetwork>(*clonedNetwork, parsedConfigCopy);
}

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const ICore* core, const ICNNNetwork& network,
    RemoteContext::Ptr context, const std::map<std::string, std::string>& config) {
    UNUSED(core);
    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    std::shared_ptr<ICNNNetwork> clonedNetwork(nullptr);

    if (auto networkNGraph = dynamic_cast<const CNNNetworkNGraphImpl*>(&network)) {
        auto nGraphNetwork = networkNGraph->cloneNGraphImpl();
        clonedNetwork = nGraphNetwork->getCNNNetwork();
    } else {
        clonedNetwork = cloneNet(network);
    }

    ConstTransformer transformator(clonedNetwork.get());
    transformator.fullTrim();

    return std::make_shared<HDDL2Plugin::ExecutableNetwork>(*clonedNetwork, parsedConfigCopy, context);
}

IExecutableNetwork::Ptr Engine::ImportNetwork(
    const std::string& modelFileName, const std::map<std::string, std::string>& config) {
    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config, ConfigMode::RunTime);

    const auto executableNetwork = std::make_shared<ExecutableNetwork>(modelFileName, parsedConfigCopy);

    return IExecutableNetwork::Ptr(new ExecutableNetworkBase<ExecutableNetworkInternal>(executableNetwork),
        [](InferenceEngine::details::IRelease* p) {
            p->Release();
        });
}

InferenceEngine::ExecutableNetwork Engine::ImportNetworkImpl(
    std::istream& networkModel, const std::map<std::string, std::string>& config) {
    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config, ConfigMode::RunTime);

    const auto executableNetwork = std::make_shared<ExecutableNetwork>(networkModel, parsedConfigCopy);

    return InferenceEngine::ExecutableNetwork {
        IExecutableNetwork::Ptr(new ExecutableNetworkBase<ExecutableNetworkInternal>(executableNetwork),
            [](InferenceEngine::details::IRelease* p) {
                p->Release();
            })};
}

InferenceEngine::ExecutableNetwork Engine::ImportNetworkImpl(
    std::istream& networkModel, const RemoteContext::Ptr& context, const std::map<std::string, std::string>& config) {
    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config, ConfigMode::RunTime);

    const auto executableNetwork = std::make_shared<ExecutableNetwork>(networkModel, parsedConfigCopy, context);

    return InferenceEngine::ExecutableNetwork {
        IExecutableNetwork::Ptr(new ExecutableNetworkBase<ExecutableNetworkInternal>(executableNetwork),
            [](InferenceEngine::details::IRelease* p) {
                p->Release();
            })};
}

void Engine::SetConfig(const std::map<std::string, std::string>& config) {
    std::cout << "SetConfig call" << std::endl;
    UNUSED(config);
}

void Engine::QueryNetwork(const InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string>& config,
    InferenceEngine::QueryNetworkResult& res) const {
    UNUSED(network);
    UNUSED(config);
    UNUSED(res);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED;
}

RemoteContext::Ptr Engine::CreateContext(const ParamMap& map) {
    return std::make_shared<HDDL2Plugin::HDDL2RemoteContext>(map);
}

IE_SUPPRESS_DEPRECATED_START

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode)
CreatePluginEngine(IInferencePlugin*& plugin, ResponseDesc* resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin({{2, 1}, CI_BUILD_NUMBER, "HDDL2Plugin"}, std::make_shared<Engine>());
        return InferenceEngine::StatusCode ::OK;
    } catch (std::exception& ex) {
        return DescriptionBuffer(InferenceEngine::GENERAL_ERROR, resp) << ex.what();
    }
}

IE_SUPPRESS_DEPRECATED_END
