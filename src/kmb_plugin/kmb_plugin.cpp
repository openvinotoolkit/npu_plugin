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

#include <memory>
#include <vector>

#include <inference_engine.hpp>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>

#include <vpu/kmb_plugin_config.hpp>

#include "kmb_plugin.h"

using namespace InferenceEngine;
using namespace vpu::KmbPlugin;

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const ICore * /*core*/, ICNNNetwork &network,
                                                          const std::map<std::string, std::string> &config) {
    InputsDataMap networkInputs;
    OutputsDataMap networkOutputs;

    network.getInputsInfo(networkInputs);
    network.getOutputsInfo(networkOutputs);

    for (auto networkInput : networkInputs) {
        auto input_precision = networkInput.second->getPrecision();

        if (input_precision != Precision::FP16
            && input_precision != Precision::FP32
            && input_precision != Precision::U8) {
            THROW_IE_EXCEPTION << "Input image format " << input_precision << " is not supported yet.\n"
                               << "Supported formats:F16, FP32 and U8.";
        }
    }

    // override what was set globally for plugin, otherwise - override default config without touching config for plugin
    auto configCopy = _config;
    for (auto &&entry : config) {
        configCopy[entry.first] = entry.second;
    }

    return std::make_shared<ExecutableNetwork>(network, configCopy);
}

void Engine::SetConfig(const std::map<std::string, std::string> &userConfig) {
    KmbConfig kmbConfig(userConfig);

    for (auto &&entry : userConfig) {
        _config[entry.first] = entry.second;
    }
}

void Engine::QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                          QueryNetworkResult& res) const {
    UNUSED(config);
#ifdef ENABLE_MCM_COMPILER
    std::shared_ptr<mv::CompilationUnit> tmpCompiler =
            std::make_shared<mv::CompilationUnit>(network.getName());
    if (tmpCompiler == nullptr) {
        THROW_IE_EXCEPTION << "CompilationUnit have not been created.\n"
                       << "Supported format: FP32 and FP16.";
    }

    auto copyNet = ie::CNNNetwork(InferenceEngine::cloneNet(network));
    auto layerNames = getSupportedLayersMcm(copyNet,
        tmpCompiler->model(),
        config);

    for (auto && layerName : layerNames) {
        res.supportedLayersMap.insert({ layerName, GetName() });
    }

#else
    UNUSED(network);
    UNUSED(res);
#endif
}

Engine::Engine() {
    _pluginName = "KMB";

    KmbConfig config;
    _config = config.getDefaultConfig();
#ifdef ENABLE_MCM_COMPILER
    std::shared_ptr<mv::CompilationUnit> tmpCompiler =
            std::make_shared<mv::CompilationUnit>("testModel");
    if (tmpCompiler == nullptr) {
        THROW_IE_EXCEPTION << "CompilationUnit have not been created.\n"
                       << "Supported format: FP32 and FP16.";
    }
#endif
}

// TODO: ImportNetwork and LoadNetwork handle the config parameter in different ways.
// ImportNetwork gets a config provided by an user. LoadNetwork gets the plugin config and merge it with user's config.
// Need to found a common way to handle configs
IExecutableNetwork::Ptr Engine::ImportNetwork(const std::string &modelFileName, const std::map<std::string, std::string> &config) {
    std::ifstream blobFile(modelFileName, std::ios::binary);

    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << NETWORK_NOT_READ;
    }

    IExecutableNetwork::Ptr executableNetwork;
    // Use config provided by an user ignoring default config
    executableNetwork.reset(new ExecutableNetworkBase<ExecutableNetworkInternal>(
                                std::make_shared<ExecutableNetwork>(modelFileName, config)),
                                [](InferenceEngine::details::IRelease *p) {p->Release();});

    return executableNetwork;
}

IE_SUPPRESS_DEPRECATED_START

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin *&plugin, ResponseDesc *resp) noexcept {
    std::map<std::string, std::string> config;
//    config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_DEBUG);
    try {
        plugin = make_ie_compatible_plugin({{2, 1}, CI_BUILD_NUMBER, "kmbPlugin"}, std::make_shared<Engine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}

IE_SUPPRESS_DEPRECATED_END
