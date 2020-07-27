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
#include <cnn_network_impl.hpp>
#include <convert_function_to_cnn_network.hpp>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <details/ie_irelease.hpp>
#include <fstream>
#include <generic_ie.hpp>
#include <ie_icore.hpp>
#include <ie_metric_helpers.hpp>
#include <ie_util_internal.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>

// Plugin include
#include "hddl2_executable_network.h"
#include "ie_macro.hpp"
#include "hddl2_params.hpp"
#include "hddl2_plugin.h"

using namespace vpu::HDDL2Plugin;

Engine::Engine() { _pluginName = "HDDL2"; }

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
    const ICNNNetwork& network, const std::map<std::string, std::string>& config) {
    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    std::shared_ptr<ICNNNetwork> clonedNetwork = cloneNetwork(network);

    if (auto nGraphFunc = clonedNetwork->getFunction()) {
        // Disable shape inference (WA for generic operations)
        ::ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

        // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
        ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(nGraphFunc);
        clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, *clonedNetwork);
    }

    auto implNetwork = std::dynamic_pointer_cast<CNNNetworkImpl>(clonedNetwork);
    if (implNetwork) {
        // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
        ConstTransformer transformator(implNetwork.get());
        transformator.fullTrim();
    }

    return std::make_shared<HDDL2Plugin::ExecutableNetwork>(*clonedNetwork, parsedConfigCopy);
}

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const ICNNNetwork& network,
    RemoteContext::Ptr context, const std::map<std::string, std::string>& config) {
    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    std::shared_ptr<ICNNNetwork> clonedNetwork = cloneNetwork(network);

    if (clonedNetwork->getFunction()) {
        auto nGraphFunc = clonedNetwork->getFunction();
        // Disable shape inference (WA for generic operations)
        ::ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

        // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
        ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(nGraphFunc);
        ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(nGraphFunc);
        clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, *clonedNetwork);
    }

    auto implNetwork = std::dynamic_pointer_cast<CNNNetworkImpl>(clonedNetwork);
    if (implNetwork) {
        // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
        ConstTransformer transformator(implNetwork.get());
        transformator.fullTrim();
    }

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
    _parsedConfig.update(config);

    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }
}

void Engine::QueryNetwork(const InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string>& config,
    InferenceEngine::QueryNetworkResult& res) const {
    UNUSED(network);
    UNUSED(config);
    UNUSED(res);
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED;
}

RemoteContext::Ptr Engine::CreateContext(const ParamMap& map) {
    return std::make_shared<HDDL2Plugin::HDDL2RemoteContext>(map, _parsedConfig);
}

InferenceEngine::Parameter Engine::GetMetric(
    const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& /*options*/) const {
    if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, HDDL2Metrics::GetAvailableDevicesNames());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, _metrics.SupportedMetrics());
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, _metrics.GetFullDevicesNames());
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, _metrics.GetSupportedConfigKeys());
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, _metrics.GetOptimizationCapabilities());
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, _metrics.GetRangeForAsyncInferRequest());
    } else if (name == METRIC_KEY(RANGE_FOR_STREAMS)) {
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, _metrics.GetRangeForStreams());
    }
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
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
