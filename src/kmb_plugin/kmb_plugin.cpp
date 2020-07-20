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

#include "kmb_plugin.h"

#include <kmb_remote_context.h>

#include <cnn_network_impl.hpp>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <ie_util_internal.hpp>
#include <inference_engine.hpp>
#include <memory>
#include <vector>
#include <vpu/kmb_plugin_config.hpp>

#include "ie_macro.hpp"

using namespace InferenceEngine;
using namespace vpu::KmbPlugin;

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
    const ICNNNetwork& network, const std::map<std::string, std::string>& config) {
    IE_PROFILING_AUTO_SCOPE(LoadExeNetworkImpl);
    InputsDataMap networkInputs;
    OutputsDataMap networkOutputs;

    network.getInputsInfo(networkInputs);
    network.getOutputsInfo(networkOutputs);

    for (auto networkInput : networkInputs) {
        auto input_precision = networkInput.second->getPrecision();

        if (input_precision != Precision::FP16 && input_precision != Precision::FP32 &&
            input_precision != Precision::U8) {
            THROW_IE_EXCEPTION << "Input image format " << input_precision << " is not supported yet.\n"
                               << "Supported formats:F16, FP32 and U8.";
        }
    }

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    std::shared_ptr<ICNNNetwork> clonedNetwork = cloneNetwork(network);

    auto implNetwork = std::dynamic_pointer_cast<CNNNetworkImpl>(clonedNetwork);
    if (implNetwork) {
        // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
        ConstTransformer transformator(implNetwork.get());
        transformator.fullTrim();
    }

    return std::make_shared<ExecutableNetwork>(
        *clonedNetwork, parsedConfigCopy, GetDefaultContext(parsedConfigCopy.deviceId()));
}

void Engine::SetConfig(const std::map<std::string, std::string>& config) {
    _parsedConfig.update(config);

    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }
}

void Engine::QueryNetwork(
    const ICNNNetwork& network, const std::map<std::string, std::string>& config, QueryNetworkResult& res) const {
    if (network.getFunction()) {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str << " ngraph::Function is not supported nativelly";
    }
#ifdef ENABLE_MCM_COMPILER
    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    auto copyNet = ie::CNNNetwork(InferenceEngine::cloneNet(network));
    auto layerNames = MCMAdapter::getSupportedLayers(copyNet, parsedConfigCopy);

    for (auto&& layerName : layerNames) {
        res.supportedLayersMap.insert({layerName, GetName()});
    }

#else
    UNUSED(network);
    UNUSED(res);
    UNUSED(config);
#endif
}

Engine::Engine(): _metrics(), _defaultContext({}) {
    _pluginName = "KMB";

#ifdef ENABLE_MCM_COMPILER
    if (!MCMAdapter::isMCMCompilerAvailable()) {
        THROW_IE_EXCEPTION << "Compiler not found";
    }
#endif
}

IExecutableNetwork::Ptr Engine::ImportNetwork(
    const std::string& modelFileName, const std::map<std::string, std::string>& config) {
    IE_PROFILING_AUTO_SCOPE(ImportNetwork);
    std::ifstream blobFile(modelFileName, std::ios::binary);

    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << NETWORK_NOT_READ;
    }

    InferenceEngine::ExportMagic magic = {};
    blobFile.seekg(0, blobFile.beg);
    blobFile.read(magic.data(), magic.size());
    auto exportedWithName = (exportMagic == magic);
    if (exportedWithName) {
        std::string tmp;
        std::getline(blobFile, tmp);
    } else {
        blobFile.seekg(0, blobFile.beg);
    }

    return ImportNetworkImpl(blobFile, config);
}

InferenceEngine::ExecutableNetwork Engine::ImportNetworkImpl(
    std::istream& networkModel, const std::map<std::string, std::string>& config) {
    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config, ConfigMode::RunTime);

    const auto executableNetwork = std::make_shared<ExecutableNetwork>(
        networkModel, parsedConfigCopy, GetDefaultContext(parsedConfigCopy.deviceId()));

    return InferenceEngine::ExecutableNetwork{IExecutableNetwork::Ptr(
        new ExecutableNetworkBase<ExecutableNetworkInternal>(executableNetwork), [](ie::details::IRelease* p) {
            p->Release();
        })};
}

InferenceEngine::Parameter Engine::GetMetric(
    const std::string& name, const std::map<std::string, InferenceEngine::Parameter>&) const {
    if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, _metrics.AvailableDevicesNames());
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

RemoteContext::Ptr Engine::CreateContext(const ParamMap& map) {
    return std::make_shared<KmbPlugin::KmbRemoteContext>(map, _parsedConfig);
}

InferenceEngine::ExecutableNetwork Engine::ImportNetworkImpl(
    std::istream& networkModel, const RemoteContext::Ptr& ctx, const std::map<std::string, std::string>& config) {
    if (std::dynamic_pointer_cast<KmbRemoteContext>(ctx) == nullptr) {
        THROW_IE_EXCEPTION << "Remote context is not compatible";
    }

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config, ConfigMode::RunTime);

    const auto executableNetwork = std::make_shared<ExecutableNetwork>(networkModel, parsedConfigCopy, ctx);

    return InferenceEngine::ExecutableNetwork{IExecutableNetwork::Ptr(
        new ExecutableNetworkBase<ExecutableNetworkInternal>(executableNetwork), [](ie::details::IRelease* p) {
            p->Release();
        })};
}

RemoteContext::Ptr Engine::GetDefaultContext() {
    std::lock_guard<std::mutex> contextCreateGuard(_contextCreateMutex);
    std::string firstAvaliableDevice = "";
    if (_defaultContext.empty()) {
        std::vector<std::string> deviceIdList = GetMetric(METRIC_KEY(AVAILABLE_DEVICES), {});
        if (deviceIdList.empty()) {
            // TODO throw exception when no available devices found
            // vpualHost must call VpualDispatcherResource destructor in free the slice properly
            // however, VpualDispatcherResource is implemented as a static object
            // thus, it is destroyed when libVpualDispatcher.so is unloaded from memory
            // which never happens unless vpualHost is compiled with --no-gnu-unique
            // hard code first device name as a fallback now
            firstAvaliableDevice = "vpu-slice-0";
        } else {
            firstAvaliableDevice = deviceIdList.at(0);
        }
        // assign the first available device
        const ParamMap ctxParams = {
            {InferenceEngine::KMB_PARAM_KEY(DEVICE_ID), firstAvaliableDevice},
        };
        _defaultContext[firstAvaliableDevice] = std::make_shared<KmbPlugin::KmbRemoteContext>(ctxParams, _parsedConfig);
    }
    return std::dynamic_pointer_cast<RemoteContext>(_defaultContext.at(firstAvaliableDevice));
}

RemoteContext::Ptr Engine::GetDefaultContext(const std::string& deviceId) {
    std::lock_guard<std::mutex> contextCreateGuard(_contextCreateMutex);
    auto defaultCtxIter = _defaultContext.find(deviceId);
    if (defaultCtxIter == _defaultContext.end()) {
        const ParamMap ctxParams = {
            {InferenceEngine::KMB_PARAM_KEY(DEVICE_ID), deviceId},
        };
        _defaultContext[deviceId] = std::make_shared<KmbPlugin::KmbRemoteContext>(ctxParams, _parsedConfig);
    }
    return std::dynamic_pointer_cast<RemoteContext>(_defaultContext.at(deviceId));
}

IE_SUPPRESS_DEPRECATED_START

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin*& plugin, ResponseDesc* resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin({{2, 1}, CI_BUILD_NUMBER, "kmbPlugin"}, std::make_shared<Engine>());
        return OK;
    } catch (std::exception& ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}

IE_SUPPRESS_DEPRECATED_END
