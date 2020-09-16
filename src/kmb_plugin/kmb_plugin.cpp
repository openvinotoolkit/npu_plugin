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

#include <graph_transformer.h>
#include <kmb_remote_context.h>

#include <cnn_network_impl.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <ie_itt.hpp>
#include <ie_util_internal.hpp>
#include <inference_engine.hpp>
#include <memory>
#include <vector>
#include <vpu/kmb_plugin_config.hpp>

#include "file_reader.h"
#include "ie_macro.hpp"

using namespace InferenceEngine;
using namespace vpu::KmbPlugin;

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
    const ICNNNetwork& network, const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "LoadExeNetworkImp");
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

    const auto& devices = _backend->getDevices();
    bool isDeviceExist = devices.find(parsedConfigCopy.deviceId()) != devices.end();
    const std::shared_ptr<vpux::Device> device = isDeviceExist ? devices.at(parsedConfigCopy.deviceId()) : nullptr;
    bool foundCSRAM = devices.find(vpux::CSRAM_DEVICE_ID) != devices.end();
    const std::shared_ptr<vpux::Device> CSRAMdev = foundCSRAM ? devices.at(vpux::CSRAM_DEVICE_ID) : nullptr;

    return std::make_shared<ExecutableNetwork>(*clonedNetwork, parsedConfigCopy, device, CSRAMdev);
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

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    auto copyNet = ie::CNNNetwork(InferenceEngine::cloneNet(network));

    auto layerNames = _compiler->getSupportedLayers(copyNet);

    for (auto&& layerName : layerNames) {
        res.supportedLayersMap.insert({layerName, GetName()});
    }
}

Engine::Engine()
    : _backend(vpux::EngineBackendConfigurator::findBackend()),
      _metrics(KmbMetrics(_backend->getDevices())),
      _defaultContextMap({}) {
    _pluginName = DEVICE_NAME;  //"KMB";
    _compiler = vpux::Compiler::create(vpux::CompilerType::MCMCompiler);
    _parsedConfig.expandSupportedOptions(_compiler->getSupportedOptions());
}

IExecutableNetwork::Ptr Engine::ImportNetwork(
    const std::string& modelFileName, const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "ImportNetwork");
    std::ifstream blobStream(modelFileName, std::ios::binary);
    return ImportNetworkImpl(utils::skipMagic(blobStream), config);
}

InferenceEngine::ExecutableNetwork Engine::ImportNetworkImpl(
    std::istream& networkModel, const std::map<std::string, std::string>& config) {
    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    const auto& devices = _backend->getDevices();
    bool isDeviceExist = devices.find(parsedConfigCopy.deviceId()) != devices.end();
    const std::shared_ptr<vpux::Device> device = isDeviceExist ? devices.at(parsedConfigCopy.deviceId()) : nullptr;
    bool foundCSRAM = devices.find(vpux::CSRAM_DEVICE_ID) != devices.end();
    const std::shared_ptr<vpux::Device> CSRAMdev = foundCSRAM ? devices.at(vpux::CSRAM_DEVICE_ID) : nullptr;
    const auto executableNetwork =
        std::make_shared<ExecutableNetwork>(networkModel, parsedConfigCopy, device, CSRAMdev);

    return InferenceEngine::make_executable_network(executableNetwork);
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
    const auto& deviceId = map.at(InferenceEngine::KMB_PARAM_KEY(DEVICE_ID));
    const auto& device = _backend->getDevices().at(deviceId);
    return std::make_shared<KmbPlugin::KmbRemoteContext>(map, _parsedConfig, device);
}

InferenceEngine::ExecutableNetwork Engine::ImportNetworkImpl(
    std::istream& networkModel, const RemoteContext::Ptr& ctx, const std::map<std::string, std::string>& config) {
    if (std::dynamic_pointer_cast<KmbRemoteContext>(ctx) == nullptr) {
        THROW_IE_EXCEPTION << "remote context is not compatible";
    }

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    const auto& devices = _backend->getDevices();
    bool isDeviceExist = devices.find(parsedConfigCopy.deviceId()) != devices.end();
    const std::shared_ptr<vpux::Device> device = isDeviceExist ? devices.at(parsedConfigCopy.deviceId()) : nullptr;
    bool foundCSRAM = devices.find(vpux::CSRAM_DEVICE_ID) != devices.end();
    const std::shared_ptr<vpux::Device> CSRAMdev = foundCSRAM ? devices.at(vpux::CSRAM_DEVICE_ID) : nullptr;
    const auto executableNetwork =
        std::make_shared<ExecutableNetwork>(networkModel, parsedConfigCopy, device, CSRAMdev);

    return InferenceEngine::make_executable_network(executableNetwork);
}

RemoteContext::Ptr Engine::GetDefaultContext(const std::string& deviceId) {
    std::lock_guard<std::mutex> contextCreateGuard(_contextCreateMutex);
    auto defaultCtxIter = _defaultContextMap.find(deviceId);
    if (defaultCtxIter == _defaultContextMap.end()) {
        const ParamMap ctxParams = {
            {InferenceEngine::KMB_PARAM_KEY(DEVICE_ID), deviceId},
        };

        const auto& device = _backend->getDevices().at(deviceId);
        _defaultContextMap[deviceId] = std::make_shared<KmbPlugin::KmbRemoteContext>(ctxParams, _parsedConfig, device);
    }
    return std::dynamic_pointer_cast<RemoteContext>(_defaultContextMap.at(deviceId));
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "kmbPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, version)
