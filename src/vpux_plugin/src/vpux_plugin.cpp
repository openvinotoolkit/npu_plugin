//
// Copyright 2019 Intel Corporation.
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

// System include
#include <fstream>
#include <map>
#include <memory>
#include <string>

// Inference Engine include
#include <ie_ngraph_utils.hpp>
#include <ie_icore.hpp>
#include <ie_metric_helpers.hpp>
#include <ie_ngraph_utils.hpp>


// Plugin include
#include "file_reader.h"
#include "vpux.hpp"
#include "vpux_executable_network.h"
#include "vpux_metrics.h"
#include "vpux_plugin.h"
#include "vpux_remote_context.h"

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include <device_helpers.hpp>

namespace vpux {
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static VPUXConfig mergePluginAndNetworkConfigs(const VPUXConfig& pluginConfig,
                                               const std::map<std::string, std::string>& config) {
    auto parsedConfigCopy = pluginConfig;
    parsedConfigCopy.update(config);
    return parsedConfigCopy;
}

//------------------------------------------------------------------------------
// TODO: generation of available backends list can be done during execution of CMake scripts
static const std::vector<std::string> backendRegistry = {
#if defined(_WIN32) || defined(_WIN64)
        "zero_backend",
#endif
#if defined(__arm__) || defined(__aarch64__)
        "vpual_backend",
#endif
#if defined(ENABLE_HDDL2)
        "hddl2_backend",
#endif
#if defined(ENABLE_EMULATOR)
        "emulator_backend",
#endif
};
Engine::Engine(): _backends(std::make_shared<VPUXBackends>(backendRegistry)), _metrics(_backends) {
    _pluginName = DEVICE_NAME;  // "VPUX"
    const auto compiler = Compiler::create(_parsedConfig);
    _parsedConfig.expandSupportedCompileOptions(compiler->getSupportedOptions());
    _parsedConfig.expandSupportedRunTimeOptions(_backends->getSupportedOptions());
}

//------------------------------------------------------------------------------
//      Load network
//------------------------------------------------------------------------------
IE::ExecutableNetworkInternal::Ptr Engine::LoadExeNetwork(const IE::CNNNetwork& network,
                                                          std::shared_ptr<Device>& device,
                                                          const VPUXConfig& networkConfig) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "LoadExeNetwork");
    try {
        return std::make_shared<ExecutableNetwork>(network, device, networkConfig);
    } catch (const std::exception&) {
        throw;
    } catch (...) {
        IE_THROW() << "VPUX LoadExeNetwork got unexpected exception from ExecutableNetwork";
    }
}

IE::ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const IE::CNNNetwork& network,
                                                              const std::map<std::string, std::string>& config) {
    auto networkConfig = mergePluginAndNetworkConfigs(_parsedConfig, config);
    auto device = _backends->getDevice(networkConfig.deviceId());
    if (device != nullptr && _backends->getBackendName() == "HDDL2") {
        const auto platform = _backends->getCompilationPlatform(networkConfig.platform());
        networkConfig.update({{VPUX_CONFIG_KEY(PLATFORM), platform}});
    }

    return LoadExeNetwork(network, device, networkConfig);
}

IE::ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const IE::CNNNetwork& network,
                                                              IE::RemoteContext::Ptr context,
                                                              const std::map<std::string, std::string>& config) {
    auto networkConfig = mergePluginAndNetworkConfigs(_parsedConfig, config);
    auto device = _backends->getDevice(context);
    if (device != nullptr && _backends->getBackendName() == "HDDL2") {
        const auto platform = _backends->getCompilationPlatform(networkConfig.platform());
        networkConfig.update({{VPUX_CONFIG_KEY(PLATFORM), platform}});
    }

    return LoadExeNetwork(network, device, networkConfig);
}

//------------------------------------------------------------------------------
//      Import network
//------------------------------------------------------------------------------
IE::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(const std::string& modelFileName,
                                                          const std::map<std::string, std::string>& config) {
    std::ifstream blobStream(modelFileName, std::ios::binary);
    return ImportNetworkImpl(vpu::KmbPlugin::utils::skipMagic(blobStream), config);
}

IE::ExecutableNetworkInternal::Ptr Engine::ImportNetworkImpl(std::istream& networkModel,
                                                             const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "ImportNetwork");
    auto networkConfig = mergePluginAndNetworkConfigs(_parsedConfig, config);
    auto device = _backends->getDevice(networkConfig.deviceId());
    return std::make_shared<ExecutableNetwork>(networkModel, device, networkConfig);
}

IE::ExecutableNetworkInternal::Ptr Engine::ImportNetworkImpl(std::istream& networkModel, const IE::RemoteContext::Ptr& context,
                                                             const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "ImportNetwork");
    auto networkConfig = mergePluginAndNetworkConfigs(_parsedConfig, config);
    auto device = _backends->getDevice(context);
    return std::make_shared<ExecutableNetwork>(networkModel, device, networkConfig);
}

//------------------------------------------------------------------------------
void Engine::SetConfig(const std::map<std::string, std::string>& config) {
    _parsedConfig.update(config);
    if (_backends != nullptr)
        _backends->setup(_parsedConfig);
    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }
}

IE::QueryNetworkResult Engine::QueryNetwork(const IE::CNNNetwork& network,
                                            const std::map<std::string, std::string>& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "QueryNetwork");

    if (nullptr == network.getFunction()) {
         IE_THROW() << "VPUX Plugin supports only ngraph cnn network representation";
    }

    auto networkConfig = mergePluginAndNetworkConfigs(_parsedConfig, config);
    Compiler::Ptr compiler = Compiler::create(networkConfig);

    return compiler->query(network, networkConfig);
}

IE::RemoteContext::Ptr Engine::CreateContext(const IE::ParamMap& map) {
    // Device in this case will be searched inside RemoteContext creation
    const auto device = _backends->getDevice(map);
    if (device == nullptr) {
        IE_THROW() << "CreateContext: Failed to find suitable device to use";
    }
    return std::make_shared<VPUXRemoteContext>(device, map, _parsedConfig);
}

IE::Parameter Engine::GetConfig(const std::string& name,
                                const std::map<std::string, IE::Parameter>& /*options*/) const {
    if (name == CONFIG_KEY(LOG_LEVEL)) {
        return IE::Parameter(static_cast<int>(_parsedConfig.logLevel()));
    } else if (name == CONFIG_KEY(PERF_COUNT)) {
        return IE::Parameter(_parsedConfig.performanceCounting());
    } else if (name == CONFIG_KEY(DEVICE_ID)) {
        return IE::Parameter(_parsedConfig.deviceId());
    } else if ((name == VPUX_CONFIG_KEY(THROUGHPUT_STREAMS)) || (name == KMB_CONFIG_KEY(THROUGHPUT_STREAMS))) {
        return IE::Parameter(_parsedConfig.throughputStreams());
    } else if (name == VPUX_CONFIG_KEY(INFERENCE_SHAVES)) {
        return IE::Parameter(_parsedConfig.numberOfNnCoreShaves());
    } else if (name == VPUX_CONFIG_KEY(PLATFORM)) {
        return IE::Parameter(static_cast<int>(_parsedConfig.platform()));
    } else {
        IE_THROW(NotImplemented);
    }
}

IE::Parameter Engine::GetMetric(const std::string& name,
                                const std::map<std::string, IE::Parameter>& /*options*/) const {
    if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, _metrics.GetAvailableDevicesNames());
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
    } else if (name == METRIC_KEY(IMPORT_EXPORT_SUPPORT)) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    }
    IE_THROW(NotImplemented);
}

static const IE::Version version = {{2, 1}, CI_BUILD_NUMBER, "VPUXPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, version)

}  // namespace vpux
