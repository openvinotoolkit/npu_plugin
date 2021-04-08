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
Engine::Engine(): _backends(std::make_shared<VPUXBackends>(_parsedConfig)), _metrics(_backends) {
    _pluginName = DEVICE_NAME;  // "VPUX"
    const auto compiler = Compiler::create(_parsedConfig);
    _parsedConfig.expandSupportedCompileOptions(compiler->getSupportedOptions());
    _parsedConfig.expandSupportedRunTimeOptions(_backends == nullptr ? std::unordered_set<std::string>()
                                                                     : _backends->getSupportedOptions());
}

//------------------------------------------------------------------------------
//      Load network
//------------------------------------------------------------------------------
IE::ExecutableNetworkInternal::Ptr Engine::LoadExeNetwork(const IE::CNNNetwork& network,
                                                          std::shared_ptr<Device>& device,
                                                          const VPUXConfig& networkConfig) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "LoadExeNetwork");
    IE::CNNNetwork clonedNetwork = IE::details::cloneNetwork(network);

    return std::make_shared<ExecutableNetwork>(clonedNetwork, device, networkConfig);
}

IE::ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const IE::CNNNetwork& network,
                                                              const std::map<std::string, std::string>& config) {
    auto networkConfig = mergePluginAndNetworkConfigs(_parsedConfig, config);
    auto device = _backends->getDevice(networkConfig.deviceId());
    return LoadExeNetwork(network, device, networkConfig);
}

IE::ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const IE::CNNNetwork& network,
                                                              IE::RemoteContext::Ptr context,
                                                              const std::map<std::string, std::string>& config) {
    auto networkConfig = mergePluginAndNetworkConfigs(_parsedConfig, config);
    auto device = _backends->getDevice(context);
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
    VPUX_UNUSED(network);
    VPUX_UNUSED(config);
    IE_THROW(NotImplemented);
    return {};
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
