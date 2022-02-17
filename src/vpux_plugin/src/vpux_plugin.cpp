//
// Copyright 2021 Intel Corporation.
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
#include <ie_icore.hpp>
#include <ie_metric_helpers.hpp>
#include <ie_ngraph_utils.hpp>
#include <openvino/runtime/properties.hpp>

// Plugin include
#include "file_reader.h"
#include "vpux.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/al/config/mcm_compiler.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux_executable_network.h"
#include "vpux_metrics.h"
#include "vpux_plugin.h"
#include "vpux_private_metrics.hpp"
#include "vpux_remote_context.h"

#include <device_helpers.hpp>
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/plugin/plugin_name.hpp"

namespace vpux {
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static Config mergeConfigs(const Config& globalConfig, const std::map<std::string, std::string>& rawConfig,
                           OptionMode mode = OptionMode::Both) {
    if (globalConfig.get<PLATFORM>() == IE::VPUXConfigParams::VPUXPlatform::EMULATOR) {
        const auto deviceIdPlatform = rawConfig.find(CONFIG_KEY(DEVICE_ID));

        if (deviceIdPlatform != rawConfig.end() && globalConfig.has<DEVICE_ID>()) {
            if (deviceIdPlatform->second != globalConfig.get<DEVICE_ID>())
                VPUX_THROW("mergePluginAndNetworkConfigs: device id platform does not match platform config key for "
                           "emulator: {0} and {1}",
                           deviceIdPlatform->second, globalConfig.get<DEVICE_ID>());
        }
    }

    Config localConfig = globalConfig;
    localConfig.update(rawConfig, mode);
    return localConfig;
}

//------------------------------------------------------------------------------

Engine::Engine()
        : _options(std::make_shared<OptionsDesc>()), _globalConfig(_options), _logger("VPUXEngine", LogLevel::Error) {
    _pluginName = "VPUX";

    registerCommonOptions(*_options);
    registerCompilerOptions(*_options);
    registerMcmCompilerOptions(*_options);
    registerRunTimeOptions(*_options);

    // TODO: generation of available backends list can be done during execution of CMake scripts
    std::vector<std::string> backendRegistry;

#if defined(OPENVINO_STATIC_LIBRARY)
    backendRegistry.push_back("vpux_level_zero_backend");
#else

#if defined(ENABLE_IMD_BACKEND)
    if (const auto* envVar = std::getenv("IE_VPUX_USE_IMD_BACKEND")) {
        if (envVarStrToBool("IE_VPUX_USE_IMD_BACKEND", envVar)) {
            backendRegistry.push_back("vpux_imd_backend");
        }
    }
#endif

#if defined(_WIN32) || defined(_WIN64) || (defined(__linux__) && defined(__x86_64__))
    backendRegistry.push_back("vpux_level_zero_backend");
#endif
#if defined(__arm__) || defined(__aarch64__)
    backendRegistry.push_back("vpux_vpual_backend");
#endif
#if defined(ENABLE_HDDL2)
    backendRegistry.push_back("vpux_hddl2_backend");
#endif
#if defined(ENABLE_EMULATOR)
    backendRegistry.push_back("vpux_emulator_backend");
#endif

#endif

    _backends = std::make_shared<VPUXBackends>(backendRegistry);
    _backends->registerOptions(*_options);

    _metrics = std::make_unique<Metrics>(_backends);

    _globalConfig.parseEnvVars();
    Logger::global().setLevel(_globalConfig.get<LOG_LEVEL>());
}

//------------------------------------------------------------------------------
//      Load network
//------------------------------------------------------------------------------
IE::IExecutableNetworkInternal::Ptr Engine::LoadExeNetwork(const IE::CNNNetwork& network,
                                                           std::shared_ptr<Device>& device,
                                                           const Config& networkConfig) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "LoadExeNetwork");
    try {
        return std::make_shared<ExecutableNetwork>(network, device, networkConfig);
    } catch (const std::exception&) {
        throw;
    } catch (...) {
        IE_THROW() << "VPUX LoadExeNetwork got unexpected exception from ExecutableNetwork";
    }
}

IE::IExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const IE::CNNNetwork& network,
                                                               const std::map<std::string, std::string>& config) {
    auto localConfig = mergeConfigs(_globalConfig, config);

    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    auto device = _backends->getDevice(localConfig.get<DEVICE_ID>());
    localConfig.update({{VPUX_CONFIG_KEY(PLATFORM), platform}});

    return LoadExeNetwork(network, device, localConfig);
}

IE::IExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const IE::CNNNetwork& network,
                                                               const IE::RemoteContext::Ptr& context,
                                                               const std::map<std::string, std::string>& config) {
    auto localConfig = mergeConfigs(_globalConfig, config);

    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    auto device = _backends->getDevice(context);
    localConfig.update({{VPUX_CONFIG_KEY(PLATFORM), platform}});

    return LoadExeNetwork(network, device, localConfig);
}

//------------------------------------------------------------------------------
//      Import network
//------------------------------------------------------------------------------
IE::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(const std::string& modelFileName,
                                                          const std::map<std::string, std::string>& config) {
    std::ifstream blobStream(modelFileName, std::ios::binary);
#if defined(__arm__) || defined(__aarch64__)
    try {
        if (_encryptionModel.isLibraryFound()) {
            std::stringstream sstream;
            return ImportNetwork(
                    vpu::KmbPlugin::utils::skipMagic(_encryptionModel.getDecryptedStream(blobStream, sstream)), config);
        }
    } catch (const std::exception& ex) {
        _logger.warning("{0}", ex.what());
    }
#endif
    return ImportNetwork(vpu::KmbPlugin::utils::skipMagic(blobStream), config);
}

IE::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(std::istream& networkModel,
                                                          const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "ImportNetwork");
    try {
        auto localConfig = mergeConfigs(_globalConfig, config, OptionMode::RunTime);
        auto device = _backends->getDevice(localConfig.get<DEVICE_ID>());
        const auto executableNetwork = std::make_shared<ExecutableNetwork>(networkModel, device, localConfig);
        executableNetwork->SetPointerToPlugin(shared_from_this());
        return executableNetwork;
    } catch (const std::exception&) {
        throw;
    } catch (...) {
        IE_THROW() << "VPUX ImportNetwork got unexpected exception from ExecutableNetwork";
    }
}

IE::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(std::istream& networkModel,
                                                          const IE::RemoteContext::Ptr& context,
                                                          const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "ImportNetwork");
    try {
        auto localConfig = mergeConfigs(_globalConfig, config, OptionMode::RunTime);
        auto device = _backends->getDevice(context);
        const auto executableNetwork = std::make_shared<ExecutableNetwork>(networkModel, device, localConfig);
        executableNetwork->SetPointerToPlugin(shared_from_this());
        return executableNetwork;
    } catch (const std::exception&) {
        throw;
    } catch (...) {
        IE_THROW() << "VPUX ImportNetwork got unexpected exception from ExecutableNetwork";
    }
}

//------------------------------------------------------------------------------
void Engine::SetConfig(const std::map<std::string, std::string>& config) {
    _globalConfig.update(config);
    Logger::global().setLevel(_globalConfig.get<LOG_LEVEL>());

    if (_backends != nullptr) {
        _backends->setup(_globalConfig);
    }

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

    auto localConfig = mergeConfigs(_globalConfig, config, OptionMode::CompileTime);
    Compiler::Ptr compiler = Compiler::create(localConfig);

    return compiler->query(network, localConfig);
}

IE::RemoteContext::Ptr Engine::CreateContext(const IE::ParamMap& map) {
    // Device in this case will be searched inside RemoteContext creation
    const auto device = _backends->getDevice(map);
    if (device == nullptr) {
        IE_THROW() << "CreateContext: Failed to find suitable device to use";
    }
    return std::make_shared<VPUXRemoteContext>(device, map, _globalConfig.get<LOG_LEVEL>());
}

IE::Parameter Engine::GetConfig(const std::string& name,
                                const std::map<std::string, IE::Parameter>& /*options*/) const {
    if (GetCore()->isNewAPI()) {
        if (name == ov::num_streams) {
            return _globalConfig.get<THROUGHPUT_STREAMS>();
        } else if (name == ov::enable_profiling) {
            return _globalConfig.get<PERF_COUNT>();
        } else if (name == ov::hint::performance_mode) {
            return _globalConfig.has<PERFORMANCE_HINT>() ? cvtPerformanceHint(_globalConfig.get<PERFORMANCE_HINT>())
                                                         : ov::hint::PerformanceMode::UNDEFINED;
        } else if (name == ov::log::level) {
            return cvtLogLevel(_globalConfig.get<LOG_LEVEL>());
        } else if (name == ov::device::id) {
            return _globalConfig.get<DEVICE_ID>();
        }
    }

    if (name == CONFIG_KEY(LOG_LEVEL)) {
        return static_cast<int>(_globalConfig.get<LOG_LEVEL>());
    } else if (name == CONFIG_KEY(PERF_COUNT)) {
        return _globalConfig.get<PERF_COUNT>();
    } else if (name == CONFIG_KEY(DEVICE_ID)) {
        return _globalConfig.get<DEVICE_ID>();
    } else if (name == CONFIG_KEY(PERFORMANCE_HINT)) {
        const auto val = _globalConfig.has<PERFORMANCE_HINT>()
                                 ? stringifyEnum(_globalConfig.get<PERFORMANCE_HINT>()).str()
                                 : std::string{};
        return val;
    } else if ((name == VPUX_CONFIG_KEY(THROUGHPUT_STREAMS)) || (name == KMB_CONFIG_KEY(THROUGHPUT_STREAMS))) {
        return checked_cast<int>(_globalConfig.get<THROUGHPUT_STREAMS>());
    } else if (name == VPUX_CONFIG_KEY(INFERENCE_SHAVES)) {
        return checked_cast<int>(_globalConfig.get<INFERENCE_SHAVES>());
    } else if (name == VPUX_CONFIG_KEY(COMPILATION_MODE_PARAMS)) {
        return _globalConfig.get<COMPILATION_MODE_PARAMS>();
    }

    VPUX_THROW("Unsupported configuration key {0}", name);
}

IE::Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, IE::Parameter>& options) const {
    const auto getSpecifiedDeviceName = [&options]() {
        std::string specifiedDeviceName;
        if (options.count(ov::device::id.name())) {
            specifiedDeviceName = options.at(ov::device::id.name()).as<std::string>();
        }
        if (options.count(CONFIG_KEY(DEVICE_ID)) && options.at(CONFIG_KEY(DEVICE_ID)).is<std::string>()) {
            specifiedDeviceName = options.at(CONFIG_KEY(DEVICE_ID)).as<std::string>();
        }
        return specifiedDeviceName;
    };

    if (GetCore()->isNewAPI()) {
        const auto RO_property = [](const std::string& propertyName) {
            return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
        };
        const auto RW_property = [](const std::string& propertyName) {
            return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
        };

        if (name == ov::supported_properties) {
            static const std::vector<ov::PropertyName> baseProperties{
                    RO_property(ov::supported_properties.name()),            //
                    RO_property(ov::available_devices.name()),               //
                    RO_property(ov::device::capabilities.name()),            //
                    RO_property(ov::range_for_async_infer_requests.name()),  //
                    RO_property(ov::range_for_streams.name()),               //

                    RW_property(ov::num_streams.name()),            //
                    RW_property(ov::enable_profiling.name()),        //
                    RW_property(ov::hint::performance_mode.name()),  //
                    RW_property(ov::log::level.name()),              //
                    RW_property(ov::device::id.name()),              //
            };
            static const std::vector<ov::PropertyName> supportedProperties =
            [&]() {
                std::vector<ov::PropertyName> retProperties = baseProperties;
                if (!_metrics->GetAvailableDevicesNames().empty()) {
                    retProperties.push_back(RO_property(ov::device::full_name.name()));
                    retProperties.push_back(RO_property(ov::device::architecture.name()));
                }
                return retProperties;
            }();
            return supportedProperties;
        } else if (name == ov::available_devices) {
            return _metrics->GetAvailableDevicesNames();
        } else if (name == ov::device::full_name) {
            const auto specifiedDeviceName = getSpecifiedDeviceName();
            return _metrics->GetFullDeviceName(specifiedDeviceName);
        } else if (name == ov::device::capabilities) {
            return _metrics->GetOptimizationCapabilities();
        } else if (name == ov::range_for_async_infer_requests) {
            return _metrics->GetRangeForAsyncInferRequest();
        } else if (name == ov::range_for_streams) {
            return _metrics->GetRangeForStreams();
        } else if (name == ov::device::architecture) {
            const auto specifiedDeviceName = getSpecifiedDeviceName();
            return _metrics->GetDeviceArchitecture(specifiedDeviceName);
        }
    }

    if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, _metrics->GetAvailableDevicesNames());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, _metrics->SupportedMetrics());
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        const auto specifiedDeviceName = getSpecifiedDeviceName();
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, _metrics->GetFullDeviceName(specifiedDeviceName));
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, _metrics->GetSupportedConfigKeys());
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, _metrics->GetOptimizationCapabilities());
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, _metrics->GetRangeForAsyncInferRequest());
    } else if (name == METRIC_KEY(RANGE_FOR_STREAMS)) {
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, _metrics->GetRangeForStreams());
    } else if (name == METRIC_KEY(IMPORT_EXPORT_SUPPORT)) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    } else if (name == METRIC_KEY(DEVICE_ARCHITECTURE)) {
        const auto specifiedDeviceName = getSpecifiedDeviceName();
        IE_SET_METRIC_RETURN(DEVICE_ARCHITECTURE, _metrics->GetDeviceArchitecture(specifiedDeviceName));
    } else if (name == VPUX_METRIC_KEY(BACKEND_NAME)) {
        IE_SET_METRIC_RETURN(VPUX_BACKEND_NAME, _metrics->GetBackendName());
    }

    VPUX_THROW("Unsupported metric {0}", name);
}

static const IE::Version version = {{2, 1}, CI_BUILD_NUMBER, vpux::VPUX_PLUGIN_LIB_NAME};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, version)

}  // namespace vpux
