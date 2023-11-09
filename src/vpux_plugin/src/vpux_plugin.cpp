//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
#include "vpux/al/config/runtime.hpp"
#include "vpux/vpux_metrics.hpp"
#include "vpux_executable_network.h"
#include "vpux_metrics.h"
#include "vpux_plugin.h"

#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <device_helpers.hpp>
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/plugin/plugin_name.hpp"

namespace ie = InferenceEngine;

namespace vpux {
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static Config mergeConfigs(const Config& globalConfig, const std::map<std::string, std::string>& rawConfig,
                           OptionMode mode = OptionMode::Both) {
    if (globalConfig.get<PLATFORM>() == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
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
namespace {
auto getSpecifiedDeviceName(const Config config) {
    if (config.has<DEVICE_ID>()) {
        return config.get<DEVICE_ID>();
    }
    return std::string();
}

Config addPlatformToTheConfig(Config config, std::string platform) {
    config.update({{ov::intel_vpux::vpux_platform.name(), platform}});
    return config;
}
}  // namespace

Engine::Engine()
        : _options(std::make_shared<OptionsDesc>()), _globalConfig(_options), _logger("NPUEngine", LogLevel::Error) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "Engine::Engine");
    _pluginName = "NPU";

    registerCommonOptions(*_options);
    registerCompilerOptions(*_options);
    registerRunTimeOptions(*_options);

    // parse env_variables to get LOG_LEVEL if needed
    _globalConfig.parseEnvVars();
    Logger::global().setLevel(_globalConfig.get<LOG_LEVEL>());

    // TODO: generation of available backends list can be done during execution of CMake scripts
    std::vector<std::string> backendRegistry;

#if defined(OPENVINO_STATIC_LIBRARY)
    backendRegistry.push_back("npu_level_zero_backend");
#else

#if defined(ENABLE_IMD_BACKEND)
    if (const auto* envVar = std::getenv("IE_NPU_USE_IMD_BACKEND")) {
        if (envVarStrToBool("IE_NPU_USE_IMD_BACKEND", envVar)) {
            backendRegistry.push_back("npu_imd_backend");
        }
    }
#endif

#if defined(_WIN32) || defined(_WIN64) || (defined(__linux__) && defined(__x86_64__))
    backendRegistry.push_back("npu_level_zero_backend");
#endif
#if defined(ENABLE_EMULATOR)
    backendRegistry.push_back("npu_emulator_backend");
#endif

#endif

    OV_ITT_TASK_CHAIN(ENGINE, itt::domains::VPUXPlugin, "Engine::Engine", "NPUBackends");
    _backends = std::make_shared<VPUXBackends>(backendRegistry, _globalConfig);
    OV_ITT_TASK_NEXT(ENGINE, "registerOptions");
    _backends->registerOptions(*_options);

    OV_ITT_TASK_NEXT(ENGINE, "Metrics");
    _metrics = std::make_unique<Metrics>(_backends);

    // parse again env_variables after backend is initialized to get backend proprieties
    _globalConfig.parseEnvVars();

    // Map from name to function {Config -> ie::Parameter}
    // Note that some properties are RW before network is loaded, and become RO after network is loaded
    propertiesOv2 = {
            // from Engine::GetConfig
            {ov::supported_properties.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  return supportedProperties0v2;
              }}},
            {ov::caching_properties.name(),
             {true, ov::PropertyMutability::RW,
              [&](const Config&) {
                  return _metrics->GetCachingProperties();
              }}},
            {ov::streams::num.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.get<NUM_STREAMS>();
              }}},
            {ov::optimal_number_of_infer_requests.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  return static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(addPlatformToTheConfig(
                          config, _backends->getCompilationPlatform(config.get<PLATFORM>(), config.get<DEVICE_ID>()))));
              }}},
            {ov::enable_profiling.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<PERF_COUNT>();
              }}},
            {ov::hint::performance_mode.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<PERFORMANCE_HINT>();
              }}},
            {ov::hint::num_requests.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<PERFORMANCE_HINT_NUM_REQUESTS>();
              }}},
            {ov::log::level.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return cvtLogLevel(config.get<LOG_LEVEL>());
              }}},
            {ov::cache_dir.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  return config.get<CACHE_DIR>();
              }}},
            {ov::device::id.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<DEVICE_ID>();
              }}},
            {ov::intel_vpux::dpu_groups.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<DPU_GROUPS>();
              }}},
            {ov::intel_vpux::dma_engines.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<DMA_ENGINES>();
              }}},
            {ov::intel_vpux::compilation_mode.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<COMPILATION_MODE>();
              }}},
            {ov::intel_vpux::compilation_mode_params.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<COMPILATION_MODE_PARAMS>();
              }}},
            {ov::intel_vpux::compiler_type.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return stringifyEnum(config.get<COMPILER_TYPE>()).str();
              }}},
            {ov::intel_vpux::print_profiling.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<PRINT_PROFILING>();
              }}},
            {ov::intel_vpux::profiling_output_file.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<PROFILING_OUTPUT_FILE>();
              }}},
            {ov::intel_vpux::vpux_platform.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<PLATFORM>();
              }}},
            {ov::hint::model_priority.name(),
             {false, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<MODEL_PRIORITY>();
              }}},
            {ov::intel_vpux::use_elf_compiler_backend.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<USE_ELF_COMPILER_BACKEND>();
              }}},
            {ov::intel_vpux::device_total_mem_size.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  IE_SET_METRIC_RETURN(NPU_DEVICE_TOTAL_MEM_SIZE,
                                       _metrics->GetDeviceTotalMemSize(getSpecifiedDeviceName(config)));
              }}},
            {ov::intel_vpux::driver_version.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  IE_SET_METRIC_RETURN(NPU_DRIVER_VERSION, _metrics->GetDriverVersion(getSpecifiedDeviceName(config)));
              }}},
            // from Engine::GetConfig

            // from Engine::GetMetric
            {ov::available_devices.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  return _metrics->GetAvailableDevicesNames();
              }}},
            {ov::device::capabilities.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  return _metrics->GetOptimizationCapabilities();
              }}},
            {ov::optimal_number_of_infer_requests.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  return static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(addPlatformToTheConfig(
                          config, _backends->getCompilationPlatform(config.get<PLATFORM>(), config.get<DEVICE_ID>()))));
              }}},
            {ov::range_for_async_infer_requests.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  return _metrics->GetRangeForAsyncInferRequest();
              }}},
            {ov::range_for_streams.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  return _metrics->GetRangeForStreams();
              }}},
            {ov::device::uuid.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  const auto specifiedDeviceName = getSpecifiedDeviceName(config);
                  auto devUuid = _metrics->GetDeviceUuid(specifiedDeviceName);
                  return decltype(ov::device::uuid)::value_type{devUuid};
              }}},
            // from Engine::GetMetric

            // Add FULL_DEVICE_NAME and DEVICE_ARCHITECTURE in supported
            // properties list only in case of non-empty device list (#1424144d)
            {ov::device::architecture.name(),
             {!_metrics->GetAvailableDevicesNames().empty(), ov::PropertyMutability::RO,
              [&](const Config& config) {
                  const auto specifiedDeviceName = getSpecifiedDeviceName(config);
                  return _metrics->GetDeviceArchitecture(specifiedDeviceName);
              }}},

            {ov::device::full_name.name(),
             {!_metrics->GetAvailableDevicesNames().empty(), ov::PropertyMutability::RO,
              [&](const Config& config) {
                  const auto specifiedDeviceName = getSpecifiedDeviceName(config);
                  return _metrics->GetFullDeviceName(specifiedDeviceName);
              }}},
    };

    for (auto& prop : propertiesOv2) {
        if (std::get<0>(prop.second)) {
            supportedProperties0v2.emplace_back(ov::PropertyName(prop.first, std::get<1>(prop.second)));
        }
    }

    propertiesOv1 = {
            // from Engine::GetConfig
            {CONFIG_KEY(LOG_LEVEL),
             {true,
              [&](const Config& config) {
                  return static_cast<int>(config.get<LOG_LEVEL>());
              }}},
            {CONFIG_KEY(PERF_COUNT),
             {true,
              [&](const Config& config) {
                  return config.get<PERF_COUNT>();
              }}},
            {CONFIG_KEY(DEVICE_ID),
             {true,
              [&](const Config& config) {
                  return config.get<DEVICE_ID>();
              }}},
            {CONFIG_KEY(PERFORMANCE_HINT),
             {true,
              [&](const Config& config) {
                  return stringifyEnum(config.get<PERFORMANCE_HINT>()).str();
              }}},
            {CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS),
             {true,
              [&](const Config& config) {
                  return config.get<PERFORMANCE_HINT_NUM_REQUESTS>();
              }}},
            {VPUX_CONFIG_KEY(DMA_ENGINES),
             {true,
              [&](const Config& config) {
                  return config.get<DMA_ENGINES>();
              }}},
            {VPUX_CONFIG_KEY(DPU_GROUPS),
             {true,
              [&](const Config& config) {
                  return config.get<DPU_GROUPS>();
              }}},
            {VPUX_CONFIG_KEY(COMPILATION_MODE),
             {true,
              [&](const Config& config) {
                  return config.get<COMPILATION_MODE>();
              }}},
            {VPUX_CONFIG_KEY(COMPILATION_MODE_PARAMS),
             {true,
              [&](const Config& config) {
                  return config.get<COMPILATION_MODE_PARAMS>();
              }}},
            {VPUX_CONFIG_KEY(USE_ELF_COMPILER_BACKEND),
             {true,
              [&](const Config& config) {
                  return config.get<USE_ELF_COMPILER_BACKEND>();
              }}},
            {VPUX_CONFIG_KEY(COMPILER_TYPE),
             {true,
              [&](const Config& config) {
                  return stringifyEnum(config.get<COMPILER_TYPE>()).str();
              }}},
            {CONFIG_KEY(CACHE_DIR),
             {true,
              [&](const Config& config) {
                  return config.get<CACHE_DIR>();
              }}},
            {ov::caching_properties.name(),
             {true,
              [&](const Config&) {
                  return _metrics->GetCachingProperties();
              }}},
            {ov::num_streams.name(),
             {true,
              [&](const Config& config) {
                  return config.get<NUM_STREAMS>();
              }}},
            {ov::supported_properties.name(),
             {true,
              [&](const Config&) {
                  return supportedProperties0v1;
              }}},
            // from Engine::GetConfig

            // from Engine::GetMetric
            {VPUX_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE),
             {true,
              [&](const Config& config) {
                  IE_SET_METRIC_RETURN(NPU_DEVICE_TOTAL_MEM_SIZE,
                                       _metrics->GetDeviceTotalMemSize(getSpecifiedDeviceName(config)));
              }}},
            {VPUX_METRIC_KEY(DRIVER_VERSION),
             {true,
              [&](const Config& config) {
                  IE_SET_METRIC_RETURN(NPU_DRIVER_VERSION, _metrics->GetDriverVersion(getSpecifiedDeviceName(config)));
              }}},
            {METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
             {true,
              [&](const Config& config) {
                  return static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(addPlatformToTheConfig(
                          config, _backends->getCompilationPlatform(config.get<PLATFORM>(), config.get<DEVICE_ID>()))));
              }}},
            {METRIC_KEY(AVAILABLE_DEVICES),
             {true,
              [&](const Config&) {
                  IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, _metrics->GetAvailableDevicesNames());
              }}},
            {METRIC_KEY(SUPPORTED_METRICS),
             {true,
              [&](const Config&) {
                  IE_SET_METRIC_RETURN(SUPPORTED_METRICS, _metrics->SupportedMetrics());
              }}},
            {METRIC_KEY(FULL_DEVICE_NAME),
             {true,
              [&](const Config& config) {
                  const auto specifiedDeviceName = getSpecifiedDeviceName(config);
                  IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, _metrics->GetFullDeviceName(specifiedDeviceName));
              }}},
            {METRIC_KEY(SUPPORTED_CONFIG_KEYS),
             {true,
              [&](const Config&) {
                  IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, _metrics->GetSupportedConfigKeys());
              }}},
            {METRIC_KEY(OPTIMIZATION_CAPABILITIES),
             {true,
              [&](const Config&) {
                  IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, _metrics->GetOptimizationCapabilities());
              }}},
            {METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS),
             {true,
              [&](const Config&) {
                  IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, _metrics->GetRangeForAsyncInferRequest());
              }}},
            {METRIC_KEY(RANGE_FOR_STREAMS),
             {true,
              [&](const Config&) {
                  IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, _metrics->GetRangeForStreams());
              }}},
            {METRIC_KEY(IMPORT_EXPORT_SUPPORT),
             {true,
              [&](const Config&) {
                  IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
              }}},
            {METRIC_KEY(DEVICE_ARCHITECTURE),
             {true,
              [&](const Config& config) {
                  const auto specifiedDeviceName = getSpecifiedDeviceName(config);
                  IE_SET_METRIC_RETURN(DEVICE_ARCHITECTURE, _metrics->GetDeviceArchitecture(specifiedDeviceName));
              }}},
            {VPUX_METRIC_KEY(BACKEND_NAME),
             {true,
              [&](const Config&) {
                  IE_SET_METRIC_RETURN(NPU_BACKEND_NAME, _metrics->GetBackendName());
              }}},
            // from Engine::GetMetric
    };

    for (auto& prop : propertiesOv1) {
        if (std::get<0>(prop.second)) {
            supportedProperties0v1.emplace_back(ov::PropertyName(prop.first));
        }
    }
}

//------------------------------------------------------------------------------
//      Load network
//------------------------------------------------------------------------------
ie::IExecutableNetworkInternal::Ptr Engine::LoadExeNetwork(const ie::CNNNetwork& network,
                                                           std::shared_ptr<Device>& device,
                                                           const Config& networkConfig) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "Engine::LoadExeNetwork");
    try {
        return std::make_shared<ExecutableNetwork>(network, device, networkConfig, GetCore()->isNewAPI());
    } catch (const std::exception& ex) {
        IE_THROW(Unexpected) << ex.what();
    } catch (...) {
        IE_THROW(Unexpected) << "NPU LoadExeNetwork got unexpected exception from ExecutableNetwork";
    }
}

ie::IExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const ie::CNNNetwork& network,
                                                               const std::map<std::string, std::string>& config) {
    auto localConfig = mergeConfigs(_globalConfig, config);

    const auto set_cache_dir = localConfig.get<CACHE_DIR>();
    if (!set_cache_dir.empty()) {
        const auto compilerType = localConfig.get<COMPILER_TYPE>();
        if (compilerType == cvtCompilerType(ov::intel_vpux::CompilerType::MLIR))
            IE_THROW() << "Option 'CACHE_DIR' is not supported with MLIR compiler type";
    }

    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    auto device = _backends->getDevice(localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_vpux::vpux_platform.name(), platform}});
    return LoadExeNetwork(network, device, localConfig);
}

ie::IExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const ie::CNNNetwork&, const ie::RemoteContext::Ptr&,
                                                               const std::map<std::string, std::string>&) {
    IE_THROW() << "LoadExeNetworkImpl failed due to RemoteContext deprecation";
}

//------------------------------------------------------------------------------
//      Import network
//------------------------------------------------------------------------------
ie::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(const std::string& modelFileName,
                                                          const std::map<std::string, std::string>& config) {
    OV_ITT_TASK_CHAIN(IMPORT_NETWORK, itt::domains::VPUXPlugin, "Engine::ImportNetwork", "FileToStream");
    std::ifstream blobStream(modelFileName, std::ios::binary);
    OV_ITT_TASK_SKIP(IMPORT_NETWORK);

    auto localConfig = mergeConfigs(_globalConfig, config);
    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_vpux::vpux_platform.name(), platform}});

    return ImportNetwork(vpu::KmbPlugin::utils::skipMagic(blobStream), config);
}

ie::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(std::istream& networkModel,
                                                          const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "Engine::ImportNetwork");
    try {
        auto localConfig = mergeConfigs(_globalConfig, config, OptionMode::RunTime);
        const auto platform =
                _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
        localConfig.update({{ov::intel_vpux::vpux_platform.name(), platform}});
        auto device = _backends->getDevice(localConfig.get<DEVICE_ID>());
        const auto executableNetwork = std::make_shared<ExecutableNetwork>(networkModel, device, localConfig);
        executableNetwork->SetPointerToPlugin(shared_from_this());
        return executableNetwork;
    } catch (const std::exception& ex) {
        IE_THROW(Unexpected) << "Can't import network: " << ex.what();
    } catch (...) {
        IE_THROW(Unexpected) << "NPU ImportNetwork got unexpected exception from ExecutableNetwork";
    }
}

ie::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(std::istream&, const ie::RemoteContext::Ptr&,
                                                          const std::map<std::string, std::string>&) {
    IE_THROW() << "ImportNetwork failed due to RemoteContext deprecation";
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

ie::QueryNetworkResult Engine::QueryNetwork(const ie::CNNNetwork& network,
                                            const std::map<std::string, std::string>& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "Engine::QueryNetwork");
    ie::QueryNetworkResult queryNetworkResult;

    if (nullptr == network.getFunction()) {
        IE_THROW() << "NPU Plugin supports only ngraph cnn network representation";
    }

    auto localConfig = mergeConfigs(_globalConfig, config, OptionMode::CompileTime);
    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_vpux::vpux_platform.name(), platform}});

    Compiler::Ptr compiler = Compiler::create(localConfig);

    const std::shared_ptr<const ov::Model>& model = network.getFunction();
    queryNetworkResult.supportedLayersMap = compiler->query(model, localConfig);
    return queryNetworkResult;
}

ie::RemoteContext::Ptr Engine::CreateContext(const ie::ParamMap&) {
    IE_THROW() << "CreateContext failed due to RemoteContext deprecation";
}

ie::Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, ie::Parameter>& options) const {
    std::map<std::string, std::string> amends;
    for (auto option : options) {
        amends.insert({option.first, option.second.as<std::string>()});
    }
    const Config amendedConfig = mergeConfigs(_globalConfig, amends);

    if (GetCore()->isNewAPI()) {
        auto&& cit = propertiesOv2.find(name);
        if (cit != propertiesOv2.cend()) {
            return std::get<2>(cit->second)(amendedConfig);
        }
    }

    auto&& cit = propertiesOv1.find(name);
    if (cit != propertiesOv1.cend()) {
        return std::get<1>(cit->second)(amendedConfig);
    }

    VPUX_THROW("Unsupported configuration key {0}", name);
}

ie::Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, ie::Parameter>& options) const {
    std::map<std::string, std::string> amends;
    for (auto option : options) {
        amends.insert({option.first, option.second.as<std::string>()});
    }
    const Config amendedConfig = mergeConfigs(_globalConfig, amends);

    if (GetCore()->isNewAPI()) {
        auto&& cit = propertiesOv2.find(name);
        if (cit != propertiesOv2.cend()) {
            return std::get<2>(cit->second)(amendedConfig);
        }
    }

    auto&& cit = propertiesOv1.find(name);
    if (cit != propertiesOv1.cend()) {
        return std::get<1>(cit->second)(amendedConfig);
    }

    VPUX_THROW("Unsupported metric {0}", name);
}

static const ie::Version version = {{2, 1}, CI_BUILD_NUMBER, vpux::VPUX_PLUGIN_LIB_NAME};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, version)

}  // namespace vpux
