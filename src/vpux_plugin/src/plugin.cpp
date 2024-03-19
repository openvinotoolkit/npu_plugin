//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <fstream>

#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>

#include "compiled_model.hpp"
#include "device_helpers.hpp"
#include "plugin.hpp"
#include "vpux.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux/properties.hpp"
#include "vpux_metrics.hpp"

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/file_reader.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/plugin/plugin_name.hpp"

namespace {

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

/**
 * @brief Creates an "ov::Model" object which contains only the given "parameter" and "result" nodes.
 * @details Using an "ov::Model" object to create the "CompiledModel" is the preferred way of using the OV API.
 * This path allows making use of the already written funtions/attributes for handling the I/O infromation.
 *
 * Note that a stored compiled model does not hold the original IR model within it. The only related information
 * which may be extracted is the original model's "parameter"/"result" nodes. Thus, we need to build a dummy model
 * starting from these fields in order to satisfy the API.
 * @param parameterDescriptors Describes the input nodes.
 * @param resultDescriptors Describes the output nodes.
 * @param inputNames The names of the inputs registered in the order given by the model.
 * @param outputNames The names of the outputs registered in the order given by the model.
 */
std::shared_ptr<ov::Model> create_dummy_model(const vpux::IONodeDescriptorMap& parameterDescriptors,
                                              const vpux::IONodeDescriptorMap& resultDescriptors,
                                              const std::vector<std::string>& inputNames,
                                              const std::vector<std::string>& outputNames) {
    ov::ParameterVector parameters;
    ov::NodeVector results;

    for (const std::string& inputName : inputNames) {
        const vpux::IONodeDescriptor& parameterDescriptor = parameterDescriptors.at(inputName);
        std::shared_ptr<ov::op::v0::Parameter> parameter = std::make_shared<ov::op::v0::Parameter>(
                parameterDescriptor.precision, parameterDescriptor.transposedShape);
        parameter->set_friendly_name(parameterDescriptor.currentNodeName);
        parameter->output(0).get_tensor().set_names(parameterDescriptor.outputTensorNames);
        parameters.push_back(parameter);
    }

    // The "result" nodes require a parent node in order to satisfy the legacy API naming conventions as well (in
    // the 1.0 API, the name of an output is given by the parent of the "result" node). Additionally, a dummy shape for
    // the "Constant" node was required since the specific constructor does not accept "ov::PartialShape" values (a
    // constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
    // potentially dynamic, output shape.
    for (const std::string& outputName : outputNames) {
        const vpux::IONodeDescriptor& resultDescriptor = resultDescriptors.at(outputName);
        std::shared_ptr<ov::Node> constantDummy =
                std::make_shared<ov::op::v0::Constant>(resultDescriptor.precision, CONSTANT_NODE_DUMMY_SHAPE);
        constantDummy->set_friendly_name(resultDescriptor.legacyName);

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy = std::make_shared<ov::descriptor::Tensor>(
                resultDescriptor.precision, resultDescriptor.transposedShape, resultDescriptor.outputTensorNames);

        std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v0::Result>(constantDummy);
        result->output(0).set_tensor_ptr(tensorDummy);
        result->set_friendly_name(resultDescriptor.currentNodeName);
        results.push_back(result);
    }

    return std::make_shared<ov::Model>(results, parameters);
}

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}

}  // namespace

namespace vpux {

static Config merge_configs(const Config& globalConfig, const std::map<std::string, std::string>& rawConfig,
                            OptionMode mode = OptionMode::Both) {
    Config localConfig = globalConfig;
    localConfig.update(rawConfig, mode);
    return localConfig;
}

auto get_specified_device_name(const Config config) {
    if (config.has<DEVICE_ID>()) {
        return config.get<DEVICE_ID>();
    }
    return std::string();
}

Config add_platform_to_the_config(Config config, std::string platform) {
    config.update({{ov::intel_vpux::platform.name(), platform}});
    return config;
}

Plugin::Plugin()
        : _options(std::make_shared<OptionsDesc>()), _globalConfig(_options), _logger("NPUPlugin", LogLevel::Error) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "Plugin::Plugin");
    set_device_name("NPU");

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

#endif

    OV_ITT_TASK_CHAIN(PLUGIN, itt::domains::VPUXPlugin, "Plugin::Plugin", "NPUBackends");
    _backends = std::make_shared<VPUXBackends>(backendRegistry, _globalConfig);
    OV_ITT_TASK_NEXT(PLUGIN, "registerOptions");
    _backends->registerOptions(*_options);

    OV_ITT_TASK_NEXT(PLUGIN, "Metrics");
    _metrics = std::make_unique<Metrics>(_backends);

    // parse again env_variables after backend is initialized to get backend proprieties
    _globalConfig.parseEnvVars();

    // Map from name to function {Config -> ov::Any}
    // Note that some properties are RW before network is loaded, and become RO after network is loaded
    _properties = {
            {ov::supported_properties.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  return _supportedProperties;
              }}},
            {ov::internal::caching_properties.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  return _metrics->GetCachingProperties();
              }}},
            {ov::internal::exclusive_async_requests.name(),
             {true, ov::PropertyMutability::RW,
              [&](const Config& config) {
                  return config.get<EXCLUSIVE_ASYNC_REQUESTS>();
              }}},
            {ov::streams::num.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.get<NUM_STREAMS>();
              }}},
            {ov::optimal_number_of_infer_requests.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  return static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(add_platform_to_the_config(
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
            {ov::hint::enable_cpu_pinning.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<ENABLE_CPU_PINNING>();
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
                  return config.getString<COMPILER_TYPE>();
              }}},
            {ov::intel_vpux::print_profiling.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.getString<PRINT_PROFILING>();
              }}},
            {ov::intel_vpux::profiling_output_file.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<PROFILING_OUTPUT_FILE>();
              }}},
            {ov::intel_vpux::platform.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.getString<PLATFORM>();
              }}},
            {ov::hint::model_priority.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<MODEL_PRIORITY>();
              }}},
            {ov::intel_vpux::use_elf_compiler_backend.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.getString<USE_ELF_COMPILER_BACKEND>();
              }}},
            {ov::intel_vpux::device_alloc_mem_size.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  return _metrics->GetDeviceAllocMemSize(get_specified_device_name(config));
              }}},
            {ov::intel_vpux::device_total_mem_size.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  return _metrics->GetDeviceTotalMemSize(get_specified_device_name(config));
              }}},
            {ov::intel_vpux::driver_version.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  return _metrics->GetDriverVersion(get_specified_device_name(config));
              }}},

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
                  return static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(add_platform_to_the_config(
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
            {ov::num_streams.name(),
             {true, ov::PropertyMutability::RW,
              [&](const Config& config) {
                  return config.get<NUM_STREAMS>();
              }}},
            {ov::intel_vpux::backend_name.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  return _metrics->GetBackendName();
              }}},
            {ov::device::uuid.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  const auto specifiedDeviceName = get_specified_device_name(config);
                  auto devUuid = _metrics->GetDeviceUuid(specifiedDeviceName);
                  return decltype(ov::device::uuid)::value_type{devUuid};
              }}},
            {ov::internal::supported_properties.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  return _metrics->GetInternalSupportedProperties();
              }}},

            // Add FULL_DEVICE_NAME and DEVICE_ARCHITECTURE in supported
            // properties list only in case of non-empty device list (#1424144d)
            {ov::device::architecture.name(),
             {!_metrics->GetAvailableDevicesNames().empty(), ov::PropertyMutability::RO,
              [&](const Config& config) {
                  const auto specifiedDeviceName = get_specified_device_name(config);
                  return _metrics->GetDeviceArchitecture(specifiedDeviceName);
              }}},

            {ov::device::full_name.name(),
             {!_metrics->GetAvailableDevicesNames().empty(), ov::PropertyMutability::RO,
              [&](const Config& config) {
                  const auto specifiedDeviceName = get_specified_device_name(config);
                  return _metrics->GetFullDeviceName(specifiedDeviceName);
              }}},
            {ov::intel_vpux::profiling_type.name(),
             {true, ov::PropertyMutability::RW,
              [](const Config& config) {
                  return config.get<PROFILING_TYPE>();
              }}},
    };

    for (auto& property : _properties) {
        if (std::get<0>(property.second)) {
            _supportedProperties.emplace_back(ov::PropertyName(property.first, std::get<1>(property.second)));
        }
    }
}

void Plugin::set_property(const ov::AnyMap& properties) {
    const std::map<std::string, std::string>& config = any_copy(properties);
    for (const auto& configEntry : config) {
        if (std::get<0>(_properties[configEntry.first]) == false) {
            OPENVINO_THROW("Unsupported configuration key: ", configEntry.first);
        }
    }

    _globalConfig.update(config);
    Logger::global().setLevel(_globalConfig.get<LOG_LEVEL>());
    if (_backends != nullptr) {
        _backends->setup(_globalConfig);
    }

    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    const std::map<std::string, std::string>& amends = any_copy(arguments);
    const Config amendedConfig = merge_configs(_globalConfig, amends);

    auto&& configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        return std::get<2>(configIterator->second)(amendedConfig);
    }

    OPENVINO_THROW("Unsupported configuration key: ", name);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "Plugin::compile_model");
    OV_ITT_TASK_CHAIN(PLUGIN_COMPILE_MODEL, itt::domains::VPUXPlugin, "Plugin::compile_model", "merge_configs");
    auto localConfig = merge_configs(_globalConfig, any_copy(properties));

    const auto set_cache_dir = localConfig.get<CACHE_DIR>();
    if (!set_cache_dir.empty()) {
        const auto compilerType = localConfig.get<COMPILER_TYPE>();
        if (compilerType == ov::intel_vpux::CompilerType::MLIR) {
            OPENVINO_THROW("Option 'CACHE_DIR' is not supported with MLIR compiler type");
        }
    }

    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    auto device = _backends->getDevice(localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_vpux::platform.name(), platform}});

    OV_ITT_TASK_NEXT(PLUGIN_COMPILE_MODEL, "clone_model");

    // Cloning the model is required in order to drop its "constness" and keep the original model used higher in the
    // software stack intact
    std::shared_ptr<ov::Model> cloned_model = model->clone();

    OV_ITT_TASK_NEXT(PLUGIN_COMPILE_MODEL, "compile");

    std::shared_ptr<const NetworkDescription> networkDescription;
    std::shared_ptr<ov::ICompiledModel> compiledModel;

    try {
        _compiler = Compiler::create(localConfig);
        networkDescription = _compiler->compile(cloned_model, cloned_model->get_name(), localConfig);
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        _logger.error("Unexpected exception");
        OPENVINO_THROW("NPU ExecutableNetwork got unexpected exception from compiler");
    }

    try {
        compiledModel = std::make_shared<CompiledModel>(cloned_model, shared_from_this(), networkDescription, device,
                                                        localConfig);
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        OPENVINO_THROW("Unexpected exception thrown upon attempting to create the \"CompiledModel\" object");
    }

    ++_compiledModelLoadCounter;
    OV_ITT_TASK_SKIP(PLUGIN_COMPILE_MODEL);

    return compiledModel;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& /*model*/,
                                                          const ov::AnyMap& /*properties*/,
                                                          const ov::SoPtr<ov::IRemoteContext>& /*context*/) const {
    OPENVINO_ASSERT_HELPER(::ov::NotImplemented, "", false, "Not Implemented",
                           "The remote context feature is not supported by the NPU plugin");
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& /*remote_properties*/) const {
    OPENVINO_ASSERT_HELPER(::ov::NotImplemented, "", false, "Not Implemented",
                           "The remote context feature is not supported by the NPU plugin");
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap& /*remote_properties*/) const {
    OPENVINO_ASSERT_HELPER(::ov::NotImplemented, "", false, "Not Implemented",
                           "The remote context feature is not supported by the NPU plugin");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& modelPath, const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "Plugin::import_model");

    OV_ITT_TASK_CHAIN(PLUGIN_IMPORT_MODEL, itt::domains::VPUXPlugin, "Plugin::import_model", "merge_configs");
    auto localConfig = merge_configs(_globalConfig, any_copy(properties), OptionMode::RunTime);
    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_vpux::platform.name(), platform}});
    auto device = _backends->getDevice(localConfig.get<DEVICE_ID>());

    std::shared_ptr<const NetworkDescription> networkDescription;
    std::shared_ptr<ov::ICompiledModel> compiledModel;

    OV_ITT_TASK_NEXT(PLUGIN_IMPORT_MODEL, "parse");
    try {
        _compiler = Compiler::create(localConfig);

        const std::string networkName = "net" + std::to_string(_compiledModelLoadCounter);
        networkDescription = _compiler->parse(modelPath, localConfig, networkName);

        const std::shared_ptr<ov::Model> modelDummy = create_dummy_model(
                networkDescription->getParameterDescriptors(), networkDescription->getResultDescriptors(),
                networkDescription->getInputNames(), networkDescription->getOutputNames());

        compiledModel = std::make_shared<CompiledModel>(modelDummy, shared_from_this(), networkDescription, device,
                                                        localConfig);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Can't import network: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("NPU import_model got unexpected exception from CompiledModel");
    }

    ++_compiledModelLoadCounter;
    OV_ITT_TASK_SKIP(PLUGIN_IMPORT_MODEL);

    return compiledModel;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& /*modelPath*/,
                                                         const ov::SoPtr<ov::IRemoteContext>& /*context*/,
                                                         const ov::AnyMap& /*properties*/) const {
    OPENVINO_ASSERT_HELPER(::ov::NotImplemented, "", false, "Not Implemented",
                           "The remote context feature is not supported by the NPU plugin");
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "Plugin::query_model");

    auto localConfig = merge_configs(_globalConfig, any_copy(properties), OptionMode::CompileTime);
    const auto platform = _backends->getCompilationPlatform(localConfig.get<PLATFORM>(), localConfig.get<DEVICE_ID>());
    localConfig.update({{ov::intel_vpux::platform.name(), platform}});

    _compiler = Compiler::create(localConfig);

    return _compiler->query(model, localConfig);
}

std::atomic<int> Plugin::_compiledModelLoadCounter{1};

static const ov::Version version = {CI_BUILD_NUMBER, vpux::VPUX_PLUGIN_LIB_NAME};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

}  // namespace vpux
