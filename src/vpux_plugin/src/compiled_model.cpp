//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <fstream>
#include <string_view>

#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/system_conf.hpp>
#include <openvino/runtime/threading/executor_manager.hpp>
#include <transformations/utils/utils.hpp>

#include "async_infer_request.hpp"
#include "compiled_model.hpp"
#include "vpux/utils/IE/config.hpp"
#include "vpux/utils/IE/itt.hpp"

#include "vpux.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux_compiler.hpp"

namespace {

constexpr std::string_view NO_EXECUTOR_FOR_INFERENCE =
        "Can't create infer request!\n"
        "Please make sure that the device is available. Only exports can be made.";

std::uint32_t hash(const std::vector<char>& data) {
    std::uint32_t result = 1171117u;
    for (const char& c : data)
        result = ((result << 7) + result) + static_cast<uint32_t>(c);
    return result;
}

}  // namespace

namespace vpux {

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model> model,
                             const std::shared_ptr<const ov::IPlugin> plugin,
                             const std::shared_ptr<const NetworkDescription> networkDescription,
                             const std::shared_ptr<Device> device, const Config& config)
        : ov::ICompiledModel(model, plugin),
          _networkPtr(networkDescription),
          _model(model),
          _config(config),
          _logger("CompiledModel", config.get<LOG_LEVEL>()),
          _device(device) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompiledModel::CompiledModel");

    if (_networkPtr == nullptr) {
        OPENVINO_THROW("Network is null!");
    }

    OV_ITT_TASK_CHAIN(COMPILED_MODEL, itt::domains::VPUXPlugin, "CompiledModel::CompiledModel",
                      "initialize_properties");
    initialize_properties();
    configure_stream_executors();

    OV_ITT_TASK_NEXT(COMPILED_MODEL, "create_executor");
    const bool configCreateExecutor = _config.get<CREATE_EXECUTOR>();
    static const auto envVar = std::getenv("IE_NPU_CREATE_EXECUTOR");
    const bool IE_NPU_CREATE_EXECUTOR =
            envVar ? vpux::envVarStrToBool("IE_NPU_CREATE_EXECUTOR", envVar) : configCreateExecutor;

    if (IE_NPU_CREATE_EXECUTOR) {
        _logger.info("Creating the executor inside the \"CompiledModel\" constructor");

        // If no device has been defined, the executor shall keep the default value of "nullptr". In this scenario, only
        // export operations will be allowed
        if (_device != nullptr) {
            _executorPtr = _device->createExecutor(_networkPtr, _config);
        }
    } else {
        _logger.info("Executor will not be created inside the \"CompiledModel\" constructor");
    }

    OV_ITT_TASK_SKIP(COMPILED_MODEL);
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompiledModel::create_infer_request");

    if (_executorPtr == nullptr && _device != nullptr) {
        _executorPtr = _device->createExecutor(_networkPtr, _config);
    }
    if (_executorPtr == nullptr) {
        OPENVINO_THROW(NO_EXECUTOR_FOR_INFERENCE);
    }

    const std::shared_ptr<SyncInferRequest>& syncInferRequest =
            _device->createInferRequest(shared_from_this(), _networkPtr, _executorPtr, _config);
    syncInferRequest->initialize_states();

    return std::make_shared<AsyncInferRequest>(syncInferRequest, get_task_executor(), _resultExecutor,
                                               get_callback_executor());
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    OPENVINO_ASSERT_HELPER(::ov::NotImplemented, "", false, "Not Implemented",
                           "The synchronous inference request structure implemented by the NPU plugin does not inherit "
                           "the \"ov::ISyncInferRequest\" class");
}

void CompiledModel::export_model(std::ostream& model) const {
    auto compiledModelBuffer = _networkPtr->getCompiledNetwork();
    model.write(compiledModelBuffer.data(), compiledModelBuffer.size());

    std::stringstream str;
    str << "Blob size: " << compiledModelBuffer.size() << ", hash: " << std::hex << hash(compiledModelBuffer);
    _logger.info("{0}", str.str());
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    return _model;
}

void CompiledModel::set_property(const ov::AnyMap& /*properties*/) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    auto&& configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        return std::get<2>(configIterator->second)(_config);
    }

    OPENVINO_THROW("Unsupported property ", name);
}

void CompiledModel::configure_stream_executors() {
    std::shared_ptr<ov::threading::ITaskExecutor> task_executor;
    if (get_plugin()->get_property(ov::internal::exclusive_async_requests.name(), {}).as<bool>()) {
        task_executor = ov::threading::executor_manager()->get_executor("NPU");
    } else if (get_property(ov::hint::enable_cpu_pinning.name()).as<bool>()) {
        auto executor_config = ov::threading::IStreamsExecutor::Config{
                "Intel NPU plugin executor",
                0,
                0,
                ov::threading::IStreamsExecutor::ThreadBindingType::CORES,
                1,
                0,
                0,
                ov::threading::IStreamsExecutor::Config::PreferredCoreType::BIG,
                {{get_plugin()->get_property(ov::num_streams.name(), {}).as<ov::streams::Num>(), ov::MAIN_CORE_PROC, 1,
                  0, 0}},
                true};
        auto post_config = ov::threading::IStreamsExecutor::Config::reserve_cpu_threads(executor_config);
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(post_config);
    } else {
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(
                ov::threading::IStreamsExecutor::Config{"NPUPlugin executor"});
    }

    set_task_executor(task_executor);
    _resultExecutor = ov::threading::executor_manager()->get_executor(_networkPtr->getName() + "_NPUResultExecutor");
}

void CompiledModel::initialize_properties() {
    _properties = {
            {ov::supported_properties.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  return _supportedProperties;
              }}},
            {ov::device::id.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.get<DEVICE_ID>();
              }}},
            {ov::intel_vpux::print_profiling.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.getString<PRINT_PROFILING>();
              }}},
            {ov::intel_vpux::profiling_type.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.getString<PROFILING_TYPE>();
              }}},
            {ov::intel_vpux::profiling_output_file.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.get<PROFILING_OUTPUT_FILE>();
              }}},
            {ov::intel_vpux::platform.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.getString<PLATFORM>();
              }}},
            {ov::hint::model_priority.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.get<MODEL_PRIORITY>();
              }}},
            {ov::enable_profiling.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.get<PERF_COUNT>();
              }}},
            {ov::hint::performance_mode.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.get<PERFORMANCE_HINT>();
              }}},
            {ov::hint::num_requests.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.get<PERFORMANCE_HINT_NUM_REQUESTS>();
              }}},
            {ov::hint::enable_cpu_pinning.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.get<ENABLE_CPU_PINNING>();
              }}},
            {ov::intel_vpux::use_elf_compiler_backend.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.getString<USE_ELF_COMPILER_BACKEND>();
              }}},
            {ov::intel_vpux::create_executor.name(),
             {true, ov::PropertyMutability::RO,
              [](const Config& config) {
                  return config.get<CREATE_EXECUTOR>();
              }}},

            {ov::model_name.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  OPENVINO_ASSERT(_networkPtr != nullptr, "Missing network descriptor");
                  return _networkPtr->getName();
              }}},
            {ov::optimal_number_of_infer_requests.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config& config) {
                  // value is allowed to be queried prior the network is compiled
                  return static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(config));
              }}},
            {ov::execution_devices.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  VPUX_THROW_WHEN(_device == nullptr, "GetMetric: device is not initialized");
                  return _device->getName();
              }}},
            {ov::internal::caching_properties.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  static const std::vector<ov::PropertyName> supportedProperty{
                          ov::PropertyName(ov::log::level.name(), ov::PropertyMutability::RO),
                  };
                  return supportedProperty;
              }}},
            {ov::internal::supported_properties.name(),
             {true, ov::PropertyMutability::RO,
              [&](const Config&) {
                  static const std::vector<ov::PropertyName> supportedProperty{
                          ov::PropertyName(ov::internal::caching_properties.name(), ov::PropertyMutability::RO),
                  };
                  return supportedProperty;
              }}},
    };

    for (auto& property : _properties) {
        if (std::get<0>(property.second)) {
            _supportedProperties.emplace_back(ov::PropertyName(property.first, std::get<1>(property.second)));
        }
    }
}

}  // namespace vpux
