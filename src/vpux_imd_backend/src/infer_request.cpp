//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/infer_request.hpp"

#include <debug.h>
#include <openvino/runtime/make_tensor.hpp>
#include <openvino/util/file_util.hpp>

#include "vpux/IMD/executor.hpp"
#include "vpux/IMD/parsed_properties.hpp"
#include "vpux/IMD/platform_helpers.hpp"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"

#include <device_helpers.hpp>
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Program.h>

namespace {

constexpr void* NO_PREALLOCATED_BUFFER = nullptr;
constexpr bool STATE_TENSOR = true;
constexpr bool NOT_STATE_TENSOR = false;

}  // namespace

namespace vpux {

IMDInferRequest::IMDInferRequest(const std::shared_ptr<const ov::ICompiledModel> compiledModel,
                                 const std::shared_ptr<const NetworkDescription> networkDescription,
                                 const Executor::Ptr executor, const Config& config)
        : SyncInferRequest(compiledModel, networkDescription),
          _executorPtr(executor),
          _config(config),
          _logger("IMDInferRequest", config.get<LOG_LEVEL>()),
          _inputOrder(networkDescription->getInputOrder()),
          _outputOrder(networkDescription->getOutputOrder()) {
    for (const std::string& inputName : _inputNames) {
        const IONodeDescriptor& parameterDescriptor = _parameterDescriptors.at(inputName);

        // No I/O buffers have been allocated so far by the plugin - allocate new ones here
        allocate_tensor(inputName, parameterDescriptor, NO_PREALLOCATED_BUFFER, NOT_STATE_TENSOR);
    }

    for (const std::string& outputName : _outputNames) {
        const IONodeDescriptor& resultDescriptor = _resultDescriptors.at(outputName);
        allocate_tensor(outputName, resultDescriptor, NO_PREALLOCATED_BUFFER, NOT_STATE_TENSOR);
    }

    for (const std::string& stateName : _stateNames) {
        const IONodeDescriptor& stateDescriptor = _stateDescriptors.at(stateName);
        allocate_tensor(stateName, stateDescriptor, NO_PREALLOCATED_BUFFER, STATE_TENSOR);
    }
}

void IMDInferRequest::infer() {
    infer_async();
    get_result();
}

void IMDInferRequest::infer_async() {
    _logger.debug("InferRequest::infer_async started");
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "infer_async");

    _logger.info("Run inference using InferenceManagerDemo application");

    _workDirectory = create_temporary_work_directory();

    store_compiled_model();
    store_network_inputs();
    run_app();
    _logger.debug("InferRequest::infer_async finished");
}

void IMDInferRequest::get_result() {
    _logger.debug("InferRequest::get_result started");
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "get_result");

    load_network_outputs();

    _logger.trace("Remove the temporary working directory '{0}'", _workDirectory);
    const auto errc = llvm::sys::fs::remove_directories(_workDirectory);

    if (errc) {
        _logger.error("Failed to remove temporary working directory : {0}", errc.message());
    }

    _logger.debug("InferRequest::get_result finished");
}

SmallString IMDInferRequest::create_temporary_work_directory() {
    _logger.trace("Create unique temporary working directory");

    SmallString _workDirectory;
    const auto errc = llvm::sys::fs::createUniqueDirectory("vpux-IMD", _workDirectory);
    VPUX_THROW_WHEN(errc, "Failed to create temporary working directory : {0}", errc.message());

    _logger.nest().trace("{0}", _workDirectory);

    return _workDirectory;
}

void IMDInferRequest::store_compiled_model() {
    _logger.trace("Store the compile model");

    const std::vector<char>& compiledModel =
            static_cast<IMDExecutor*>(_executorPtr.get())->getNetworkDesc().getCompiledNetwork();

    const std::string fileName = "test.blob";
    const auto modelFilePath = printToString("{0}/{1}", _workDirectory.str(), fileName);
    std::ofstream file(modelFilePath, std::ios::binary);

    VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for write", modelFilePath);

    file.write(compiledModel.data(), compiledModel.size());

    _logger.nest().trace("{0}", modelFilePath);
}

void IMDInferRequest::store_network_inputs() {
    _logger.trace("Store the network inputs");

    size_t inputIndex;

    for (const auto& name : _inputAndStateInputNames) {
        const std::shared_ptr<ov::ITensor>& inputTensor = _allTensors.at(name);

        if (!isStateOutputName(name)) {
            inputIndex = _inputOrder.at(name);
        } else {
            inputIndex = _inputOrder.at(stateOutputToStateInputName(name));
        }

        const auto inputFilePath = printToString("{0}/input-{1}.bin", _workDirectory.str(), inputIndex);
        std::ofstream file(inputFilePath, std::ios_base::binary | std::ios_base::out);

        VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for write", inputFilePath);

        file.write(reinterpret_cast<const char*>(inputTensor->data()), inputTensor->get_byte_size());

        _logger.nest().trace("{0} - {1}", name, inputFilePath);
    }
}

void IMDInferRequest::run_app() {
    _logger.trace("Run the application");

    SmallString curPath;
    auto errc = llvm::sys::fs::current_path(curPath);
    VPUX_THROW_WHEN(errc, "Failed to get current path : {0}", errc.message());

    VPUX_SCOPE_EXIT {
        _logger.nest().trace("Restore current working directory '{0}'", curPath);
        errc = llvm::sys::fs::set_current_path(curPath);

        if (errc) {
            _logger.error("Failed to restore current path : {0}", errc.message());
        }
    };

    _logger.nest().trace("Change current working directory to the new temporary folder '{0}'", _workDirectory.str());
    errc = llvm::sys::fs::set_current_path(_workDirectory.str());
    VPUX_THROW_WHEN(errc, "Failed to change current path : {0}", errc.message());

    const std::string emptyString = "";
    SmallVector<std::optional<StringRef>> redirects = {
            std::nullopt,  // stdin(0)
            std::nullopt,  // stdout(1)
            std::nullopt   // stderr(2)
    };

    if (_logger.level() < LogLevel::Error) {
        // diconnect stderr file descriptor
        redirects[2] = StringRef(emptyString);
    }

    if (_logger.level() < LogLevel::Info) {
        // diconnect stdout file descriptor
        redirects[1] = StringRef(emptyString);
    }

    std::string errMsg;
    auto app = static_cast<IMDExecutor*>(_executorPtr.get())->getApp();
    SmallVector<StringRef> args(app.runArgs.begin(), app.runArgs.end());
    _logger.trace("exec: {0}", app.runProgram);
    _logger.trace("args: {0}", args);

    const auto procErr = llvm::sys::ExecuteAndWait(app.runProgram, args,
                                                   /*Env=*/std::nullopt, llvm::ArrayRef(redirects),
                                                   checked_cast<uint32_t>(app.timeoutSec),
                                                   /*MemoryLimit=*/0, &errMsg);
    VPUX_THROW_WHEN(procErr != 0, "Failed to run InferenceManagerDemo ({0}) : {1}", procErr, errMsg);
}

void IMDInferRequest::read_from_file(const std::string& path, const std::shared_ptr<ov::ITensor>& tensor) {
    VPUX_THROW_UNLESS(tensor->data() != nullptr, "Tensor was not allocated");

    std::ifstream file(path, std::ios_base::binary | std::ios_base::ate);
    VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for reading", path);

    const std::size_t tensorByteSize = tensor->get_byte_size();
    const auto fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios_base::beg);
    VPUX_THROW_UNLESS(fileSize == tensorByteSize, "File '{0}' contains {1} bytes, but {2} expected", path, fileSize,
                      tensorByteSize);

    file.read(reinterpret_cast<char*>(tensor->data()), static_cast<std::streamsize>(tensorByteSize));
}

void IMDInferRequest::load_network_outputs() {
    _logger.trace("Load the network outputs");

    for (const auto& name : _outputAndStateOutputNames) {
        const std::shared_ptr<ov::ITensor>& outputTensor = _allTensors.at(name);

        const auto outputFilePath = printToString("{0}/output-{1}.bin", _workDirectory.str(), _outputOrder.at(name));
        read_from_file(outputFilePath, outputTensor);

        _logger.nest().trace("{0} - {1}", name, outputFilePath);
    }

    const NetworkDescription& networkDescription = static_cast<IMDExecutor*>(_executorPtr.get())->getNetworkDesc();
    const IONodeDescriptorMap& profilingOutputDescriptors = networkDescription.getProfilingOutputDescriptors();

    if (profilingOutputDescriptors.size()) {
        _logger.info("Load profiling output");
        OPENVINO_ASSERT(profilingOutputDescriptors.size() == 1);

        const IONodeDescriptor& profilingOutputDescriptor = profilingOutputDescriptors.begin()->second;
        const std::shared_ptr<ov::ITensor>& profilingOutputTensor = ov::make_tensor(
                profilingOutputDescriptor.precision, profilingOutputDescriptor.transposedShape.get_shape());
        read_from_file(printToString("{0}/profiling-0.bin", _workDirectory.str()), profilingOutputTensor);
        _rawProfilingData = profilingOutputTensor;
    }
}

void IMDInferRequest::check_network_precision(const ov::element::Type_t precision) {
    switch (precision) {
    case ov::element::Type_t::f32:
        break;
    case ov::element::Type_t::f16:
        break;
    case ov::element::Type_t::u8:
        break;
    case ov::element::Type_t::i8:
        break;
    case ov::element::Type_t::u16:
        break;
    case ov::element::Type_t::i16:
        break;
    case ov::element::Type_t::u32:
        break;
    case ov::element::Type_t::i32:
        break;
    case ov::element::Type_t::u64:
        break;
    case ov::element::Type_t::i64:
        break;
    case ov::element::Type_t::boolean:
        break;
    default:
        OPENVINO_THROW("Unsupported tensor precision: " + ov::element::Type(precision).get_type_name() +
                       "! Supported precisions: FP32, FP16, U8, I8, U16, I16, U32, I32, U64, I64, BOOLEAN");
    }
}

std::vector<ov::ProfilingInfo> IMDInferRequest::get_profiling_info() const {
    OPENVINO_ASSERT(_rawProfilingData->data() != nullptr);

    auto executorPtr = static_cast<IMDExecutor*>(_executorPtr.get());
    const auto& compiledModel = executorPtr->getNetworkDesc().getCompiledNetwork();
    return profiling::getLayerStatistics(reinterpret_cast<const uint8_t*>(_rawProfilingData->data()),
                                         _rawProfilingData->get_byte_size(), compiledModel);
}

}  // namespace vpux
