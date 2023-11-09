//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/infer_request.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <debug.h>
#include <ie_blob.h>
#include <blob_factory.hpp>
#include <openvino/util/file_util.hpp>

#include <vpux_variable_state.hpp>
#include "vpux/IMD/executor.hpp"
#include "vpux/IMD/parsed_config.hpp"
#include "vpux/IMD/platform_helpers.hpp"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"

#include <device_helpers.hpp>
#include "vpux/utils/IE/data_attributes_check.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Program.h>

namespace ie = InferenceEngine;
using namespace InferenceEngine;

namespace vpux {

namespace {
void checkNetworkPrecision(const ie::Precision& precision) {
    if (precision != ie::Precision::FP32 && precision != ie::Precision::FP16 && precision != ie::Precision::U8 &&
        precision != ie::Precision::I8 && precision != ie::Precision::I32 && precision != ie::Precision::U32) {
        IE_THROW(ParameterMismatch) << "Unsupported input precision: " << precision
                                    << "! Supported precisions: FP32, FP16, U8, I8, I32, U32";
    }
}

ie::Blob::Ptr allocateLocalBlob(const ie::TensorDesc& tensorDesc,
                                const std::shared_ptr<InferenceEngine::IAllocator>& allocator) {
    checkNetworkPrecision(tensorDesc.getPrecision());

    ie::Blob::Ptr blob;
    if (allocator == nullptr) {
        blob = make_blob_with_precision(tensorDesc);
    } else {
        blob = make_blob_with_precision(tensorDesc, allocator);
    }
    if (blob == nullptr) {
        IE_THROW() << "InputBlob is nullptr.";
    }
    blob->allocate();
    return blob;
}
}  // namespace

SmallString IMD::IMDInferRequest::createTempWorkDir() {
    _logger.trace("Create unique temporary working directory...");

    SmallString workDir;
    const auto errc = llvm::sys::fs::createUniqueDirectory("vpux-IMD", workDir);
    VPUX_THROW_WHEN(errc, "Failed to create temporary working directory : {0}", errc.message());

    _logger.nest().trace("{0}", workDir);

    return workDir;
}

void IMD::IMDInferRequest::storeNetworkBlob(StringRef workDir) {
    _logger.trace("Store the network blob...");

    const auto& compiledBlob = static_cast<ExecutorImpl*>(_executorPtr.get())->getNetworkDesc().getCompiledNetwork();

    const std::string fileName = "test.blob";
    const auto modelFilePath = printToString("{0}/{1}", workDir, fileName);
    std::ofstream file(modelFilePath, std::ios::binary);
    VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for write", modelFilePath);
    file.write(compiledBlob.data(), compiledBlob.size());

    _logger.nest().trace("{0}", modelFilePath);
}

void IMD::IMDInferRequest::storeNetworkInputs(StringRef workDir, const BlobMap& inputs) {
    _logger.trace("Store the network inputs...");

    const auto& networkDescriptor = static_cast<ExecutorImpl*>(_executorPtr.get())->getNetworkDesc();
    const auto& deviceInputsInfo = networkDescriptor.getDeviceInputsInfo();

    for (const auto& p : deviceInputsInfo | indexed) {
        const auto& blobName = p.value().first;
        const auto& inputData = inputs.at(blobName);
        const auto ind = p.index();

        const TensorDesc& inputDataAttributes = inputData->getTensorDesc();
        const TensorDesc& deviceDataAttributes = p.value().second->getTensorDesc();
        checkDataAttributesMatch(inputDataAttributes, deviceDataAttributes);

        const auto& dataMemoryBlob = as<MemoryBlob>(inputData);
        VPUX_THROW_UNLESS(dataMemoryBlob != nullptr, "Got non MemoryBlob");

        const auto mem = dataMemoryBlob->rmap();
        const auto ptr = mem.as<const char*>();
        VPUX_THROW_UNLESS(ptr != nullptr, "Blob was not allocated");

        const auto inputFilePath = printToString("{0}/input-{1}.bin", workDir, ind);
        std::ofstream file(inputFilePath, std::ios_base::binary | std::ios_base::out);
        VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for write", inputFilePath);
        file.write(ptr, dataMemoryBlob->byteSize());

        _logger.nest().trace("{0} - {1}", blobName, inputFilePath);
    }
}

void IMD::IMDInferRequest::runApp(StringRef workDir) {
    _logger.trace("Run the application...");

    SmallString curPath;
    auto errc = llvm::sys::fs::current_path(curPath);
    VPUX_THROW_WHEN(errc, "Failed to get current path : {0}", errc.message());

    VPUX_SCOPE_EXIT {
        _logger.nest().trace("Restore current working directory '{0}'...", curPath);
        errc = llvm::sys::fs::set_current_path(curPath);

        if (errc) {
            _logger.error("Failed to restore current path : {0}", errc.message());
        }
    };

    _logger.nest().trace("Change current working directory to the new temporary folder '{0}'...", workDir);
    errc = llvm::sys::fs::set_current_path(workDir);
    VPUX_THROW_WHEN(errc, "Failed to change current path : {0}", errc.message());

    _logger.nest().trace("{0}", static_cast<ExecutorImpl*>(_executorPtr.get())->getApp().runArgs);

    const std::string emptyString = "";
    SmallVector<Optional<StringRef>> redirects = {
            llvm::None,  // stdin(0)
            llvm::None,  // stdout(1)
            llvm::None   // stderr(2)
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
    const auto procErr = llvm::sys::ExecuteAndWait(
            static_cast<ExecutorImpl*>(_executorPtr.get())->getApp().runProgram,
            llvm::makeArrayRef(static_cast<ExecutorImpl*>(_executorPtr.get())->getApp().runArgs),
            /*Env=*/None, llvm::makeArrayRef(redirects),
            checked_cast<uint32_t>(static_cast<ExecutorImpl*>(_executorPtr.get())->getApp().timeoutSec),
            /*MemoryLimit=*/0, &errMsg);
    VPUX_THROW_WHEN(procErr != 0, "Failed to run InferenceManagerDemo : {0}", errMsg);
}

void IMD::IMDInferRequest::readFromFile(const std::string& path, const MemoryBlob::Ptr& dataMemoryBlob) {
    const auto mem = dataMemoryBlob->rwmap();
    const auto ptr = mem.as<char*>();
    VPUX_THROW_UNLESS(ptr != nullptr, "Blob was not allocated");

    std::ifstream file(path, std::ios_base::binary | std::ios_base::ate);
    VPUX_THROW_UNLESS(file.is_open(), "Can't open file '{0}' for reading", path);

    const auto fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios_base::beg);
    VPUX_THROW_UNLESS(fileSize == dataMemoryBlob->byteSize(), "File '{0}' contains {1} bytes, but {2} expected", path,
                      fileSize, dataMemoryBlob->byteSize());

    file.read(ptr, static_cast<std::streamsize>(dataMemoryBlob->byteSize()));
}

void IMD::IMDInferRequest::loadNetworkOutputs(StringRef workDir, const BlobMap& outputs) {
    _logger.trace("Load the network outputs...");

    const auto& networkDescriptor = static_cast<ExecutorImpl*>(_executorPtr.get())->getNetworkDesc();
    const auto& deviceOutputsInfo = networkDescriptor.getDeviceOutputsInfo();

    for (const auto& p : deviceOutputsInfo | indexed) {
        const auto& blobName = p.value().first;
        const auto& outputData = outputs.at(blobName);
        const auto ind = p.index();

        const TensorDesc& outputDataAttributes = outputData->getTensorDesc();
        const TensorDesc& deviceDataAttributes = p.value().second->getTensorDesc();
        checkDataAttributesMatch(outputDataAttributes, deviceDataAttributes);

        const auto outputFilePath = printToString("{0}/output-{1}.bin", workDir, ind);
        const MemoryBlob::Ptr& dataMemoryBlob = as<MemoryBlob>(outputData);
        readFromFile(outputFilePath, dataMemoryBlob);

        _logger.nest().trace("{0} - {1}", blobName, outputFilePath);
    }

    const auto& profOutputsInfo = networkDescriptor.getDeviceProfilingOutputsInfo();
    if (profOutputsInfo.size()) {
        _logger.warning("Load profiling output");
        IE_ASSERT(profOutputsInfo.size() == 1);
        const auto& devProfInfo = profOutputsInfo.begin()->second;
        const auto& devProfMemoryBlob = as<MemoryBlob>(make_blob_with_precision(devProfInfo->getTensorDesc()));
        devProfMemoryBlob->allocate();
        readFromFile(printToString("{0}/profiling-0.bin", workDir), devProfMemoryBlob);
        _rawProfilingData = devProfMemoryBlob;
    }
}

//------------------------------------------------------------------------------
IMD::IMDInferRequest::IMDInferRequest(const ie::InputsDataMap& networkInputs, const ie::OutputsDataMap& networkOutputs,
                                      const Executor::Ptr& executor, const Config& config,
                                      const std::string& /*netName*/,
                                      const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                      const std::vector<std::shared_ptr<const ov::Node>>& results,
                                      const NetworkIOVector& networkStatesInfo,
                                      const std::shared_ptr<InferenceEngine::IAllocator>& allocator)
        : IInferRequest(networkInputs, networkOutputs),
          _executorPtr(executor),
          _config(config),
          _logger("IMDInferRequest", config.get<LOG_LEVEL>()),
          _allocator(allocator),
          _statesInfo(networkStatesInfo) {
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        IE_THROW() << "No information about network's output/input.";
    }

    _parameters = parameters;
    _results = results;

    for (const auto& networkInput : _networkInputs) {
        const std::string& inputName = networkInput.first;
        const ie::TensorDesc inputTensorDesc = networkInput.second->getTensorDesc();

        _inputs[inputName] = allocateLocalBlob(inputTensorDesc, _allocator);
    }

    for (const auto& networkOutput : _networkOutputs) {
        const std::string& outputName = networkOutput.first;
        const ie::TensorDesc outputTensorDesc = networkOutput.second->getTensorDesc();

        _outputs[outputName] = allocateLocalBlob(outputTensorDesc, _allocator);
    }
}

void IMD::IMDInferRequest::pull(const BlobMap& inputs, BlobMap& outputs) {
    _logger.info("Run inference using InferenceManagerDemo application...");
    _logger = _logger.nest();
    VPUX_SCOPE_EXIT {
        _logger = _logger.unnest();
    };

    const auto workDir = createTempWorkDir();
    VPUX_SCOPE_EXIT {
        _logger.trace("Remove the temporary working directory '{0}'...", workDir);
        const auto errc = llvm::sys::fs::remove_directories(workDir);

        if (errc) {
            _logger.error("Failed to remove temporary working directory : {0}", errc.message());
        }
    };

    storeNetworkBlob(workDir.str());
    storeNetworkInputs(workDir.str(), inputs);
    runApp(workDir.str());
    loadNetworkOutputs(workDir.str(), outputs);
}

void IMD::IMDInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void IMD::IMDInferRequest::InferAsync() {
    _logger.debug("InferRequest::InferAsync started");
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "InferAsync");

    execDataPreprocessing(_inputs);
}

IMD::LayerStatistics IMD::IMDInferRequest::GetPerformanceCounts() const {
    const auto mem = _rawProfilingData->rmap();
    const auto rawBytes = mem.as<const uint8_t*>();
    IE_ASSERT(rawBytes != nullptr);
    auto executorPtr = static_cast<ExecutorImpl*>(_executorPtr.get());
    const auto& compiledBlob = executorPtr->getNetworkDesc().getCompiledNetwork();
    return getLayerStatistics(rawBytes, _rawProfilingData->byteSize(), compiledBlob);
}

std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> IMD::IMDInferRequest::QueryState() {
    for (auto& stateInfo : _statesInfo) {
        const auto readValueName = READVALUE_PREFIX + stateInfo.first;

        IE_ASSERT(1 == _networkInputs.count(readValueName));
        IE_ASSERT(1 == _networkOutputs.count(ASSIGN_PREFIX + stateInfo.first));

        _states.push_back(std::make_shared<VariableState>(stateInfo.first, this->GetBlob(readValueName)));
    }
    return _states;
}

void IMD::IMDInferRequest::GetResult() {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "GetResult");
    pull(_inputs, _outputs);
    _logger.debug("InferRequest::GetResult finished");
}

}  // namespace vpux
