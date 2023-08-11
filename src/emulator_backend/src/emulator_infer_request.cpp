//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "emulator_infer_request.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <debug.h>
#include <ie_blob.h>
#include <ie_compound_blob.h>
#include <ie_layouts.h>
#include <blob_factory.hpp>

#include "emulator_executor.hpp"

#include <device_helpers.hpp>
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux/utils/IE/blob.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <file_utils.h>
#include <dims_parser.hpp>

namespace ie = InferenceEngine;

namespace vpux {

namespace {

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static void checkNetworkPrecision(const ie::Precision& precision) {
    if (precision != ie::Precision::FP32 && precision != ie::Precision::FP16 && precision != ie::Precision::U8 &&
        precision != ie::Precision::I8 && precision != ie::Precision::I32 && precision != ie::Precision::U32 &&
        precision != ie::Precision::I64) {
        IE_THROW(ParameterMismatch) << "Unsupported input precision: " << precision
                                    << "! Supported precisions: FP32, FP16, U8, I8, I32, I64, U32";
    }
}

static ie::Blob::Ptr allocateLocalBlob(const ie::TensorDesc& tensorDesc,
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

ie::TensorDesc getTensorDescForNewDims(const ie::TensorDesc& origDesc, const std::size_t newSz) {
    const auto actualLayout = origDesc.getLayout();
    const auto& precision = origDesc.getPrecision();

    std::size_t dimN, dimZ, dimY, dimX, dimD;
    vpu::parseDims(origDesc.getDims(), dimN, dimZ, dimY, dimX, dimD);

    switch (newSz) {
    case 3:
        VPUX_THROW_UNLESS(dimN == 1 && dimD == 1, "Expected dim N and D to be 1");
        switch (actualLayout) {
        case ie::Layout::NDHWC:
        case ie::Layout::NHWC:
            return ie::TensorDesc(precision, {dimY, dimX, dimZ}, ie::Layout::HWC);
        case ie::Layout::NCDHW:
        case ie::Layout::NCHW:
            return ie::TensorDesc(precision, {dimZ, dimY, dimX}, ie::Layout::CHW);
            ;
        default:
            VPUX_THROW("Unsupported layout for actual blob: {0}", actualLayout);
        }
    case 4:
        VPUX_THROW_UNLESS(dimD == 1, "Expected dim D to be 1");
        switch (actualLayout) {
        case ie::Layout::NDHWC:
        case ie::Layout::HWC:
            return ie::TensorDesc(precision, {dimN, dimY, dimX, dimZ}, ie::Layout::NHWC);
        case ie::Layout::NCDHW:
        case ie::Layout::CHW:
            return ie::TensorDesc(precision, {dimN, dimZ, dimY, dimX}, ie::Layout::NCHW);
        default:
            VPUX_THROW("Unsupported layout for actual blob: {0}", actualLayout);
        }
    case 5:
        switch (actualLayout) {
        case ie::Layout::NHWC:
        case ie::Layout::HWC:
            return ie::TensorDesc(precision, {dimN, dimD, dimY, dimX, dimZ}, ie::Layout::NDHWC);
        case ie::Layout::NCHW:
        case ie::Layout::CHW:
            return ie::TensorDesc(precision, {dimN, dimZ, dimD, dimY, dimX}, ie::Layout::NCDHW);
        default:
            VPUX_THROW("Unsupported layout for actual blob: {0}", actualLayout);
        }
    default:
        VPUX_THROW("Unsupported dimensions layout");
        break;
    }
}

ie::MemoryBlob::Ptr adjustDims(const ie::MemoryBlob::Ptr& tensor, const ie::TensorDesc& targetDesc) {
    const auto& actualDesc = tensor->getTensorDesc();

    if (actualDesc.getDims().size() == targetDesc.getDims().size())
        return tensor;

    const auto mem = tensor->rmap();
    const auto newDesc = getTensorDescForNewDims(actualDesc, targetDesc.getDims().size());
    return makeBlob(newDesc, nullptr, mem.as<void*>());
}

bool needsLayoutChange(const ie::Layout& origLayout, const ie::Layout& targetLayout) {
    const std::vector<ie::Layout> compatibleLayouts = {ie::Layout::C, ie::Layout::NC, ie::Layout::HW};
    const auto isCompatibleLayout = [&compatibleLayouts](const ie::Layout& layout) {
        return std::find(compatibleLayouts.begin(), compatibleLayouts.end(), layout) != compatibleLayouts.end();
    };

    return !(origLayout == targetLayout || isCompatibleLayout(origLayout) || isCompatibleLayout(targetLayout));
}

}  // namespace

EmulatorInferRequest::EmulatorInferRequest(const ie::InputsDataMap& networkInputs,
                                           const ie::OutputsDataMap& networkOutputs, const Executor::Ptr& executor,
                                           const Config& config, const std::string& netName,
                                           const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                           const std::vector<std::shared_ptr<const ov::Node>>& results,
                                           const vpux::DataMap& networkStatesInfo,
                                           const std::shared_ptr<InferenceEngine::IAllocator>& allocator)
        : IInferRequest(networkInputs, networkOutputs),
          _executorPtr(executor),
          _config(config),
          _allocator(allocator),
          _logger("EmulatorInferRequest", _config.get<LOG_LEVEL>()),
          _manager(ie::getIELibraryPath() + "/vpux_emulator", vpux::stringifyEnum(config.get<LOG_LEVEL>()).data(),
                   config.get<DEVICE_ID>()) {
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        IE_THROW() << "No information about network's output/input.";
    }
    _parameters = parameters;
    _results = results;

    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;
        const ie::TensorDesc inputTensorDesc = networkInput.second->getTensorDesc();

        _inputs[inputName] = allocateLocalBlob(inputTensorDesc, _allocator);
    }

    for (auto& networkOutput : _networkOutputs) {
        const std::string outputName = networkOutput.first;
        const ie::TensorDesc outputTensorDesc = networkOutput.second->getTensorDesc();

        _outputs[outputName] = allocateLocalBlob(outputTensorDesc, _allocator);
    }
}

ie::Blob::Ptr EmulatorInferRequest::repackTensor(const ie::Blob::Ptr& tensor, const ie::TensorDesc& targetDesc) {
    const auto& actualDesc = tensor->getTensorDesc();
    const auto& actualPrecision = actualDesc.getPrecision();
    const auto& actualLayout = actualDesc.getLayout();

    const auto& devicePrecision = targetDesc.getPrecision();
    const auto& deviceLayout = targetDesc.getLayout();

    auto tensorBlob = ie::as<ie::MemoryBlob>(tensor);

    if (actualPrecision != devicePrecision) {
        _logger.warning("Blob is inconsistent with network input/output. "
                        "Need to do convert precision from {0} to {1}.",
                        actualPrecision, devicePrecision);
        tensorBlob = toPrecision(tensorBlob, devicePrecision, vpux::None);
    }

    if (needsLayoutChange(actualLayout, deviceLayout)) {
        _logger.warning("Blob is inconsistent with network input/output. "
                        "Need to do convert layout from {0} to {1}.",
                        actualLayout, deviceLayout);

        tensorBlob = adjustDims(tensorBlob, targetDesc);
        tensorBlob = toLayout(tensorBlob, deviceLayout);
    }

    return tensorBlob;
}

void EmulatorInferRequest::push(const ie::BlobMap& inputs) {
    _logger.debug("EmulatorExecutor::push() started");

    _manager.reset(static_cast<EmulatorExecutor*>(_executorPtr.get())->getNetworkDesc().getNetworkModel());
    const auto& deviceInputs =
            static_cast<EmulatorExecutor*>(_executorPtr.get())->getNetworkDesc().getDeviceInputsInfo();
    auto inputIt = inputs.cbegin();

    for (const auto inputName : _manager.getNetworkInputs()) {
        if (deviceInputs.find(inputName) == deviceInputs.end()) {
            VPUX_THROW("Emulator inputs are different from network inputs.");
        }
        const ie::Blob::Ptr& blob = inputIt->second;
        const auto& deviceInputDesc = deviceInputs.at(inputName)->getTensorDesc();
        const auto updatedInput = repackTensor(blob, deviceInputDesc);

        _manager.populate(inputName, updatedInput->cbuffer().as<const void*>());
        ++inputIt;
    }
    _manager.run();
    _logger.debug("EmulatorExecutor::push() finished");
}

void EmulatorInferRequest::pull(ie::BlobMap& outputs) {
    _logger.debug("EmulatorExecutor::pull() started");
    const auto& deviceOutputs =
            static_cast<EmulatorExecutor*>(_executorPtr.get())->getNetworkDesc().getDeviceOutputsInfo();
    auto outputIt = outputs.begin();
    for (const auto outputName : _manager.getNetworkOutputs()) {
        if (deviceOutputs.find(outputName) == deviceOutputs.end()) {
            VPUX_THROW("Emulator outputs are different from network outputs.");
        }
        ie::Blob::Ptr blob = outputIt->second;
        const auto& deviceDesc = deviceOutputs.at(outputName)->getTensorDesc();
        auto deviceBlob = make_blob_with_precision(deviceDesc);
        deviceBlob->allocate();
        std::copy_n(_manager.data(outputName).data(), deviceBlob->byteSize(), deviceBlob->buffer().as<char*>());

        deviceBlob = repackTensor(deviceBlob, blob->getTensorDesc());
        std::copy_n(deviceBlob->buffer().as<char*>(), blob->byteSize(), blob->buffer().as<char*>());
        ++outputIt;
    }
    _logger.debug("EmulatorExecutor::pull() finished");
}

void EmulatorInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void EmulatorInferRequest::InferAsync() {
    execDataPreprocessing(_inputs);
    push(_inputs);
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    if (std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH") != nullptr) {
        dumpBlobs(_inputs, std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH"), "input");
    }
#endif
}

void EmulatorInferRequest::GetResult() {
    pull(_outputs);
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    const char* dumpOutputPathEnv = std::getenv("IE_VPU_KMB_DUMP_OUTPUT_PATH");
    if (dumpOutputPathEnv != nullptr) {
        dumpBlobs(_outputs, dumpOutputPathEnv, "output");
    }
#endif
    _logger.debug("InferRequest::GetResult finished");
}

std::map<std::string, ie::InferenceEngineProfileInfo> EmulatorInferRequest::GetPerformanceCounts() const {
    return {};
}

}  // namespace vpux
