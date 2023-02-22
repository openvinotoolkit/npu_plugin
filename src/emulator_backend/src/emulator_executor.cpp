//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "emulator_executor.hpp"

#include "vpux/al/config/common.hpp"
#include "vpux/utils/IE/blob.hpp"

#include <dims_parser.hpp>

#include <file_utils.h>

namespace ie = InferenceEngine;

namespace vpux {

namespace {

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

EmulatorExecutor::EmulatorExecutor(const vpux::NetworkDescription::Ptr& network, const Config& config)
        : _logger("EmulatorBackend", LogLevel::Debug /*_config.logLevel()*/),
          _config(config),
          _network(network),
          _manager(ie::getIELibraryPath() + "/vpux_emulator", vpux::stringifyEnum(config.get<LOG_LEVEL>()).data(),
                   config.get<DEVICE_ID>()) {
}

ie::Blob::Ptr EmulatorExecutor::repackTensor(const ie::Blob::Ptr& tensor, const ie::TensorDesc& targetDesc) {
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

void EmulatorExecutor::push(const ie::BlobMap& inputs, const PreprocMap&) {
    push(inputs);
}

void EmulatorExecutor::push(const ie::BlobMap& inputs) {
    _logger.debug("EmulatorExecutor::push() started");
    _manager.reset(_network->getNetworkModel());

    const auto& deviceInputs = _network->getDeviceInputsInfo();
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

void EmulatorExecutor::pull(ie::BlobMap& outputs) {
    _logger.debug("EmulatorExecutor::pull() started");
    const auto& deviceOutputs = _network->getDeviceOutputsInfo();
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

Executor::Ptr EmulatorExecutor::clone() const {
    return std::make_shared<EmulatorExecutor>(this->_network, this->_config);
}

}  // namespace vpux
