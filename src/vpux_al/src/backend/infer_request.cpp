//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <debug.h>
#include <ie_blob.h>
#include <ie_compound_blob.h>
#include <ie_layouts.h>
#include <blob_factory.hpp>

#include "vpux.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux_remote_blob.h"

#include <device_helpers.hpp>
#include "vpux/utils/IE/blob.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/checked_cast.hpp"

namespace vpux {
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static void checkNetworkPrecision(const IE::Precision& precision) {
    if (precision != IE::Precision::FP32 && precision != IE::Precision::FP16 && precision != IE::Precision::U8 &&
        precision != IE::Precision::I8 && precision != IE::Precision::I32 && precision != IE::Precision::U32 &&
        precision != IE::Precision::I64) {
        IE_THROW(ParameterMismatch) << "Unsupported input precision: " << precision
                                    << "! Supported precisions: FP32, FP16, U8, I8, I32, I64, U32";
    }
}

static IE::Blob::Ptr allocateLocalBlob(const IE::TensorDesc& tensorDesc,
                                       const std::shared_ptr<InferenceEngine::IAllocator>& allocator) {
    checkNetworkPrecision(tensorDesc.getPrecision());

    IE::Blob::Ptr blob;
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

//------------------------------------------------------------------------------
InferRequest::InferRequest(const IE::InputsDataMap& networkInputs, const IE::OutputsDataMap& networkOutputs,
                           const Executor::Ptr& executor, const Config& config, const std::string& netName,
                           const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                           const std::vector<std::shared_ptr<const ov::Node>>& results,
                           const vpux::DataMap& /* networkStatesInfo */,
                           const std::shared_ptr<InferenceEngine::IAllocator>& allocator)
        : IInferRequest(networkInputs, networkOutputs),
          _executorPtr(executor),
          _config(config),
          _logger("InferRequest", config.get<LOG_LEVEL>()),
          _allocator(allocator),
          _deviceId(utils::getSliceIdByDeviceName(config.get<DEVICE_ID>())),
          _netUniqueId(netName),
          _preprocBuffer(nullptr, [this](uint8_t* buffer) {
              _allocator->free(buffer);
          }) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "InferRequest::InferRequest");
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        IE_THROW() << "No information about network's output/input.";
    }
    _parameters = parameters;
    _results = results;

    for (const auto& networkInput : _networkInputs) {
        const std::string inputName = networkInput.first;
        const IE::TensorDesc inputTensorDesc = networkInput.second->getTensorDesc();

        _inputs[inputName] = allocateLocalBlob(inputTensorDesc, _allocator);
    }

    for (auto& networkOutput : _networkOutputs) {
        const std::string outputName = networkOutput.first;
        const IE::TensorDesc outputTensorDesc = networkOutput.second->getTensorDesc();

        _outputs[outputName] = allocateLocalBlob(outputTensorDesc, _allocator);
    }
}

void InferRequest::Infer() {
    checkBlobs();
    InferImpl();
}

void InferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

PreprocMap InferRequest::preparePreProcessing(
        const IE::InputsDataMap& networkInputs,
        const std::map<std::string, InferenceEngine::PreProcessDataPtr>& preProcData) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "InferRequest::preparePreProcessing");
    PreprocMap preProcMap;
    for (const auto& input : networkInputs) {
        const std::string inputName = input.second->name();
        const auto& preProcDataIt = preProcData.find(inputName);
        if (preProcDataIt != preProcData.end()) {
            const IE::Blob::Ptr& blobForPreProcessing = preProcDataIt->second->getRoiBlob();
            if (preProcessingRequired(input.second, blobForPreProcessing)) {
                preProcMap.emplace(input.first, input.second->getPreProcess());
            }
        }
    }
    return preProcMap;
}

void InferRequest::moveBlobsForPreprocessingToInputs(
        IE::BlobMap& inputs, const IE::InputsDataMap& networkInputs,
        const std::map<std::string, InferenceEngine::PreProcessDataPtr>& preProcData) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "InferRequest::moveBlobsForPreprocessingToInputs");
    for (auto& input : networkInputs) {
        const std::string inputName = input.second->name();
        const auto& preProcDataIt = preProcData.find(inputName);
        if (preProcDataIt != preProcData.end()) {
            const IE::Blob::Ptr& blobForPreProcessing = preProcDataIt->second->getRoiBlob();
            if (preProcessingRequired(input.second, blobForPreProcessing)) {
                IE::Blob::Ptr blobForPreProc = preProcDataIt->second->getRoiBlob();
                /// If pre-processing required, we need use PP blobs instead of NN for inputs
                inputs.at(inputName) = blobForPreProc;
            }
        }
    }
}

void InferRequest::InferAsync() {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "InferRequest::InferAsync");

    const auto preProcMap = preparePreProcessing(_networkInputs, _preProcData);
    if (_executorPtr->isPreProcessingSupported(preProcMap)) {
        moveBlobsForPreprocessingToInputs(_inputs, _networkInputs, _preProcData);
        updateRemoteBlobs(_inputs, preProcMap);
        _executorPtr->push(_inputs, preProcMap);
    } else {
        _logger.info("Preprocessing cannot be executed on device. IE preprocessing will be executed.");
        execDataPreprocessing(_inputs);
        updateRemoteBlobs(_inputs, preProcMap);
        _executorPtr->push(_inputs);
    }
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    if (std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH") != nullptr) {
        dumpBlobs(_inputs, std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH"), "input");
    }
#endif
}

void InferRequest::GetResult() {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "InferRequest::GetResult");
    _executorPtr->pull(_outputs);
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    const char* dumpOutputPathEnv = std::getenv("IE_VPU_KMB_DUMP_OUTPUT_PATH");
    if (dumpOutputPathEnv != nullptr) {
        dumpBlobs(_outputs, dumpOutputPathEnv, "output");
    }
#endif
    _logger.debug("InferRequest::GetResult finished");
}

std::map<std::string, IE::InferenceEngineProfileInfo> InferRequest::GetPerformanceCounts() const {
    if (_config.get<PERF_COUNT>()) {
        return _executorPtr->getLayerStatistics();
    } else {
        return {};
    }
}

void InferRequest::SetBlob(const std::string& name, const IE::Blob::Ptr& data) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "InferRequest::SetBlob");
    if (!isRemoteAnyBlob(data)) {
        IE::IInferRequestInternal::SetBlob(name, data);
        return;
    }
    if (name.empty()) {
        IE_THROW(NotFound) << "Failed to set blob with empty name";
    }
    if (!data)
        IE_THROW(NotAllocated) << "Failed to set empty blob with name: \'" << name << "\'";
    const bool isCompoundBlob = data->is<IE::CompoundBlob>();

    IE::InputInfo::Ptr foundInput;
    IE::DataPtr foundOutput;
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        if (foundInput->getPrecision() != data->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                    << "Failed to set Blob with precision not corresponding to user input precision";
        }

        // We need to revert results of moveBlobsForPreprocessingToInputs() due to preprocessing requirements
        _inputs[name] = allocateLocalBlob(foundInput->getTensorDesc(), _allocator);
        const bool preProcRequired = preProcessingRequired(foundInput, data);
        if (isCompoundBlob && !preProcRequired) {
            IE_THROW(NotImplemented) << "Cannot set compound blob: supported only for input pre-processing";
        }

        if (preProcRequired) {
            if (_preProcData.find(name) == _preProcData.end()) {
                _preProcData.emplace(name, IE::CreatePreprocDataHelper());
            }
            _preProcData[name]->isApplicable(data, _inputs[name]);
            _preProcData[name]->setRoiBlob(data);
        } else {
            // TODO In ROI case in might be not true. How to handle this?
            /*
            size_t inputSize = IE::details::product(foundInput->getTensorDesc().getDims());
            if (dataSize != inputSize) {
                IE_THROW() << "Input blob size is not equal network input size (" << dataSize
                                   << "!=" << inputSize << ").";
            }
             */
            _inputs[name] = data;
        }
    } else {
        if (isCompoundBlob) {
            IE_THROW(NotImplemented) << "cannot set compound blob: supported only for input pre-processing";
        }

        size_t dataSize = data->size();
        size_t outputSize = IE::details::product(foundOutput->getDims());
        if (dataSize != outputSize) {
            IE_THROW() << "Output blob size is not equal network output size (" << dataSize << "!=" << outputSize
                       << ").";
        }
        if (foundOutput->getPrecision() != data->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                    << "Failed to set Blob with precision not corresponding to user output precision";
        }
        _outputs[name] = data;
    }
}

void InferRequest::checkBlobs() {
    for (const auto& input : _inputs) {
        if (!input.second->is<VPUXRemoteBlob>())
            checkBlob(input.second, input.first, true);
    }
    for (const auto& output : _outputs) {
        if (!output.second->is<VPUXRemoteBlob>())
            checkBlob(output.second, output.first, false);
    }
}

void InferRequest::updateRemoteBlobs(IE::BlobMap& inputs, const PreprocMap& preProcMap) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "InferRequest::updateRemoteBlobs");
    for (auto& input : inputs) {
        auto colorFormat = IE::ColorFormat::BGR;

        const std::string inputName = input.first;
        const auto& preProcIt = preProcMap.find(inputName);
        if (preProcIt != preProcMap.end()) {
            colorFormat = preProcIt->second.getColorFormat();
        }

        auto inputBlob = input.second;
        updateRemoteBlobColorFormat(inputBlob, colorFormat);
    }
}

void InferRequest::updateRemoteBlobColorFormat(InferenceEngine::Blob::Ptr& blob,
                                               const InferenceEngine::ColorFormat colorFormat) {
    if (!isRemoteAnyBlob(blob))
        return;

    VPUXRemoteBlob::Ptr remoteBlob = nullptr;
    if (isRemoteBlob(blob)) {
        remoteBlob = IE::as<VPUXRemoteBlob>(blob);
        if (remoteBlob == nullptr) {
            IE_THROW() << "Failed to cast to the VPUXRemoteBlob.";
        }
        remoteBlob->updateColorFormat(colorFormat);
    } else if (isRemoteNV12Blob(blob)) {
        IE::NV12Blob::Ptr nv12Blob = IE::as<IE::NV12Blob>(blob);

        remoteBlob = IE::as<VPUXRemoteBlob>(nv12Blob->y());
        if (remoteBlob == nullptr) {
            IE_THROW() << "Failed to cast Y plane to the VPUXRemoteBlob.";
        }
        remoteBlob->updateColorFormat(colorFormat);

        remoteBlob = IE::as<VPUXRemoteBlob>(nv12Blob->uv());
        if (remoteBlob == nullptr) {
            IE_THROW() << "Failed to cast UV plane to the VPUXRemoteBlob.";
        }
        remoteBlob->updateColorFormat(colorFormat);
    }
}

}  // namespace vpux
