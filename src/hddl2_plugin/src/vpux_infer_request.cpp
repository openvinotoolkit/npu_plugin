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

// System
#include <map>
#include <memory>
#include <string>
#include <vector>
// IE
#include <ie_blob.h>
#include <ie_layouts.h>
// Plugin
#include "ie_itt.hpp"
#include "ie_utils.hpp"
#include "vpux_infer_request.h"
#include "vpux_remote_blob.h"
// TODO KMB-standalone preprocessing details should be not exposed to plugin [Track number: S#43193]
// Low-level
#ifdef __aarch64__
#include <kmb_preproc.hpp>
#endif

namespace vpux {
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static void checkNetworkPrecision(const IE::Precision& precision) {
    if (precision != IE::Precision::FP32 && precision != IE::Precision::FP16 && precision != IE::Precision::U8 &&
        precision != IE::Precision::I8) {
        THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input precision: " << precision
                           << "! Supported precisions: FP32, FP16, U8, I8";
    }
}

static IE::Blob::Ptr allocateLocalBlob(
    const IE::TensorDesc& tensorDesc, const std::shared_ptr<InferenceEngine::IAllocator>& allocator) {
    checkNetworkPrecision(tensorDesc.getPrecision());

    IE::Blob::Ptr blob;
    if (allocator == nullptr) {
        blob = make_blob_with_precision(tensorDesc);
    } else {
        blob = make_blob_with_precision(tensorDesc, allocator);
    }
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "InputBlob is nullptr.";
    }
    blob->allocate();
    return blob;
}

//------------------------------------------------------------------------------
InferRequest::InferRequest(const IE::InputsDataMap& networkInputs, const IE::OutputsDataMap& networkOutputs,
    const Executor::Ptr& executor, const VPUXConfig& config, const std::string& netName,
    const std::shared_ptr<InferenceEngine::IAllocator>& allocator)
    : InferRequestInternal(networkInputs, networkOutputs),
      _executorPtr(executor),
      _config(config),
      _logger(std::make_shared<vpu::Logger>("InferRequest", config.logLevel(), vpu::consoleOutput())),
      _allocator(allocator),
      _deviceId(utils::extractIdFromDeviceName(config.deviceId())),
      _netUniqueId(netName),
      _preprocBuffer(nullptr, [this](uint8_t* buffer) {
          _allocator->free(buffer);
      }) {
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "No information about network's output/input.";
    }

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

PreprocMap InferRequest::preparePreProcessing(const IE::InputsDataMap& networkInputs,
    const std::map<std::string, InferenceEngine::PreProcessDataPtr>& preProcData) {
    PreprocMap preProcMap;
    for (auto& input : networkInputs) {
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

void InferRequest::moveBlobForPreprocessingToInputs(IE::BlobMap& inputs, const IE::InputsDataMap& networkInputs,
    const std::map<std::string, InferenceEngine::PreProcessDataPtr>& preProcData) {
    for (auto& input : networkInputs) {
        const std::string inputName = input.second->name();
        const auto& preProcDataIt = preProcData.find(inputName);
        if (preProcDataIt != preProcData.end()) {
            const IE::Blob::Ptr& blobForPreProcessing = preProcDataIt->second->getRoiBlob();
            if (preProcessingRequired(input.second, blobForPreProcessing)) {
                IE::Blob::Ptr blobForPreProc = preProcDataIt->second->getRoiBlob();
                /// If pre-processing required, we need use PP blobs instead of NN for inputs
                inputs.at(inputName) = blobForPreProcessing;
            }
        }
    }
}
// TODO [Track number: S#43193]
#ifdef __aarch64__
void InferRequest::execPreprocessing(InferenceEngine::BlobMap& inputs) {
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "execPreprocessing");
    if ((_config.useSIPP() || _config.useM2I()) && IE::KmbPreproc::isApplicable(inputs, _preProcData, _networkInputs)) {
        relocationAndExecKmbDataPreprocessing(
            inputs, _networkInputs, _config.graphColorFormat(), _config.numberOfSIPPShaves(), _config.SIPPLpi());
    } else {
        _logger->warning("SIPP/M2I is enabled but configuration is not supported.");
        execDataPreprocessing(inputs);
    }
}

// TODO: SIPP preprocessing usage can be merged to common preprocessing pipeline
void InferRequest::relocationAndExecKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
    InferenceEngine::InputsDataMap& networkInputs, InferenceEngine::ColorFormat out_format, unsigned int numShaves,
    unsigned int lpi) {
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "relocationAndExecKmbDataPreprocessing");
    std::map<std::string, IE::PreProcessDataPtr> preprocDataRealloc;
    for (const auto& input : inputs) {
        const std::string& inputName = input.first;
        auto preProcDataIter = _preProcData.find(inputName);
        if (preProcDataIter == _preProcData.end()) {
            continue;
        }

        preprocDataRealloc[preProcDataIter->first] = IE::CreatePreprocDataHelper();
        IE::Blob::Ptr blobData = preProcDataIter->second->getRoiBlob();
        if (blobData->is<IE::NV12Blob>()) {
            IE::NV12Blob::Ptr origNV12Blob = IE::as<IE::NV12Blob>(blobData);
            IE::Blob::Ptr& origYBlob = origNV12Blob->y();
            IE::Blob::Ptr& origUVBlob = origNV12Blob->uv();

            IE::Blob::Ptr kmbYBlob = origYBlob;
            IE::Blob::Ptr kmbUVBlob = origUVBlob;
            if (!utils::isBlobAllocatedByAllocator(origYBlob, _allocator) ||
                !utils::isBlobAllocatedByAllocator(origUVBlob, _allocator)) {
                _logger->warning("NV12 Blob located in memory not managed by plugin. Need to re-allocate the blob.");
                _preprocBuffer.reset(
                    reinterpret_cast<uint8_t*>(_allocator->alloc(origYBlob->byteSize() + origUVBlob->byteSize())));

                auto memoryBlobY = IE::as<IE::MemoryBlob>(origYBlob);
                IE_ASSERT(memoryBlobY != nullptr);
                auto y_offset_pad = memoryBlobY->getTensorDesc().getBlockingDesc().getOffsetPadding();
                auto memoryHolderYPlane = memoryBlobY->rmap();
                ie_memcpy(_preprocBuffer.get(), origYBlob->byteSize(), memoryHolderYPlane.as<uint8_t*>() + y_offset_pad,
                    origYBlob->byteSize());
                // explicitly ignore blocking descriptor
                // memory has already been cropped properly
                // just copy precision, dimensions and layout
                InferenceEngine::TensorDesc croppedYTensorDesc = {origYBlob->getTensorDesc().getPrecision(),
                    origYBlob->getTensorDesc().getDims(), origYBlob->getTensorDesc().getLayout()};
                kmbYBlob = ie::make_shared_blob<uint8_t>(croppedYTensorDesc, _preprocBuffer.get());

                auto memoryBlobUV = IE::as<IE::MemoryBlob>(origUVBlob);
                IE_ASSERT(memoryBlobUV != nullptr);
                auto uv_offset_pad = memoryBlobUV->getTensorDesc().getBlockingDesc().getOffsetPadding();
                auto memoryHolderUVPlane = memoryBlobUV->rmap();
                ie_memcpy(_preprocBuffer.get() + origYBlob->byteSize(), origUVBlob->byteSize(),
                    memoryHolderUVPlane.as<uint8_t*>() + uv_offset_pad, origUVBlob->byteSize());
                InferenceEngine::TensorDesc croppedUVTensorDesc = {origUVBlob->getTensorDesc().getPrecision(),
                    origUVBlob->getTensorDesc().getDims(), origUVBlob->getTensorDesc().getLayout()};
                kmbUVBlob =
                    ie::make_shared_blob<uint8_t>(croppedUVTensorDesc, _preprocBuffer.get() + origYBlob->byteSize());
            }

            InferenceEngine::Blob::Ptr nv12Blob =
                InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(kmbYBlob, kmbUVBlob);
            preprocDataRealloc[preProcDataIter->first]->setRoiBlob(nv12Blob);
        } else {
            THROW_IE_EXCEPTION << "Attempt to pass non-NV12 image to Kmb preprocessing.";
        }
    }
    this->execKmbDataPreprocessing(inputs, preprocDataRealloc, networkInputs, out_format, numShaves, lpi);
}

void InferRequest::execKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
    std::map<std::string, IE::PreProcessDataPtr>& preprocData, InferenceEngine::InputsDataMap& networkInputs,
    InferenceEngine::ColorFormat out_format, unsigned int numShaves, unsigned int lpi) {
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "execKmbDataPreprocessing");
    IE_ASSERT(_config.useSIPP() || _config.useM2I());
    const IE::KmbPreproc::Path ppPath = _config.useM2I() ? IE::KmbPreproc::Path::M2I : IE::KmbPreproc::Path::SIPP;
    IE::KmbPreproc::execDataPreprocessing(
        inputs, preprocData, networkInputs, out_format, numShaves, lpi, _netUniqueId, _deviceId, ppPath);
}
#endif

void InferRequest::InferAsync() {
    // TODO [Track number: S#36866]
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "InferAsync");

    const auto preProcMap = preparePreProcessing(_networkInputs, _preProcData);
    if (_executorPtr->isPreProcessingSupported(preProcMap)) {
        moveBlobForPreprocessingToInputs(_inputs, _networkInputs, _preProcData);
        _executorPtr->push(_inputs, preProcMap);
    } else {
        // TODO [Track number: S#43193] KMB preprocessing should be moved from plugin level to backend.
#ifdef __aarch64__
        execPreprocessing(_inputs);
#else
        _logger->info("Preprocessing cannot be executed on device. IE preprocessing will be executed.");
        execDataPreprocessing(_inputs);
#endif
        _executorPtr->push(_inputs);
    }
}

void InferRequest::GetResult() {
    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "GetResult");
    _executorPtr->pull(_outputs);
}

void InferRequest::GetPerformanceCounts(std::map<std::string, IE::InferenceEngineProfileInfo>& perfMap) const {
    if (_config.performanceCounting()) {
        perfMap = _executorPtr->getLayerStatistics();
    }
}

void InferRequest::SetBlob(const char* name, const IE::Blob::Ptr& data) {
    if (!data->is<VPUXRemoteBlob>()) {
        IE::InferRequestInternal::SetBlob(name, data);
        return;
    }

    OV_ITT_SCOPED_TASK(vpu::itt::domains::KmbPlugin, "SetBlob");
    if (name == nullptr) {
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }
    if (!data) THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
    const bool isCompoundBlob = data->is<IE::CompoundBlob>();

    IE::InputInfo::Ptr foundInput;
    IE::DataPtr foundOutput;
    size_t dataSize = data->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        if (foundInput->getPrecision() != data->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding to user input precision";
        }

        const bool preProcRequired = preProcessingRequired(foundInput, data);
        if (isCompoundBlob && !preProcRequired) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "Cannot set compound blob: supported only for input pre-processing";
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
                THROW_IE_EXCEPTION << "Input blob size is not equal network input size (" << dataSize
                                   << "!=" << inputSize << ").";
            }
             */
            _inputs[name] = data;
        }
    } else {
        if (isCompoundBlob) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }
        size_t outputSize = IE::details::product(foundOutput->getDims());
        if (dataSize != outputSize) {
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size (" << dataSize
                               << "!=" << outputSize << ").";
        }
        if (foundOutput->getPrecision() != data->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding to user output precision";
        }
        _outputs[name] = data;
    }
}

void InferRequest::checkBlobs() {
    for (const auto& input : _inputs) {
        if (!input.second->is<VPUXRemoteBlob>()) checkBlob(input.second, input.first, true);
    }
    for (const auto& output : _outputs) {
        if (!output.second->is<VPUXRemoteBlob>()) checkBlob(output.second, output.first, false);
    }
}
}  // namespace vpux
