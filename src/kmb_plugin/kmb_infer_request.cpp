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

#include <memory>
#define NOMINMAX
#include <debug.h>
#include <ie_blob.h>
#include <ie_layouts.h>
#include <precision_utils.h>

#include <description_buffer.hpp>
#include <dims_parser.hpp>
#include <dumper.hpp>
#include <ie_itt.hpp>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/perf_report.hpp>

#include "kmb_infer_request.h"

// TODO: split headers and source for utils
#include <ie_utils.hpp>
#include <vpu/utils/ie_helpers.hpp>

#include "kmb_executable_network.h"
#include "kmb_preproc.hpp"

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;

KmbInferRequest::KmbInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
    const InferenceEngine::OutputsDataMap& networkOutputs, const std::vector<vpu::StageMetaInfo>& blobMetaData,
    const KmbConfig& kmbConfig, const std::shared_ptr<vpux::Executor>& executor,
    const std::shared_ptr<InferenceEngine::IAllocator>& allocator, const std::string& netName, const int deviceId)
    : InferRequestInternal(networkInputs, networkOutputs),
      _executor(executor),
      _allocator(allocator),
      _stagesMetaData(blobMetaData),
      _config(kmbConfig),
      _netUniqueId(netName),
      _deviceId(deviceId),
      _prepprocBuffer(nullptr,
          [this](uint8_t* buffer) {
              _allocator->free(buffer);
          }),
      _logger(std::make_shared<Logger>("KmbInferRequest", kmbConfig.logLevel(), consoleOutput())) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "KmbInferRequest");
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "Internal error: no information about network's output/input";
    }

    size_t inputsTotalSize = 0;
    for (const auto& networkInput : _networkInputs) {
        Precision precision = networkInput.second->getTensorDesc().getPrecision();

        if (precision != Precision::FP32 && precision != Precision::FP16 && precision != Precision::U8 &&
            precision != Precision::I8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input precision: " << precision
                               << "! Supported precisions: FP32, FP16, U8, I8";
        }

        Blob::Ptr inputBlob = make_blob_with_precision(networkInput.second->getTensorDesc(), allocator);
        inputBlob->allocate();
        _inputs[networkInput.first] = inputBlob;
        inputsTotalSize += inputBlob->byteSize();
    }

    size_t outputsTotalSize = 0;
    for (const auto& networkOutput : _networkOutputs) {
        Precision precision = networkOutput.second->getTensorDesc().getPrecision();

        if (precision != Precision::FP32 && precision != Precision::FP16 && precision != Precision::U8 &&
            precision != Precision::I8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output precision: " << precision
                               << "! Supported precisions: FP32, FP16, U8, I8";
        }

        Blob::Ptr outputBlob = nullptr;
        outputBlob = make_blob_with_precision(networkOutput.second->getTensorDesc(), allocator);
        outputBlob->allocate();
        _outputs[networkOutput.first] = outputBlob;
        outputsTotalSize += outputBlob->byteSize();
    }
}
void KmbInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void KmbInferRequest::InferAsync() {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "InferAsync");
    execPreprocessing(_inputs);

    if (std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH") != nullptr) {
        utils::dumpBlobs(_inputs, std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH"), "input", _logger);
    }

    _executor->push(_inputs);
}

void KmbInferRequest::execPreprocessing(InferenceEngine::BlobMap& inputs) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "execPreprocessing");
    if ((_config.useSIPP() || _config.useM2I()) && KmbPreproc::isApplicable(inputs, _preProcData, _networkInputs)) {
        relocationAndExecKmbDataPreprocessing(
            inputs, _networkInputs, _config.outColorFmtSIPP(), _config.numberOfSIPPShaves(), _config.SIPPLpi());
    } else {
        _logger->warning("SIPP/M2I is enabled but configuration is not supported.");
        execDataPreprocessing(inputs);
    }
}

// TODO: SIPP preprocessing usage can be merged to common preprocessing pipeline
void KmbInferRequest::relocationAndExecKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
    InferenceEngine::InputsDataMap& networkInputs, InferenceEngine::ColorFormat out_format, unsigned int numShaves,
    unsigned int lpi) {
    std::map<std::string, PreProcessDataPtr> preprocDataRealloc;
    for (const auto& input : inputs) {
        const std::string& inputName = input.first;
        auto preProcDataIter = _preProcData.find(inputName);
        if (preProcDataIter == _preProcData.end()) {
            continue;
        }

        preprocDataRealloc[preProcDataIter->first] = CreatePreprocDataHelper();
        Blob::Ptr blobData = preProcDataIter->second->getRoiBlob();
        if (blobData->is<NV12Blob>()) {
            NV12Blob::Ptr origNV12Blob = as<NV12Blob>(blobData);
            Blob::Ptr& origYBlob = origNV12Blob->y();
            Blob::Ptr& origUVBlob = origNV12Blob->uv();

            Blob::Ptr kmbYBlob = origYBlob;
            Blob::Ptr kmbUVBlob = origUVBlob;
            if (!utils::isBlobAllocatedByAllocator(origYBlob, _allocator) ||
                !utils::isBlobAllocatedByAllocator(origUVBlob, _allocator)) {
                _logger->warning("NV12 Blob located in memory not managed by plugin. Need to re-allocate the blob.");
                _prepprocBuffer.reset(
                    reinterpret_cast<uint8_t*>(_allocator->alloc(origYBlob->byteSize() + origUVBlob->byteSize())));

                auto memoryBlobY = as<MemoryBlob>(origYBlob);
                auto memoryHolderYPlane = memoryBlobY->rmap();
                ie_memcpy(_prepprocBuffer.get(), origYBlob->byteSize(), memoryHolderYPlane.as<uint8_t*>(),
                    origYBlob->byteSize());
                // explicitly ignore blocking descriptor
                // memory has already been cropped properly
                // just copy precision, dimensions and layout
                InferenceEngine::TensorDesc croppedYTensorDesc = {origYBlob->getTensorDesc().getPrecision(),
                    origYBlob->getTensorDesc().getDims(), origYBlob->getTensorDesc().getLayout()};
                kmbYBlob = ie::make_shared_blob<uint8_t>(croppedYTensorDesc, _prepprocBuffer.get());

                auto memoryBlobUV = as<MemoryBlob>(origUVBlob);
                auto memoryHolderUVPlane = memoryBlobUV->rmap();
                ie_memcpy(_prepprocBuffer.get() + origYBlob->byteSize(), origUVBlob->byteSize(),
                    memoryHolderUVPlane.as<uint8_t*>(), origUVBlob->byteSize());
                InferenceEngine::TensorDesc croppedUVTensorDesc = {origUVBlob->getTensorDesc().getPrecision(),
                    origUVBlob->getTensorDesc().getDims(), origUVBlob->getTensorDesc().getLayout()};
                kmbUVBlob =
                    ie::make_shared_blob<uint8_t>(croppedUVTensorDesc, _prepprocBuffer.get() + origYBlob->byteSize());
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

void KmbInferRequest::execKmbDataPreprocessing(InferenceEngine::BlobMap& inputs,
    std::map<std::string, PreProcessDataPtr>& preprocData, InferenceEngine::InputsDataMap& networkInputs,
    InferenceEngine::ColorFormat out_format, unsigned int numShaves, unsigned int lpi) {
    IE_ASSERT(_config.useSIPP() || _config.useM2I());
    const KmbPreproc::Path ppPath = _config.useM2I() ? KmbPreproc::Path::M2I : KmbPreproc::Path::SIPP;
    KmbPreproc::execDataPreprocessing(
        inputs, preprocData, networkInputs, out_format, numShaves, lpi, _netUniqueId, _deviceId, ppPath);
}

void KmbInferRequest::GetResult() {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "GetResult");

    _logger->debug("InferRequest::GetResult started");
    // FIXME: need a test checks if output provided or not
    _executor->pull(_outputs);
    const char* dumpOutputPathEnv = std::getenv("IE_VPU_KMB_DUMP_OUTPUT_PATH");
    if (dumpOutputPathEnv != nullptr) {
        utils::dumpBlobs(_outputs, dumpOutputPathEnv, "output", _logger);
    }
    _logger->debug("InferRequest::GetResult finished");
}

void KmbInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>&) const {
    THROW_IE_EXCEPTION << "KmbInferRequest::GetPerformanceCounts is not implemented\n";
}
