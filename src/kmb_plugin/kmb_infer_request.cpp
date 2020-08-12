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

#define NOMINMAX
#include "kmb_infer_request.h"

#include <debug.h>
#include <ie_blob.h>
#include <ie_layouts.h>
#include <precision_utils.h>

#include <description_buffer.hpp>
#include <ie_itt.hpp>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/perf_report.hpp>

#include "dims_parser.hpp"
#include "dumper.hpp"
#include "ie_utils.hpp"
#include "kmb_executable_network.h"
#include "kmb_preproc.hpp"

using namespace utils;
using namespace vpu::KmbPlugin;
using namespace vpu::KmbPlugin::utils;
using namespace InferenceEngine;

KmbInferRequest::KmbInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
    const InferenceEngine::OutputsDataMap& networkOutputs, const std::vector<vpu::StageMetaInfo>& blobMetaData,
    const KmbConfig& kmbConfig, const KmbExecutor::Ptr& executor)
    : InferRequestInternal(networkInputs, networkOutputs),
      _executor(executor),
      _stagesMetaData(blobMetaData),
      _config(kmbConfig),
      _logger(std::make_shared<Logger>("KmbInferRequest", kmbConfig.logLevel(), consoleOutput())),
      _deallocateHelper([this](uint8_t* ptr) {
          this->_executor->getKmbAllocator()->free(ptr);
      }),
      _inputBuffer(nullptr, _deallocateHelper),
      _outputBuffer(nullptr, _deallocateHelper) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "KmbInferRequest");
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "Internal error: no information about network's output/input";
    }

    size_t inputsTotalSize = 0;
    for (auto& networkInput : _networkInputs) {
        Precision precision = networkInput.second->getTensorDesc().getPrecision();

        if (precision != Precision::FP32 && precision != Precision::FP16 && precision != Precision::U8 &&
            precision != Precision::I8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input precision: " << precision
                               << "! Supported precisions: FP32, FP16, U8, I8";
        }

        Blob::Ptr inputBlob =
            make_blob_with_precision(networkInput.second->getTensorDesc(), _executor->getKmbAllocator());
        inputBlob->allocate();
        _inputs[networkInput.first] = inputBlob;
        inputsTotalSize += inputBlob->byteSize();
    }
    if (_networkInputs.size() > 1) {
        uint8_t* inputsRawPtr = reinterpret_cast<uint8_t*>(_executor->getKmbAllocator()->alloc(inputsTotalSize));
        _inputBuffer.reset(inputsRawPtr);
    }

    size_t outputsTotalSize = 0;
    for (auto& networkOutput : _networkOutputs) {
        Precision precision = networkOutput.second->getTensorDesc().getPrecision();

        if (precision != Precision::FP32 && precision != Precision::FP16 && precision != Precision::U8 &&
            precision != Precision::I8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output precision: " << precision
                               << "! Supported precisions: FP32, FP16, U8, I8";
        }

        Blob::Ptr outputBlob = nullptr;
        outputBlob = make_blob_with_precision(networkOutput.second->getTensorDesc(), _executor->getKmbAllocator());
        outputBlob->allocate();
        _outputs[networkOutput.first] = outputBlob;
        outputsTotalSize += outputBlob->byteSize();
    }
    uint8_t* outputsRawPtr = reinterpret_cast<uint8_t*>(_executor->getKmbAllocator()->alloc(outputsTotalSize));
    _outputBuffer.reset(outputsRawPtr);
}
void KmbInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void KmbInferRequest::InferAsync() {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "InferAsync");
    execPreprocessing(_inputs);

    if (std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH") != nullptr) {
        dumpBlobs(_inputs, std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH"), "input", _logger);
    }

    // TODO: would be better to find a better place for such checks
    for (const auto& input : _inputs) {
        auto const inputBlobPtr = input.second;
        auto inputBlobPrecision = inputBlobPtr->getTensorDesc().getPrecision();
        if (inputBlobPrecision != Precision::FP16 && inputBlobPrecision != Precision::FP32 &&
            inputBlobPrecision != Precision::I8 && inputBlobPrecision != Precision::U8)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input blob precision";
    }
    for (const auto& output : _outputs) {
        auto const outputBlobPtr = output.second;
        auto outputBlobPrecision = outputBlobPtr->getTensorDesc().getPrecision();
        if (outputBlobPrecision != Precision::FP16 && outputBlobPrecision != Precision::FP32 &&
            outputBlobPrecision != Precision::I8 && outputBlobPrecision != Precision::U8)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output blob precision";
    }

    const auto& deviceInputs = _executor->getDeviceInputs();
    if (deviceInputs.begin() == deviceInputs.end()) THROW_IE_EXCEPTION << "DeviceInputs are empty.";
    if (deviceInputs.size() != _inputs.size()) THROW_IE_EXCEPTION << "DeviceInputs and _inputs sizes are different.";

    if (_inputs.size() == 1) {
        // avoid memory copy for single input
        const auto& deviceInputDesc = deviceInputs.begin()->second->getTensorDesc();
        const auto& input = _inputs.begin()->second;
        auto updatedInput = prepareInputForInference(input, deviceInputDesc);
        _executor->queueInference(updatedInput->buffer().as<void*>(), updatedInput->byteSize());
    } else {
        size_t inputBufferOffset = 0;
        for (const auto& inferInput : _inputs) {
            std::string inputName = inferInput.first;
            const auto deviceInputDesc = deviceInputs.at(inputName)->getTensorDesc();
            const auto input = inferInput.second;

            auto updatedInput = prepareInputForInference(input, deviceInputDesc);

            // TODO implement memory copy inside prepareInputForInference
            ie_memcpy(_inputBuffer.get() + inputBufferOffset, _inputs[inferInput.first]->byteSize(),
                updatedInput->buffer().as<uint8_t*>(), updatedInput->byteSize());

            inputBufferOffset += updatedInput->byteSize();
        }
        _executor->queueInference(_inputBuffer.get(), inputBufferOffset);
    }
}

void KmbInferRequest::checkConfigsAndExecPreprocessing(InferenceEngine::BlobMap& inputs, bool useSipp) {
    if ((useSipp || _config.useM2I()) && KmbPreproc::isApplicable(inputs, _preProcData, _networkInputs)) {
        relocationAndExecKmbDataPreprocessing(
            inputs, _networkInputs, _config.outColorFmtSIPP(), _config.numberOfSIPPShaves(), _config.SIPPLpi());
    } else {
        _logger->warning("SIPP/M2I is enabled but configuration is not supported.");
        execDataPreprocessing(inputs);
    }
}

void KmbInferRequest::execPreprocessing(InferenceEngine::BlobMap& inputs) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "execPreprocessing");
    // TODO: [Track number: S#31121]
    // Get rid of environment variable USE_SIPP
    if (getenv("USE_SIPP") != nullptr) {
        checkConfigsAndExecPreprocessing(inputs, KmbPreproc::useSIPP());
    } else {
        checkConfigsAndExecPreprocessing(inputs, _config.useSIPP());
    }
}

static bool isBlobPlacedInShareableMemory(const Blob::Ptr& blob, const KmbAllocator::Ptr& allocator) {
    return allocator->isValidPtr(blob->buffer().as<void*>());
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
            // check if planes of nv12 blob were allocated with KMB allocator
            NV12Blob::Ptr origNV12Blob = as<NV12Blob>(blobData);
            Blob::Ptr& origYBlob = origNV12Blob->y();
            Blob::Ptr& origUVBlob = origNV12Blob->uv();

            Blob::Ptr kmbYBlob = origYBlob;
            if (!isBlobPlacedInShareableMemory(origYBlob, _executor->getKmbAllocator())) {
                kmbYBlob = reallocateBlob(origYBlob);
            }
            Blob::Ptr kmbUVBlob = origUVBlob;
            if (!isBlobPlacedInShareableMemory(origUVBlob, _executor->getKmbAllocator())) {
                kmbUVBlob = reallocateBlob(origUVBlob);
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
    IE_ASSERT(_config.useSIPP() || KmbPreproc::useSIPP() || _config.useM2I());
    const KmbPreproc::Path ppPath = _config.useM2I() ? KmbPreproc::Path::M2I : KmbPreproc::Path::SIPP;
    KmbPreproc::execDataPreprocessing(inputs, preprocData, networkInputs, out_format, numShaves, lpi, ppPath);
}

static bool needRepackForNHWC(const TensorDesc& actualDesc) {
    /* NB: Brief overview:
     * Runtime works only with NHWC layout, but actual input layout can be different
     * therefore it should be repacked, let's to observe cases:
         1) NC & C there isn't necessary to do repacking,
            because these layouts has the same representation in NCHW & NHWC
         2) NHWC isn't necessary to do repacking obviously
         3) NCHW in general case it should be repacked, however if it is 11HW it isn't necessary
         4) CHW the same as for NCHW case, it isn't necessary to do repacking in 1HW case
     */
    const auto actualLayout = actualDesc.getLayout();
    const auto& actualDims = actualDesc.getDims();
    switch (actualLayout) {
    case Layout::NHWC:
    case Layout::NC:
    case Layout::C:
        return false;
    case Layout::NCHW:
        return (actualDims[0] != 1) || (actualDims[1] != 1);
    case Layout::CHW:
        return actualDims[0] != 1;
    default:
        THROW_IE_EXCEPTION << "Unsupported layout for actual blob: " << actualLayout;
    }
}

static Blob::Ptr reallocateBlobToLayoutIgnoringOriginalLayout(
    const Blob::Ptr& blob, const Layout& srcLayout, const Layout& dstLayout, const KmbAllocator::Ptr& allocator) {
    if (blob->getTensorDesc().getDims()[1] != 3) {
        THROW_IE_EXCEPTION << "reallocateBlobToLayoutIgnoringOriginalLayout works only with channels == 3";
    }

    // it would be nicer to construct srcTensorDesc from tensorDesc of blob
    // and then call srcTensorDesc.setLayout(srcLayout) but copyBlob does work in that case
    TensorDesc srcTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), srcLayout};
    Blob::Ptr srcBlob = make_blob_with_precision(srcTensorDesc, blob->buffer());

    TensorDesc dstTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), dstLayout};
    Blob::Ptr dstBlob = make_blob_with_precision(dstTensorDesc, allocator);
    dstBlob->allocate();

    vpu::copyBlob(srcBlob, dstBlob);
    return dstBlob;
}

static Blob::Ptr reallocateBlobToLayout(
    const Blob::Ptr& blob, const Layout& layout, const KmbAllocator::Ptr& allocator) {
    TensorDesc dstTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), layout};
    Blob::Ptr kmbBlob = make_blob_with_precision(dstTensorDesc, allocator);
    kmbBlob->allocate();

    vpu::copyBlob(blob, kmbBlob);

    return kmbBlob;
}

Blob::Ptr KmbInferRequest::reallocateBlob(const Blob::Ptr& blob) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "reallocateBlob");
    return reallocateBlobToLayout(blob, blob->getTensorDesc().getLayout(), _executor->getKmbAllocator());
}

Blob::Ptr KmbInferRequest::prepareInputForInference(
    const ie::Blob::Ptr& actualInput, const InferenceEngine::TensorDesc& deviceDesc) {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "prepareInputForInference");

    // HACK: to overcome inability python API to pass a blob of NHWC layout
    if (_config.forceNCHWToNHWC()) {
        _logger->warning("VPU_KMB_FORCE_NCHW_TO_NHWC is enabled. Need to do re-layout.");
        return reallocateBlobToLayoutIgnoringOriginalLayout(
            actualInput, Layout::NCHW, Layout::NHWC, _executor->getKmbAllocator());
    }

    Blob::Ptr inputForInference;
    if (!isBlobPlacedInShareableMemory(actualInput, _executor->getKmbAllocator())) {
        _logger->warning("Input blob is located in non-shareable memory. Need to do re-allocation.");
        inputForInference = reallocateBlob(actualInput);
    } else {
        inputForInference = actualInput;
    }

    const auto& actualDesc = actualInput->getTensorDesc();
    const auto& deviceLayout = deviceDesc.getLayout();

    if (needRepackForNHWC(actualDesc) && deviceLayout == Layout::NHWC) {
        _logger->warning("Input blob is inconsistent with network input. Need to do re-layout.");
        // NB: It's possible to make repack data only with the same number of dimensions
        // So just make a view without any copy
        const auto outputMemoryBlob = as<MemoryBlob>(actualInput);
        IE_ASSERT(outputMemoryBlob != nullptr);
        const auto outputMemory = outputMemoryBlob->rmap();
        IE_ASSERT(outputMemory != nullptr);
        const auto outputPtr = outputMemory.as<void*>();
        IE_ASSERT(outputPtr != nullptr);
        Blob::Ptr actualView4D = make_blob_with_precision(getNCHW(actualInput->getTensorDesc()), outputPtr);
        inputForInference = reallocateBlobToLayout(actualView4D, deviceLayout, _executor->getKmbAllocator());
    }

    return inputForInference;
}

void KmbInferRequest::GetResult() {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "GetResult");
    auto dataName = _networkOutputs.begin()->first;

    auto foundInputBlob = _outputs.find(dataName);
    if (foundInputBlob == _outputs.end()) THROW_IE_EXCEPTION << "Error: output [" << dataName << "] is not provided.";

    // check that output layout is the same as device layout
    const InferenceEngine::OutputsDataMap& deviceOutputs = _executor->getDeviceOutputs();
    IE_ASSERT(!deviceOutputs.empty());

    size_t output_size_total = std::accumulate(
        _outputs.begin(), _outputs.end(), 0, [](size_t sum, InferenceEngine::BlobMap::value_type& outputs) {
            return sum + outputs.second->byteSize();
        });

    Blob::Ptr& outputBlobRef = _outputs.begin()->second;
    InferenceEngine::TensorDesc deviceTensorDesc = deviceOutputs.begin()->second->getTensorDesc();
    InferenceEngine::TensorDesc outputTensorDesc = outputBlobRef->getTensorDesc();

    InferenceEngine::Precision devicePrecision = deviceTensorDesc.getPrecision();
    InferenceEngine::Precision outputPrecision = outputTensorDesc.getPrecision();
    InferenceEngine::Layout deviceLayout = deviceTensorDesc.getLayout();
    InferenceEngine::Layout outputLayout = outputTensorDesc.getLayout();

    if (_outputs.size() == 1 && devicePrecision == outputPrecision && deviceLayout == outputLayout) {
        // read result directly into output, do not copy blob
        void* outputPtr = outputBlobRef->buffer();
        _executor->getResult(outputPtr, output_size_total);
    } else {
        _executor->getResult(_outputBuffer.get(), output_size_total);
        size_t outputBufferOffset = 0;
        for (const auto& inferOutput : _outputs) {
            std::string outputName = inferOutput.first;
            const auto deviceOutputDesc = deviceOutputs.at(outputName)->getTensorDesc();
            const auto outputBlob = inferOutput.second;
            const auto inferOutputDesc = outputBlob->getTensorDesc();

            const Blob::Ptr devOutBlob =
                make_blob_with_precision(deviceOutputDesc, _outputBuffer.get() + outputBufferOffset);
            // do precision conversion when necessary
            Blob::Ptr blobWithCorrectPrecision = convertPrecision(devOutBlob, inferOutputDesc.getPrecision());

            // copy blob with correct precision to the output blob
            // copyBlob does layout conversion on its own
            const auto outputMemoryBlob = as<MemoryBlob>(outputBlob);
            IE_ASSERT(outputMemoryBlob != nullptr);
            const auto outputMemory = outputMemoryBlob->rmap();
            IE_ASSERT(outputMemory != nullptr);
            const auto outputPtr = outputMemory.as<void*>();
            IE_ASSERT(outputPtr != nullptr);
            if (needRepackForNHWC(inferOutputDesc) && deviceLayout == Layout::NHWC) {
                _logger->warning("Output blob is inconsistent with network output. Need to do re-layout.");
                // NB: It's possible to make repack data only with the same number of dimensions
                // So just make a view without any copy
                const auto actualView4D = make_blob_with_precision(getNCHW(inferOutputDesc), outputPtr);
                copyBlob(blobWithCorrectPrecision, actualView4D);
            } else {
                copyBlob(blobWithCorrectPrecision, deviceOutputDesc.getLayout(), outputPtr);
            }

            outputBufferOffset += devOutBlob->byteSize();
        }
    }

    const char* dumpOutputPathEnv = std::getenv("IE_VPU_KMB_DUMP_OUTPUT_PATH");
    if (dumpOutputPathEnv != nullptr) {
        dumpBlobs(_outputs, dumpOutputPathEnv, "output", _logger);
    }
    _logger->debug("InferRequest::GetResult finished");
}

void KmbInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>&) const {
    THROW_IE_EXCEPTION << "KmbInferRequest::GetPerformanceCounts is not implemented\n";
}

void KmbInferRequest::Infer() {
    KmbInferRequest::checkBlobs();
    InferImpl();
}

void KmbInferRequest::checkBlobs() {
    OV_ITT_SCOPED_TASK(itt::domains::KmbPlugin, "checkBlobs");
    for (auto const& output : _outputs) {
        checkBlob(output.second, output.first, false);
    }
}
