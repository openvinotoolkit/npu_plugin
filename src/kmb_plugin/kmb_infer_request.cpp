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
#include <ie_plugin.hpp>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/perf_report.hpp>

#include "ie_utils.hpp"
#include "kmb_executable_network.h"
#include "kmb_preproc.hpp"

// TODO [Track number: S#21391]
// FIXME: does not work for batch != 1
static bool is2DTensor(const InferenceEngine::SizeVector& dims) {
    size_t ones = std::count(dims.begin(), dims.end(), 1);
    return (dims.size() - ones) == 1;
}

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;

static void deallocateHelper(uint8_t* ptr) { getKmbAllocator()->free(ptr); }

KmbInferRequest::KmbInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
    const InferenceEngine::OutputsDataMap& networkOutputs, const std::vector<StageMetaInfo>& blobMetaData,
    const KmbConfig& kmbConfig, const KmbExecutor::Ptr& executor)
    : InferRequestInternal(networkInputs, networkOutputs),
      _executor(executor),
      _stagesMetaData(blobMetaData),
      _config(kmbConfig),
      _logger(std::make_shared<Logger>("KmbInferRequest", kmbConfig.logLevel(), consoleOutput())),
      _inputBuffer(nullptr, deallocateHelper),
      _outputBuffer(nullptr, deallocateHelper) {
    IE_PROFILING_AUTO_SCOPE(KmbInferRequest);
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

        Blob::Ptr inputBlob = make_blob_with_precision(networkInput.second->getTensorDesc(), getKmbAllocator());
        inputBlob->allocate();
        _inputs[networkInput.first] = inputBlob;
        inputsTotalSize += inputBlob->byteSize();
    }
    if (_networkInputs.size() > 1) {
        uint8_t* inputsRawPtr = reinterpret_cast<uint8_t*>(getKmbAllocator()->alloc(inputsTotalSize));
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
        outputBlob = make_blob_with_precision(networkOutput.second->getTensorDesc(), getKmbAllocator());
        outputBlob->allocate();
        _outputs[networkOutput.first] = outputBlob;
        outputsTotalSize += outputBlob->byteSize();
    }
    uint8_t* outputsRawPtr = reinterpret_cast<uint8_t*>(getKmbAllocator()->alloc(outputsTotalSize));
    _outputBuffer.reset(outputsRawPtr);
}
void KmbInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void KmbInferRequest::dumpOutputBlobHelper(const Blob::Ptr& outputBlobPtr, const std::string& dst) {
    static unsigned dumpOutputCounter = 0;
    std::ostringstream inputFullPath;
    inputFullPath << dst;
    inputFullPath << "/output-dump";
    inputFullPath << dumpOutputCounter++;
    inputFullPath << ".bin";
    _logger->info("dumpOutputBlobHelper: dump to file ", inputFullPath.str());
    std::ofstream dumper(inputFullPath.str(), std::ios_base::binary);
    if (dumper.good()) {
        dumper.write(outputBlobPtr->cbuffer().as<char*>(), outputBlobPtr->byteSize());
    } else {
        _logger->warning("dumpOutputBlobHelper: failed to open ", inputFullPath.str());
    }
    dumper.close();
}

void KmbInferRequest::InferAsync() {
    IE_PROFILING_AUTO_SCOPE(InferAsync);
    execPreprocessing(_inputs);

    if (std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH") != nullptr) {
        dumpInputs(_inputs, std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH"));
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

    const auto& deviceInputs = _executor->getRuntimeInputs();
    if (deviceInputs.begin() == deviceInputs.end()) THROW_IE_EXCEPTION << "DeviceInputs are empty.";
    if (deviceInputs.size() != _inputs.size()) THROW_IE_EXCEPTION << "DeviceInputs and _inputs sizes are different.";

    if (_inputs.size() == 1) {
        // avoid memory copy for single input
        const auto deviceInputDesc = deviceInputs.begin()->second->getTensorDesc();
        const auto input = _inputs.begin()->second;
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
            std::memcpy(_inputBuffer.get() + inputBufferOffset, updatedInput->buffer().as<uint8_t*>(),
                updatedInput->byteSize());

            inputBufferOffset += updatedInput->byteSize();
        }
        _executor->queueInference(_inputBuffer.get(), inputBufferOffset);
    }
}

void KmbInferRequest::execPreprocessing(InferenceEngine::BlobMap& inputs) {
    IE_PROFILING_AUTO_SCOPE(execPreprocessing);
    // TODO: [Track number: S#31121]
    // Get rid of environment variable USE_SIPP
    if ((_config.useSIPP() || SippPreproc::useSIPP()) &&
        SippPreproc::isApplicable(inputs, _preProcData, _networkInputs)) {
        relocationAndExecSIPPDataPreprocessing(
            inputs, _networkInputs, _config.outColorFmtSIPP(), _config.numberOfSIPPShaves(), _config.SIPPLpi());
    } else {
        _logger->warning("SIPP is enabled but configuration is not supported.");
        execDataPreprocessing(inputs);
    }
}

static bool isBlobPlacedInShareableMemory(const Blob::Ptr& blob) {
    return getKmbAllocator()->isValidPtr(blob->buffer().as<void*>());
}

// TODO: SIPP preprocessing usage can be merged to common preprocessing pipeline
void KmbInferRequest::relocationAndExecSIPPDataPreprocessing(InferenceEngine::BlobMap& inputs,
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
            if (!isBlobPlacedInShareableMemory(origYBlob)) {
                kmbYBlob = reallocateBlob(origYBlob);
            }
            Blob::Ptr kmbUVBlob = origUVBlob;
            if (!isBlobPlacedInShareableMemory(origUVBlob)) {
                kmbUVBlob = reallocateBlob(origUVBlob);
            }

            InferenceEngine::Blob::Ptr nv12Blob =
                InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(kmbYBlob, kmbUVBlob);
            preprocDataRealloc[preProcDataIter->first]->setRoiBlob(nv12Blob);
        } else {
            THROW_IE_EXCEPTION << "Attempt to pass non-NV12 image to SIPP preprocessing.";
        }
    }
    this->execSIPPDataPreprocessing(inputs, preprocDataRealloc, networkInputs, out_format, numShaves, lpi);
}

void KmbInferRequest::execSIPPDataPreprocessing(InferenceEngine::BlobMap& inputs,
    std::map<std::string, PreProcessDataPtr>& preprocData, InferenceEngine::InputsDataMap& networkInputs,
    InferenceEngine::ColorFormat out_format, unsigned int numShaves, unsigned int lpi) {
    SippPreproc::execSIPPDataPreprocessing(inputs, preprocData, networkInputs, out_format, numShaves, lpi);
}

static bool needRepacking(const Blob::Ptr& actualInput, const TensorDesc& deviceTensorDesc) {
    // TODO: is2DTensor is a workaround for NHWC -> NC case
    // remove when mcm will support different input layout
    return (deviceTensorDesc.getLayout() != actualInput->getTensorDesc().getLayout() &&
            !is2DTensor(actualInput->getTensorDesc().getDims()));
}

static Blob::Ptr reallocateBlobToLayoutIgnoringOriginalLayout(
    const Blob::Ptr& blob, const Layout srcLayout, const Layout dstLayout) {
    if (blob->getTensorDesc().getDims()[1] != 3) {
        THROW_IE_EXCEPTION << "reallocateBlobToLayoutIgnoringOriginalLayout works only with channels == 3";
    }

    // it would be nicer to construct srcTensorDesc from tensorDesc of blob
    // and then call srcTensorDesc.setLayout(srcLayout) but copyBlob does work in that case
    TensorDesc srcTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), srcLayout};
    Blob::Ptr srcBlob = make_blob_with_precision(srcTensorDesc, blob->buffer());

    TensorDesc dstTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), dstLayout};
    Blob::Ptr dstBlob = make_blob_with_precision(dstTensorDesc, getKmbAllocator());
    dstBlob->allocate();

    vpu::copyBlob(srcBlob, dstBlob);
    return dstBlob;
}

static Blob::Ptr reallocateBlobToLayout(const Blob::Ptr& blob, const Layout layout) {
    auto allocator = getKmbAllocator();

    TensorDesc dstTensorDesc = {blob->getTensorDesc().getPrecision(), blob->getTensorDesc().getDims(), layout};
    Blob::Ptr kmbBlob = make_blob_with_precision(dstTensorDesc, allocator);
    kmbBlob->allocate();

    vpu::copyBlob(blob, kmbBlob);

    return kmbBlob;
}

Blob::Ptr KmbInferRequest::reallocateBlob(const Blob::Ptr& blob) {
    IE_PROFILING_AUTO_SCOPE(reallocateBlob);
    return reallocateBlobToLayout(blob, blob->getTensorDesc().getLayout());
}

Blob::Ptr KmbInferRequest::prepareInputForInference(
    const ie::Blob::Ptr& actualInput, const InferenceEngine::TensorDesc& expectedDesc) {
    IE_PROFILING_AUTO_SCOPE(prepareInputForInference);
    // HACK: to overcome inability python API to pass a blob of NHWC layout
    if (_config.forceNCHWToNHWC()) {
        _logger->warning("VPU_KMB_FORCE_NCHW_TO_NHWC is enabled. Need to do re-layout.");
        return reallocateBlobToLayoutIgnoringOriginalLayout(actualInput, Layout::NCHW, Layout::NHWC);
    } else {
        Blob::Ptr inputForInference;
        if (!isBlobPlacedInShareableMemory(actualInput)) {
            _logger->warning("Input blob is located in non-shareable memory. Need to do re-allocation.");
            inputForInference = reallocateBlob(actualInput);
        } else {
            inputForInference = actualInput;
        }

        if (needRepacking(actualInput, expectedDesc)) {
            _logger->warning("Input blob is inconsistent with network input. Need to do re-layout.");
            inputForInference = reallocateBlobToLayout(actualInput, expectedDesc.getLayout());
        }

        return inputForInference;
    }
}

void KmbInferRequest::dumpInputs(const InferenceEngine::BlobMap& inputs, const std::string dstPath) const {
    if (dstPath.empty()) {
        _logger->warning(
            "Can not dump inputs since destination path is empty. Please check IE_VPU_KMB_DUMP_INPUT_PATH.");
        return;
    }
    for (const auto& input : inputs) {
        dumpInputBlobHelper(input.second, dstPath);
    }
}

void KmbInferRequest::dumpInputBlobHelper(const Blob::Ptr& inputBlobPtr, const std::string& dst) const {
    static unsigned dumpInputCounter = 0;
    std::ostringstream inputFullPath;
    inputFullPath << dst;
    inputFullPath << "/input-dump";
    inputFullPath << dumpInputCounter++;
    inputFullPath << ".bin";
    _logger->info("dumpInputBlobHelper: dump to file ", inputFullPath.str());
    std::ofstream dumper(inputFullPath.str(), std::ios_base::binary);
    if (dumper.good()) {
        dumper.write(inputBlobPtr->cbuffer().as<char*>(), inputBlobPtr->byteSize());
    } else {
        _logger->warning("dumpInputBlobHelper: failed to open ", inputFullPath.str());
    }
    dumper.close();
}

void KmbInferRequest::GetResult() {
    IE_PROFILING_AUTO_SCOPE(GetResult);
    auto dataName = _networkOutputs.begin()->first;

    auto foundInputBlob = _outputs.find(dataName);
    if (foundInputBlob == _outputs.end()) THROW_IE_EXCEPTION << "Error: output [" << dataName << "] is not provided.";

    // check that output layout is the same as device layout
    const InferenceEngine::OutputsDataMap& deviceOutputs = _executor->getRuntimeOutputs();

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
            Blob::Ptr blobWithCorrectPrecision = utils::convertPrecision(devOutBlob, inferOutputDesc.getPrecision());

            // copy blob with correct precision to the output blob
            // copyBlob does layout conversion on its own
            if (inferOutputDesc.getLayout() == InferenceEngine::Layout::NC) {
                // NC tensors are copied to blob buffer as is
                copyBlob(blobWithCorrectPrecision, deviceOutputDesc.getLayout(), outputBlob->buffer());
            } else {
                // do layout conversion
                copyBlob(blobWithCorrectPrecision, outputBlob);
            }

            outputBufferOffset += devOutBlob->byteSize();
        }
    }

    const char* dumpOutputPathEnv = std::getenv("IE_VPU_KMB_DUMP_OUTPUT_PATH");
    if (dumpOutputPathEnv != nullptr) {
        dumpOutputBlobHelper(outputBlobRef, dumpOutputPathEnv);
    }
}

void KmbInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>& perfMap) const {
    UNUSED(perfMap);
    THROW_IE_EXCEPTION << "KmbInferRequest::GetPerformanceCounts is not implemented\n";
}

void KmbInferRequest::Infer() {
    KmbInferRequest::checkBlobs();
    InferImpl();
}

void KmbInferRequest::checkBlobs() {
    IE_PROFILING_AUTO_SCOPE(checkBlobs);
    for (auto const& output : _outputs) {
        checkBlob(output.second, output.first, false);
    }
}
