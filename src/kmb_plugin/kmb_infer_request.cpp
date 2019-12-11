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

#include "kmb_executable_network.h"
#include "kmb_preproc.hpp"

// TODO https://jira.devtools.intel.com/browse/CVS-21391
/**
 * @brief multiply vector's values
 * @param vec - vector with values
 * @return result of multiplication
 */
template <typename T, typename A>
static T product(std::vector<T, A> const& vec) {
    if (vec.empty()) return 0;
    T ret = vec[0];
    for (size_t i = 1; i < vec.size(); ++i) ret *= vec[i];
    return ret;
}

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;

#define MEMCPY(dst, src, bytes) std::copy_n((src), (bytes), (dst))

KmbInferRequest::KmbInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
    const InferenceEngine::OutputsDataMap& networkOutputs, const std::vector<StageMetaInfo>& blobMetaData,
    const KmbConfig& kmbConfig, const KmbExecutorPtr& executor)
    : InferRequestInternal(networkInputs, networkOutputs),
      _executor(executor),
      _stagesMetaData(blobMetaData),
      _config(kmbConfig),
      _blobWithResult(nullptr),
      _logger(std::make_shared<Logger>("KmbInferRequest", kmbConfig.logLevel(), consoleOutput())) {
    _deviceLayout = InferenceEngine::Layout::NCHW;

    // allocate inputs
    IE_ASSERT(_networkInputs.size() == 1) << "Do not support more than 1 input";
    for (auto& networkInput : _networkInputs) {
        // TODO: use TensorDesc instead of deprecated methods
        SizeVector dims = networkInput.second->getTensorDesc().getDims();
        Precision precision = networkInput.second->getTensorDesc().getPrecision();
        Layout layout = networkInput.second->getTensorDesc().getLayout();

        if (precision != Precision::FP32 && precision != Precision::FP16 && precision != Precision::U8 &&
            precision != Precision::I8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input precision: " << precision
                               << "! Supported precisions: FP32, FP16, U8, I8";
        }

        Blob::Ptr inputBlob = make_blob_with_precision(TensorDesc(precision, dims, layout), getKmbAllocator());
        if (inputBlob == nullptr) {
            THROW_IE_EXCEPTION << "InputBlob is nullptr.";
        }

        inputBlob->allocate();
        _inputs[networkInput.first] = inputBlob;
    }
    // allocate outputs
    IE_ASSERT(_networkOutputs.size() == 1) << "Do not support more than 1 output";
    for (auto& networkOutput : _networkOutputs) {
        SizeVector dims = networkOutput.second->getTensorDesc().getDims();
        Precision precision = networkOutput.second->getTensorDesc().getPrecision();
        Layout layout = networkOutput.second->getTensorDesc().getLayout();

        if (precision != Precision::FP32 && precision != Precision::FP16 && precision != Precision::U8 &&
            precision != Precision::I8) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output precision: " << precision
                               << "! Supported precisions: FP32, FP16, U8, I8";
        }

        Blob::Ptr outputBlob = make_blob_with_precision(TensorDesc(precision, dims, layout), getKmbAllocator());

        if (outputBlob == nullptr) {
            THROW_IE_EXCEPTION << "InputBlob is nullptr.";
        }

        outputBlob->allocate();
        _outputs[networkOutput.first] = outputBlob;
    }

    if (_networkOutputs.empty() || _networkInputs.empty()) {
        THROW_IE_EXCEPTION << "Internal error: no information about network's output/input";
    }
}
void KmbInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void KmbInferRequest::dumpInputBlobHelper(const Blob::Ptr& inputBlobPtr, const std::string& dst) {
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

// TODO a lot of dublications
void KmbInferRequest::InferAsync() {
    if (!_custom_inputs.empty()) {
        // execute input pre-processing
        if (SippPreproc::useSIPP() && SippPreproc::isApplicable(_custom_inputs, _preProcData, _networkInputs)) {
            InferenceEngine::BlobMap inputs;
            for (auto& input : _custom_inputs) {
                inputs[input.first] =
                    make_blob_with_precision(_custom_inputs.begin()->second->getTensorDesc(), getKmbAllocator());
                inputs[input.first]->allocate();
            }

            SippPreproc::execSIPPDataPreprocessing(
                inputs, _preProcData, _networkInputs, 1, true, _config.numberOfSIPPShaves);

            for (auto& input : inputs) {
                auto name = input.first;

                auto custom_inputBlob = input.second;
                auto inputBlob = _inputs[name];

                copyBlob(custom_inputBlob, inputBlob);
            }
        } else {
            execDataPreprocessing(_custom_inputs);
            for (auto& input : _custom_inputs) {
                auto name = input.first;

                auto custom_inputBlob = input.second;
                auto inputBlob = _inputs[name];

                copyBlob(custom_inputBlob, inputBlob);
            }
        }
    } else {
        // execute input pre-processing
        if (SippPreproc::useSIPP() && SippPreproc::isApplicable(_inputs, _preProcData, _networkInputs)) {
            for (const auto& input : _inputs) {
                const std::string& inputName = input.first;
                auto preProcDataIter = _preProcData.find(inputName);
                if (preProcDataIter == _preProcData.end()) {
                    continue;
                }

                Blob::Ptr blobData = preProcDataIter->second->getRoiBlob();
                if (blobData->is<NV12Blob>()) {
                    // check if planes of nv12 blob were allocated with KMB allocator
                    NV12Blob::Ptr origNV12Blob = as<NV12Blob>(blobData);
                    Blob::Ptr& origYBlob = origNV12Blob->y();
                    Blob::Ptr& origUVBlob = origNV12Blob->uv();

                    if (!getKmbAllocator()->isValidPtr(origYBlob->buffer())) {
                        Blob::Ptr kmbYBlob = make_blob_with_precision(origYBlob->getTensorDesc(), getKmbAllocator());
                        IE_ASSERT(kmbYBlob != nullptr);

                        kmbYBlob->allocate();
                        copyBlob(origYBlob, kmbYBlob);
                        origNV12Blob->y() = kmbYBlob;
                    }

                    if (!getKmbAllocator()->isValidPtr(origUVBlob->buffer())) {
                        Blob::Ptr kmbUVBlob = make_blob_with_precision(origUVBlob->getTensorDesc(), getKmbAllocator());
                        IE_ASSERT(kmbUVBlob != nullptr);

                        kmbUVBlob->allocate();
                        copyBlob(origUVBlob, kmbUVBlob);
                        origNV12Blob->uv() = kmbUVBlob;
                    }
                }
            }

            SippPreproc::execSIPPDataPreprocessing(
                _inputs, _preProcData, _networkInputs, 1, true, _config.numberOfSIPPShaves);
        } else {
            execDataPreprocessing(_inputs);
        }
    }

    std::string dumpInputPath = "";
    const char* dumpInputPathEnv = std::getenv("IE_VPU_KMB_DUMP_INPUT_PATH");
    if (dumpInputPathEnv != nullptr) {
        dumpInputPath = dumpInputPathEnv;
    }

    for (const auto& input : _inputs) {
        auto const inputBlobPtr = input.second;
        auto inputBlobPrecision = inputBlobPtr->getTensorDesc().getPrecision();
        if (inputBlobPrecision != Precision::FP16 && inputBlobPrecision != Precision::FP32 &&
            inputBlobPrecision != Precision::I8 && inputBlobPrecision != Precision::U8)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported input blob precision";

        if (!dumpInputPath.empty()) {
            dumpInputBlobHelper(inputBlobPtr, dumpInputPath);
        }
    }
    for (const auto& output : _outputs) {
        auto const outputBlobPtr = output.second;
        auto outputBlobPrecision = outputBlobPtr->getTensorDesc().getPrecision();
        if (outputBlobPrecision != Precision::FP16 && outputBlobPrecision != Precision::FP32 &&
            outputBlobPrecision != Precision::I8 && outputBlobPrecision != Precision::U8)
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported output blob precision";
    }

    auto dataName = _networkInputs.begin()->first;

    auto foundInputBlob = _inputs.find(dataName);
    if (foundInputBlob == _inputs.end()) THROW_IE_EXCEPTION << "Error: input [" << dataName << "] is not provided.";

    // check that input layout is the same as device layout
    const InferenceEngine::InputsDataMap& deviceInputs = _executor->getNetworkInputs();
    IE_ASSERT(deviceInputs.size() == 1) << "Networks with " << deviceInputs.size() << " inputs are not supported. "
                                        << "Only networks with 1 input are supported.";

    size_t input_size_total =
        std::accumulate(_inputs.begin(), _inputs.end(), 0, [](size_t sum, InferenceEngine::BlobMap::value_type& input) {
            return sum + input.second->byteSize();
        });

    Blob::Ptr& inputBlobRef = _inputs.begin()->second;
    InferenceEngine::TensorDesc deviceTensorDesc = deviceInputs.begin()->second->getTensorDesc();
    if (deviceTensorDesc.getLayout() != inputBlobRef->getTensorDesc().getLayout()) {
        // do layout conversion with copyBlob
        InferenceEngine::Blob::Ptr blobWithInput = make_blob_with_precision(deviceTensorDesc, getKmbAllocator());
        blobWithInput->allocate();
        copyBlob(inputBlobRef, blobWithInput);
        void* inputPtr = blobWithInput->buffer();
        _executor->queueInference(inputPtr, input_size_total, nullptr, 0);
    } else {
        // pass input as is, do not convert layout
        void* inputPtr = inputBlobRef->buffer();
        _executor->queueInference(inputPtr, input_size_total, nullptr, 0);
    }
}

static Blob::Ptr convertPrecision(
    const InferenceEngine::Blob::Ptr& sourceData, const InferenceEngine::TensorDesc& targetDesc) {
    InferenceEngine::TensorDesc sourceTensorDesc = sourceData->getTensorDesc();
    InferenceEngine::Precision targetPrecision = targetDesc.getPrecision();
    InferenceEngine::Precision sourcePrecision = sourceTensorDesc.getPrecision();
    if (sourcePrecision == targetPrecision) {
        return sourceData;
    }

    Blob::Ptr target =
        make_blob_with_precision(TensorDesc(targetPrecision, sourceTensorDesc.getDims(), sourceTensorDesc.getLayout()));
    target->allocate();
    if (sourcePrecision == InferenceEngine::Precision::FP16 && targetPrecision == InferenceEngine::Precision::FP32) {
        InferenceEngine::PrecisionUtils::f16tof32Arrays(
            target->buffer(), sourceData->cbuffer().as<ie_fp16*>(), sourceData->size(), 1.0f, 0.0f);
    } else if (sourcePrecision == InferenceEngine::Precision::FP32 &&
               targetPrecision == InferenceEngine::Precision::FP16) {
        InferenceEngine::PrecisionUtils::f32tof16Arrays(
            target->buffer(), sourceData->cbuffer().as<float*>(), sourceData->size());
    } else {
        THROW_IE_EXCEPTION << "Error: output precision conversion from " << sourcePrecision << " to " << targetPrecision
                           << " is not supported.";
    }
    return target;
}

void KmbInferRequest::GetResult() {
    auto dataName = _networkOutputs.begin()->first;

    auto foundInputBlob = _outputs.find(dataName);
    if (foundInputBlob == _outputs.end()) THROW_IE_EXCEPTION << "Error: output [" << dataName << "] is not provided.";

    // check that output layout is the same as device layout
    const InferenceEngine::OutputsDataMap& deviceOutputs = _executor->getNetworkOutputs();
    IE_ASSERT(deviceOutputs.size() == 1) << "Networks with " << deviceOutputs.size() << " outputs are not supported. "
                                         << "Only networks with 1 output are supported.";

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
    if (devicePrecision == outputPrecision && deviceLayout == outputLayout) {
        // read result directly into output, do not copy blob
        void* outputPtr = outputBlobRef->buffer();
        _executor->getResult(outputPtr, output_size_total);
    } else {
        // read result into _blobWithResult
        if (_blobWithResult == nullptr) {
            _blobWithResult = make_blob_with_precision(deviceTensorDesc);
            _blobWithResult->allocate();
        }
        void* outputPtr = _blobWithResult->buffer();
        _executor->getResult(outputPtr, output_size_total);
        // do precision conversion when necessary
        Blob::Ptr blobWithCorrectPrecision = convertPrecision(_blobWithResult, outputTensorDesc);
        // copy blob with correct precision to the output blob
        // copyBlob does layout conversion on its own
        copyBlob(blobWithCorrectPrecision, _outputs.begin()->second);
    }

    if (!_custom_outputs.empty()) {
        for (auto& output : _outputs) {
            auto name = output.first;

            auto custom_outputBlob = _custom_outputs[name];
            auto outputBlob = output.second;

            copyBlob(outputBlob, custom_outputBlob);
        }
    }
}

void KmbInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>& perfMap) const {
    UNUSED(perfMap);
    THROW_IE_EXCEPTION << "KmbInferRequest::GetPerformanceCounts is not implemented\n";
}

void KmbInferRequest::SetBlob(const char* name, const InferenceEngine::Blob::Ptr& data) {
    IE_PROFILING_AUTO_SCOPE(SetBlob)
    if (name == nullptr) {
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }
    if (!data) THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
    const bool compoundBlobPassed = data->is<CompoundBlob>();
    if (!compoundBlobPassed && data->buffer() == nullptr)
        THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
    if (data->size() == 0) {
        THROW_IE_EXCEPTION << "Input data is empty. Input name: \'" << name << "\'";
    }

    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    size_t dataSize = data->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        if (foundInput->getPrecision() != data->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding to user input precision";
        }

        const bool preProcRequired = preProcessingRequired(foundInput, data);
        if (compoundBlobPassed && !preProcRequired) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }

        if (preProcRequired) {
            // Stores the given blob as ROI blob. It will be used to fill in network input
            // during pre-processing
            Blob::Ptr orig_data;
            auto found = _inputs.find(name);
            if (found != _inputs.end()) {
                orig_data = found->second;
            } else {
                orig_data = _custom_inputs[name];
            }
            _preProcData[name] = CreatePreprocDataHelper();
            _preProcData[name]->isApplicable(data, orig_data);

            _preProcData[name]->setRoiBlob(data);
        } else {
            size_t inputSize = product(foundInput->getTensorDesc().getDims());
            if (dataSize != inputSize) {
                THROW_IE_EXCEPTION << "Input blob size is not equal network input size (" << dataSize
                                   << "!=" << inputSize << ").";
            }
            if (getKmbAllocator()->isValidPtr(data->buffer())) {
                _inputs[name] = data;
            } else {
                _logger->info("isValidPtr(): Input blob will be copied");
                _custom_inputs[name] = data;
            }
        }
    } else {
        if (compoundBlobPassed) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }
        size_t outputSize = product(foundOutput->getDims());
        if (dataSize != outputSize) {
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size (" << dataSize
                               << "!=" << outputSize << ").";
        }
        if (foundOutput->getPrecision() != data->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding to user output precision";
        }
        if (getKmbAllocator()->isValidPtr(data->buffer())) {
            _inputs[name] = data;
        } else {
            _logger->info("isValidPtr(): Input blob will be copied");
            _custom_inputs[name] = data;
        }
    }
}

void KmbInferRequest::GetBlob(const char* name, Blob::Ptr& data) {
    IE_PROFILING_AUTO_SCOPE(GetBlob)
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        // ROI blob is returned only if it was set previously. Otherwise default blob is returned.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
            data = it->second->getRoiBlob();
        } else {
            if (!_custom_inputs.empty()) {
                data = _custom_inputs[name];
            } else {
                data = _inputs[name];
            }
            checkBlob(data, name, true, foundInput->getTensorDesc().getDims());
        }
    } else {
        if (!_custom_outputs.empty()) {
            data = _custom_outputs[name];
        } else {
            data = _outputs[name];
        }
        checkBlob(data, name, false, foundOutput->getTensorDesc().getDims());
    }
}

void KmbInferRequest::Infer() {
    KmbInferRequest::checkBlobs();
    InferImpl();
}

void KmbInferRequest::checkBlobs() {
    if (_custom_inputs.empty()) {
        for (auto const& input : _inputs) {
            checkBlob(input.second, input.first, true);
        }
    } else {
        for (auto const& input : _custom_inputs) {
            checkBlob(input.second, input.first, true);
        }
    }

    if (_custom_outputs.empty()) {
        for (auto const& output : _outputs) {
            checkBlob(output.second, output.first, false);
        }
    } else {
        for (auto const& output : _custom_outputs) {
            checkBlob(output.second, output.first, false);
        }
    }
}
