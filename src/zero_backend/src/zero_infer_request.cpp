//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <debug.h>

#include "zero_infer_request.h"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/IE/prefix.hpp"

using namespace vpux;

namespace {

constexpr bool STATE_TENSOR = true;
constexpr bool NOT_STATE_TENSOR = false;

/**
 * @brief Checks that the metadata of the provided descriptor corresponds to the values registered in the Level Zero
 * structure.
 * @param nodeDescriptor The OpenVINO API specific I/O descriptor which shall be compared.
 * @param zeDescriptor The Level Zero specific structure used for comparison.
 * @param name Tensor identifier used for error logging.
 */
void check_level_zero_attributes_match(const IONodeDescriptor& nodeDescriptor,
                                       const ZeroExecutor::ArgumentDescriptor& zeDescriptor, const std::string& name) {
    const ov::element::Type_t ovPrecision = nodeDescriptor.precision;
    const ze_graph_argument_precision_t zePrecision = zeDescriptor.info.devicePrecision;

    if (zeroUtils::getZePrecision(ovPrecision) != zePrecision) {
        OPENVINO_THROW("Precision mismatch for parameter " + name);
    }

    const std::vector<size_t>& ovDimensions = nodeDescriptor.originalShape.get_shape();

    if (ovDimensions.size() > ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) {
        OPENVINO_THROW(
                "Maximum number of dimensions supported: " + std::to_string(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) +
                '\n' + "Given: " + std::to_string(ovDimensions.size()));
    }

    for (size_t index = 0; index < ovDimensions.size(); ++index) {
        if (ovDimensions[index] != zeDescriptor.info.dims[index]) {
            OPENVINO_THROW("Shape mismatch for parameter " + name);
        }
    }
    for (size_t index = ovDimensions.size(); index < ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE; ++index) {
        if (zeDescriptor.info.dims[index] != 0 && zeDescriptor.info.dims[index] != 1) {
            OPENVINO_THROW("Shape mismatch for parameter " + name);
        }
    }
}

}  // namespace

//------------------------------------------------------------------------------
ZeroInferRequest::ZeroInferRequest(const std::shared_ptr<const ov::ICompiledModel> compiledModel,
                                   const std::shared_ptr<const NetworkDescription> networkDescription,
                                   const Executor::Ptr executor, const Config& config)
        : SyncInferRequest(compiledModel, networkDescription),
          _executorPtr(executor),
          _executor(static_cast<ZeroExecutor*>(_executorPtr.get())),
          _config(config),
          _logger("ZeroInferRequest", config.get<LOG_LEVEL>()),
          _profiling_pool(_executor->graph(), zeroProfiling::POOL_SIZE, _executor->graph_profiling_ddi_table_ext()),
          _profiling_query(0, _executor->device(), _executor->graph_profiling_ddi_table_ext()) {
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorInputDescriptors =
            _executor->inputs_desc_map();
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorOutputDescriptors =
            _executor->outputs_desc_map();

    auto proftype = config.get<PROFILING_TYPE>();
    if (proftype == InferenceEngine::VPUXConfigParams::ProfilingType::INFER) {
        _vpu_profiling = std::make_shared<vpux::zeroProfiling::VpuInferProfiling>(
                _executor->context(), _executor->device(), _config.get<LOG_LEVEL>());
    }

    /// Construct pipepline
    _pipeline = makePipeline(_executorPtr, _config, _profiling_pool, _profiling_query, _vpu_profiling);

    for (const std::string& inputName : _inputNames) {
        if (!executorInputDescriptors.count(inputName)) {
            OPENVINO_THROW("Invalid graph input descriptor key: " + inputName);
        }

        const IONodeDescriptor& parameterDescriptor = _parameterDescriptors.at(inputName);
        check_level_zero_attributes_match(parameterDescriptor, executorInputDescriptors.at(inputName), inputName);

        // The I/O buffers already allocated using the Level Zero API are being reused here
        allocate_tensor(inputName, parameterDescriptor, _pipeline->inputs().getHostPtr(inputName), NOT_STATE_TENSOR);
    }

    for (const std::string& outputName : _outputNames) {
        if (!executorOutputDescriptors.count(outputName)) {
            OPENVINO_THROW("Invalid graph output descriptor key: " + outputName);
        }

        const IONodeDescriptor& resultDescriptor = _resultDescriptors.at(outputName);
        check_level_zero_attributes_match(resultDescriptor, executorOutputDescriptors.at(outputName), outputName);
        allocate_tensor(outputName, resultDescriptor, _pipeline->outputs().getHostPtr(outputName), NOT_STATE_TENSOR);
    }

    for (const std::string& stateName : _stateNames) {
        const std::string& stateInputBufferName = READVALUE_PREFIX + stateName;
        const std::string& stateOutputBufferName = ASSIGN_PREFIX + stateName;

        if (!executorInputDescriptors.count(stateInputBufferName)) {
            OPENVINO_THROW("Invalid graph input descriptor key: " + stateInputBufferName);
        }
        if (!executorOutputDescriptors.count(stateOutputBufferName)) {
            OPENVINO_THROW("Invalid graph output descriptor key: " + stateOutputBufferName);
        }

        const IONodeDescriptor& stateDescriptor = _stateDescriptors.at(stateName);
        check_level_zero_attributes_match(stateDescriptor, executorInputDescriptors.at(stateInputBufferName),
                                          stateInputBufferName);
        check_level_zero_attributes_match(stateDescriptor, executorOutputDescriptors.at(stateOutputBufferName),
                                          stateOutputBufferName);

        // Only one buffer per state variable is required, we'll use the "output" one since this one captures the latest
        // tensor value
        allocate_tensor(stateName, stateDescriptor, _pipeline->outputs().getHostPtr(stateOutputBufferName),
                        STATE_TENSOR);
    }
}

void ZeroInferRequest::infer() {
    infer_async();
    get_result();
}

void ZeroInferRequest::infer_async() {
    _logger.debug("InferRequest::infer_async started");
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "infer_async");

    for (const auto& name : _inputAndStateInputNames) {
        const std::shared_ptr<ov::ITensor>& inputTensor = _allTensors.at(name);

        // If the memory address correponding to the data buffer does not correspond to the address expected by the
        // driver then we need to perform an extra buffer copy operation
        const uint8_t* tensorBuffer = reinterpret_cast<uint8_t*>(inputTensor->data());
        if (!_pipeline->inputs().checkHostPtr(tensorBuffer)) {
            void* zeBuffer;

            if (!isStateOutputName(name)) {
                zeBuffer = _pipeline->inputs().getHostPtr(name);
            } else {
                // Input and output buffers have been allocated for each state variable using the Level Zero API. The
                // input buffers are identified by a "read value" prefix, while the output buffers and the tensors found
                // within the OpenVINO specific structures use the "assign" prefix.
                zeBuffer = _pipeline->inputs().getHostPtr(stateOutputToStateInputName(name));
            }

            if (zeBuffer == nullptr || tensorBuffer == nullptr) {
                OPENVINO_THROW("Empty buffer");
            }
            std::memcpy(zeBuffer, tensorBuffer, inputTensor->get_byte_size());
        }
    }

    _pipeline->push();
}

void ZeroInferRequest::get_result() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "get_result");

    _pipeline->pull();

    for (const auto& name : _outputAndStateOutputNames) {
        const std::shared_ptr<ov::ITensor>& outputTensor = _allTensors.at(name);

        // If the memory address correponding to the data buffer does not correspond to the address expected by the
        // driver then we need to perform an extra buffer copy operation
        uint8_t* tensorBuffer = reinterpret_cast<uint8_t*>(outputTensor->data());
        if (!_pipeline->outputs().checkHostPtr(tensorBuffer)) {
            void* zeBuffer = _pipeline->outputs().getHostPtr(name);

            if (zeBuffer == nullptr || tensorBuffer == nullptr) {
                OPENVINO_THROW("Empty buffer");
            }
            std::memcpy(tensorBuffer, zeBuffer, outputTensor->get_byte_size());
        }
    }

    _pipeline->reset();
    _logger.debug("InferRequest::get_result finished");
}

void ZeroInferRequest::check_network_precision(const ov::element::Type_t precision) {
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
    default:
        OPENVINO_THROW("Unsupported tensor precision: " + ov::element::Type(precision).get_type_name() +
                       "! Supported precisions: FP32, FP16, U8, I8, U16, I16, U32, I32, U64, I64");
    }
}

std::vector<ov::ProfilingInfo> ZeroInferRequest::get_profiling_info() const {
    if (_config.get<PERF_COUNT>()) {
        auto proftype = _config.get<PROFILING_TYPE>();
        if (proftype == InferenceEngine::VPUXConfigParams::ProfilingType::INFER) {
            return _vpu_profiling->getVpuInferStatistics();
        } else {  /// proftype = MODEL or undefined = fallback to model profiling
            return const_cast<ZeroInferRequest*>(this)->_profiling_query.getLayerStatistics(
                    _config.get<COMPILER_TYPE>(), _executor->getNetworkDesc().getCompiledNetwork());
        }
    } else {
        return {};
    }
}
