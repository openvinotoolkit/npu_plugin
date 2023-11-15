//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <debug.h>
#include <ie_blob.h>
#include <blob_factory.hpp>

#include <vpux_variable_state.hpp>
#include "zero_infer_request.h"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"

#include "vpux/utils/IE/data_attributes_check.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/IE/prefix.hpp"

namespace ie = InferenceEngine;
using namespace vpux;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
namespace {
static void checkNetworkPrecision(const ie::Precision& precision) {
    switch (precision) {
    case ie::Precision::FP32:
        break;
    case ie::Precision::FP16:
        break;
    case ie::Precision::U8:
        break;
    case ie::Precision::I8:
        break;
    case ie::Precision::U32:
        break;
    case ie::Precision::I32:
        break;
    case ie::Precision::U16:
        break;
    case ie::Precision::I16:
        break;
    default:
        IE_THROW(ParameterMismatch) << "Unsupported input precision: " << precision
                                    << "! Supported precisions: FP32, FP16, U8, I8, U32, I32, U16, I16";
    }
}

static ie::Blob::Ptr allocateLocalBlob(const ie::TensorDesc& tensorDesc, void* dataPtr) {
    checkNetworkPrecision(tensorDesc.getPrecision());

    ie::Blob::Ptr blob = make_blob_with_precision(tensorDesc, dataPtr);
    if (blob == nullptr) {
        IE_THROW() << "Can't make blob.";
    }
    return blob;
}

// check that ie Layout and zeroApi layout are the same for some argument
bool twoApiLayoutCouplingCheck(const ze_graph_argument_layout_t zeroL, const ie::Layout ieL) {
    using namespace ::InferenceEngine;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_ANY == zeroL && ANY == ieL)
        return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_NCHW == zeroL && NCHW == ieL)
        return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_NHWC == zeroL && (NHWC == ieL || NC == ieL || C == ieL))
        return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW == zeroL && NCDHW == ieL)
        return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC == zeroL && NDHWC == ieL)
        return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_OIHW == zeroL && OIHW == ieL)
        return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_C == zeroL && C == ieL)
        return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_CHW == zeroL && CHW == ieL)
        return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_HW == zeroL && HW == ieL)
        return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_NC == zeroL && NC == ieL)
        return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_CN == zeroL && CN == ieL)
        return true;
    return false;
}

template <typename T>
std::size_t getNumDims(const T& dims) {
    return std::count_if(std::begin(dims), std::end(dims), [](const std::size_t& dim) -> bool {
        return (dim > 1);
    });
}

}  // namespace

//------------------------------------------------------------------------------
ZeroInferRequest::ZeroInferRequest(const ie::InputsDataMap& networkInputs, const ie::OutputsDataMap& networkOutputs,
                                   const Executor::Ptr& executor, const Config& config, const std::string& /*netName*/,
                                   const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                   const std::vector<std::shared_ptr<const ov::Node>>& results,
                                   const vpux::NetworkIOVector& networkStatesInfo,
                                   const std::shared_ptr<ie::IAllocator>& allocator)
        : IInferRequest(networkInputs, networkOutputs),
          _executorPtr(executor),
          _executor(static_cast<ZeroExecutor*>(_executorPtr.get())),
          _config(config),
          _logger("ZeroInferRequest", config.get<LOG_LEVEL>()),
          _allocator(allocator),
          _statesInfo(networkStatesInfo),
          _profiling_pool(_executor->graph(), zeroProfiling::POOL_SIZE, _executor->graph_profiling_ddi_table_ext()),
          _profiling_query(0, _executor->device(), _executor->graph_profiling_ddi_table_ext()),
          _pipeline(makePipeline(_executorPtr, _config, _profiling_pool, _profiling_query)) {
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        IE_THROW() << "No information about network's output/input.";
    }
    _parameters = parameters;
    _results = results;

    const auto& deviceInputs = _executor->getNetworkDesc().getDeviceInputsInfo();
    for (const auto& deviceInput : deviceInputs) {
        const std::string& inputName = deviceInput.first;
        const auto& networkInputMatch = _networkInputs.find(inputName);
        if (networkInputMatch == _networkInputs.end()) {
            IE_THROW() << "Network input not found: " + inputName;
        }

        const ie::TensorDesc inputTensorDesc = networkInputMatch->second->getTensorDesc();
        _inputs[inputName] = allocateLocalBlob(inputTensorDesc, _pipeline->inputs().getHostPtr(inputName));
    }

    const auto& deviceOutputs = _executor->getNetworkDesc().getDeviceOutputsInfo();
    for (const auto& deviceOutput : deviceOutputs) {
        const std::string& outputName = deviceOutput.first;
        const auto& networkOutputMatch = _networkOutputs.find(outputName);
        if (networkOutputMatch == _networkOutputs.end()) {
            IE_THROW() << "Network output not found: " + outputName;
        }

        const ie::TensorDesc outputTensorDesc = networkOutputMatch->second->getTensorDesc();
        _outputs[outputName] = allocateLocalBlob(outputTensorDesc, _pipeline->outputs().getHostPtr(outputName));
    }
}

void ZeroInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void ZeroInferRequest::InferAsync() {
    _logger.debug("InferRequest::InferAsync started");
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "InferAsync");

    execDataPreprocessing(_inputs);
    const auto& deviceInputs = _executor->getNetworkDesc().getDeviceInputsInfo();
    const std::map<std::string, ZeroExecutor::ArgumentDescriptor>& executorInputsDescriptors =
            _executor->inputs_desc_map();

    // Copy input data to staging buffer on Cpu (input always first argument)
    for (const auto& deviceInput : deviceInputs) {
        const auto& name = deviceInput.first;
        const auto& data = deviceInput.second;
        const auto& input = _inputs.at(name);

        if (!executorInputsDescriptors.count(name)) {
            IE_THROW() << "Invalid graph input descriptor key: " + name;
        }

        const ZeroExecutor::ArgumentDescriptor& desc = executorInputsDescriptors.at(name);

        // TODO Currently L0 and Plugin might return different layouts which have dims like [1,1...]
        // They might be reinterpreted in different ways, so this check has been added to prevent that behavior
        if (std::max(getNumDims(desc.info.dims), getNumDims(data->getTensorDesc().getDims())) > 2) {
            if (!twoApiLayoutCouplingCheck(desc.info.deviceLayout, data->getLayout())) {
                IE_THROW() << "Parsing error: layouts are different for push blobs";
            }
        }
        if (desc.info.devicePrecision != zeroUtils::getZePrecision(data->getPrecision())) {
            IE_THROW() << "Parsing error: precisions are different for push blobs";
        }
        checkDataAttributesMatch(input->getTensorDesc(), data->getTensorDesc());

        // we should check memory type: host memory or generic and copy if it's a generic
        const auto memInput = ie::as<ie::MemoryBlob>(input);
        VPUX_THROW_UNLESS(memInput != nullptr, "Input ie::Blob::Ptr cannot be cast to ie::MemoryBlob::Ptr");
        const auto inputMemLock = memInput->rmap();
        const uint8_t* inputPtr = inputMemLock.as<const uint8_t*>();
        if (!_pipeline->inputs().checkHostPtr(inputPtr)) {
            void* hostMem = _pipeline->inputs().getHostPtr(name);
            if (nullptr == hostMem || nullptr == inputPtr) {
                IE_THROW() << "Memory or input blob null pointer";
            }
            if (0 != ie_memcpy(hostMem, input->byteSize(), inputPtr, input->byteSize())) {
                IE_THROW() << "memcpy error for push blob " << name;
            }
        }
    }

    _pipeline->push();
}

std::vector<std::shared_ptr<ie::IVariableStateInternal>> ZeroInferRequest::QueryState() {
    // TODO: Check that std::call_once is not redudant here
    std::call_once(_fillStatesOnceFlag, [&]() {
        for (auto& stateInfo : _statesInfo) {
            const auto readValueName = READVALUE_PREFIX + stateInfo.first;

            IE_ASSERT(1 == _networkInputs.count(readValueName));
            IE_ASSERT(1 == _networkOutputs.count(ASSIGN_PREFIX + stateInfo.first));

            _states.push_back(std::make_shared<VariableState>(stateInfo.first, this->GetBlob(readValueName)));
        }
    });

    return _states;
}

void ZeroInferRequest::GetResult() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "GetResult");
    const auto& deviceOutputs = _executor->getNetworkDesc().getDeviceOutputsInfo();

    _pipeline->pull();
    const std::map<std::string, ZeroExecutor::ArgumentDescriptor>& executorOutputsDescriptors =
            _executor->outputs_desc_map();

    // Copy output data to staging buffer on Cpu (input always first argument)
    for (auto& deviceOutput : deviceOutputs) {
        const auto& name = deviceOutput.first;
        const auto& data = deviceOutput.second;
        const auto& output = _outputs.at(name);

        if (!executorOutputsDescriptors.count(name)) {
            IE_THROW() << "Invalid graph output descriptor key: " + name;
        }

        const ZeroExecutor::ArgumentDescriptor& desc = executorOutputsDescriptors.at(name);
        if (std::max(getNumDims(desc.info.dims), getNumDims(data->getTensorDesc().getDims())) > 2) {
            if (!twoApiLayoutCouplingCheck(desc.info.deviceLayout, data->getLayout())) {
                IE_THROW() << "Parsing error: layouts are different for pull blobs";
            }
        }
        if (desc.info.devicePrecision != zeroUtils::getZePrecision(data->getPrecision())) {
            IE_THROW() << "Parsing error: precisions are different for pull blobs";
        }
        checkDataAttributesMatch(output->getTensorDesc(), data->getTensorDesc());

        // we should check memory type: host memory or generic and copy if it's a generic
        const auto memOutput = ie::as<ie::MemoryBlob>(output);
        VPUX_THROW_UNLESS(memOutput != nullptr, "Output ie::Blob::Ptr cannot be cast to ie::MemoryBlob::Ptr");
        auto outputMemLock = memOutput->wmap();
        uint8_t* outputPtr = outputMemLock.as<uint8_t*>();
        if (!_pipeline->outputs().checkHostPtr(outputPtr)) {
            const void* hostMem = _pipeline->outputs().getHostPtr(name);
            if (nullptr == hostMem || nullptr == outputPtr) {
                IE_THROW() << "Memory or output blob null pointer";
            }
            if (0 != ie_memcpy(outputPtr, output->byteSize(), hostMem, output->byteSize())) {
                IE_THROW() << "memcpy error for pull blob " << name;
            }
        }
    }

    _pipeline->reset();
    _logger.debug("InferRequest::GetResult finished");
}

std::map<std::string, ie::InferenceEngineProfileInfo> ZeroInferRequest::GetPerformanceCounts() const {
    if (_config.get<PERF_COUNT>()) {
        return const_cast<ZeroInferRequest*>(this)->_profiling_query.getLayerStatistics(
                _config.get<COMPILER_TYPE>(), _executor->getNetworkDesc().getCompiledNetwork());
    } else {
        return {};
    }
}
