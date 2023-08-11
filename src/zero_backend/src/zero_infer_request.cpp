//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <debug.h>
#include <ie_blob.h>
#include <blob_factory.hpp>

#include <vpux_variable_state.hpp>
#include "ze_api.h"
#include "zero_executor.h"
#include "zero_infer_request.h"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"

#include "vpux/utils/IE/blob.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/IE/prefix.hpp"

namespace IE = InferenceEngine;
using namespace vpux;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
namespace {
static void checkNetworkPrecision(const IE::Precision& precision) {
    if (precision != IE::Precision::FP32 && precision != IE::Precision::FP16 && precision != IE::Precision::U8 &&
        precision != IE::Precision::I8 && precision != IE::Precision::I32 && precision != IE::Precision::U32) {
        IE_THROW(ParameterMismatch) << "Unsupported input precision: " << precision
                                    << "! Supported precisions: FP32, FP16, U8, I8, I32, U32";
    }
}

static IE::Blob::Ptr allocateLocalBlob(const IE::TensorDesc& tensorDesc, void* dataPtr) {
    checkNetworkPrecision(tensorDesc.getPrecision());

    IE::Blob::Ptr blob;
    if (dataPtr) {
        blob = make_blob_with_precision(tensorDesc, dataPtr);
    } else {
        blob = make_blob_with_precision(tensorDesc);
        if (blob) {
            blob->allocate();
        }
    }
    if (nullptr == blob) {
        IE_THROW() << "Can't make blob.";
    }
    return blob;
}

// check that ie Layout and zeroApi layout are the same for some argument
bool twoApiLayoutCouplingCheck(const ze_graph_argument_layout_t zeroL, const IE::Layout ieL) {
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

bool isRepackingPossible(const IE::TensorDesc& userTensorDesc, const IE::TensorDesc& deviceTensorDesc) {
    const auto userPrecision = userTensorDesc.getPrecision();
    const auto devicePrecision = deviceTensorDesc.getPrecision();
    const auto userLayout = userTensorDesc.getLayout();
    const auto deviceLayout = deviceTensorDesc.getLayout();
    std::vector<IE::Layout> layouts{userLayout, deviceLayout};
    const auto unsupportedLayout = std::find_if(layouts.cbegin(), layouts.cend(), [](const IE::Layout& layout) -> bool {
        switch (layout) {
        case IE::Layout::ANY:
        case IE::Layout::OIHW:
        case IE::Layout::GOIHW:
        case IE::Layout::OIDHW:
        case IE::Layout::GOIDHW:
        case IE::Layout::BLOCKED:
            return true;
        default:
            break;
        }
        return false;
    });
    if (unsupportedLayout != layouts.end()) {
        return false;
    }

    // Layouts are OK for repacking, checking precisions
    std::vector<IE::Precision> precisions{userPrecision, devicePrecision};
    const auto unsupportedPrecision =
            std::find_if(precisions.cbegin(), precisions.cend(), [](const IE::Precision& precision) -> bool {
                switch (precision) {
                case IE::Precision::UNSPECIFIED:
                case IE::Precision::MIXED:
                case IE::Precision::BF16:
                case IE::Precision::FP64:
                case IE::Precision::Q78:
                case IE::Precision::U4:
                case IE::Precision::I4:
                case IE::Precision::BIN:
                case IE::Precision::BOOL:
                case IE::Precision::CUSTOM:
                    return true;
                default:
                    break;
                }
                return false;
            });
    if (unsupportedPrecision != precisions.end()) {
        return false;
    }

    return true;
}

void prepareInputForInference(const IE::Blob::Ptr& userInput, const IE::TensorDesc& deviceTensorDesc, void* destData,
                              const vpux::Optional<QuantizationParam>& quantParam, Logger logger) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Infer_request::prepareInputForInference");
    if (userInput == nullptr) {
        IE_THROW() << "User input blob null pointer";
    }
    if (destData == nullptr) {
        IE_THROW() << "Destination data null pointer";
    }
    const auto userPrecision = userInput->getTensorDesc().getPrecision();
    const auto userLayout = userInput->getTensorDesc().getLayout();
    const auto devicePrecision = deviceTensorDesc.getPrecision();
    const auto deviceLayout = deviceTensorDesc.getLayout();

    const bool isPrecisionMatched = userPrecision == devicePrecision;
    const bool isLayoutMatched = userLayout == deviceLayout;
    if (isPrecisionMatched && isLayoutMatched) {
        IE_THROW() << "There is nothing to repack";
    }

    if (!isPrecisionMatched) {
        logger.info("Different precisions of user and device input blobs.\tConversion required from {0} to {1}",
                    userPrecision.name(), devicePrecision.name());
        if (!isLayoutMatched) {
            IE::Blob::Ptr expectedInput = toPrecision(IE::as<IE::MemoryBlob>(userInput), devicePrecision, quantParam);
            std::stringstream conversionDetailsStr;
            conversionDetailsStr << "Conversion required from " << userLayout << " to " << deviceLayout << ".";
            logger.info("Different layouts of user and device input blobs.\t{0}", conversionDetailsStr.str());
            toLayout(IE::as<IE::MemoryBlob>(expectedInput), deviceLayout, nullptr, destData);
        } else {
            toPrecision(IE::as<IE::MemoryBlob>(userInput), devicePrecision, quantParam, nullptr, destData);
        }
    } else if (!isLayoutMatched) {
        std::stringstream conversionDetailsStr;
        conversionDetailsStr << "Conversion required from " << userLayout << " to " << deviceLayout << ".";
        logger.info("Different layouts of user and device input blobs.\t{0}", conversionDetailsStr.str());
        toLayout(IE::as<IE::MemoryBlob>(userInput), deviceLayout, nullptr, destData);
    }
}

void getOutputAfterInference(IE::Blob::Ptr& userOutput, const IE::TensorDesc& deviceTensorDesc, const void* srcData,
                             Logger logger) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Infer_request::getOutputsAfterInference");
    if (userOutput == nullptr) {
        IE_THROW() << "User output blob null pointer";
    }
    if (srcData == nullptr) {
        IE_THROW() << "Source data null pointer";
    }
    const auto userPrecision = userOutput->getTensorDesc().getPrecision();
    const auto userLayout = userOutput->getTensorDesc().getLayout();
    const auto userNumDims = userOutput->getTensorDesc().getDims().size();
    const auto devicePrecision = deviceTensorDesc.getPrecision();
    const auto deviceLayout = deviceTensorDesc.getLayout();
    const auto deviceNumDims = deviceTensorDesc.getDims().size();

    // [OV design flaw] OV API make_blob_with_precision doesn't have any version with const source data
    IE::Blob::Ptr expectedOutput = makeBlob(deviceTensorDesc, nullptr, const_cast<void*>(srcData));
    if (userPrecision != devicePrecision) {
        logger.info("Different precisions of user and device output blobs.\tConversion required from {0} to {1}",
                    userPrecision.name(), devicePrecision.name());
        expectedOutput = toPrecision(IE::as<IE::MemoryBlob>(expectedOutput), userPrecision);
        if (expectedOutput == nullptr) {
            IE_THROW() << "Blob data null pointer";
        }
    }
    // Default state - only memory copying is required
    auto destLayout = IE::Layout::ANY;
    if (userLayout != deviceLayout && userNumDims == deviceNumDims) {
        // Equal number of dimensions - standard layout conversion and memory copying
        destLayout = userLayout;
    } else if (deviceLayout == IE::Layout::NHWC && userLayout == IE::Layout::CHW) {
        // Special case - NHWC to NCHW layout conversion and memory copying
        destLayout = IE::Layout::NCHW;
    } else if (deviceLayout == IE::Layout::NCHW && userLayout == IE::Layout::HWC) {
        // Special case - NCHW to NHWC layout conversion and memory copying
        destLayout = IE::Layout::NHWC;
    }
    if (destLayout != IE::Layout::ANY) {
        std::stringstream conversionDetailsStr;
        conversionDetailsStr << "Conversion required from " << userLayout << " to " << deviceLayout << ".";
        logger.info("Different layouts of user and device output blobs.\t{0}", conversionDetailsStr.str());
        expectedOutput = toLayout(IE::as<IE::MemoryBlob>(expectedOutput), destLayout);
        if (expectedOutput == nullptr) {
            IE_THROW() << "Blob data null pointer";
        }
    }

    const auto memExpected = IE::as<IE::MemoryBlob>(expectedOutput);
    auto memUser = IE::as<IE::MemoryBlob>(userOutput);
    if (memExpected == nullptr || memUser == nullptr) {
        IE_THROW() << "Blob to MemoryBlob conversion error";
    }
    auto memExpectedLock = memExpected->rmap();
    auto memUserLock = memUser->wmap();
    if (memExpectedLock == nullptr || memUserLock == nullptr) {
        IE_THROW() << "Locking memory error";
    }
    if (memExpected->byteSize() != memUser->byteSize()) {
        IE_THROW() << "Different size of pull and auxiliary blobs";
    }
    if (0 != ie_memcpy(memUserLock, memExpected->byteSize(), memExpectedLock, memUser->byteSize())) {
        IE_THROW() << "memcpy error for pull blobs";
    }
}

bool isRepackingRequired(const IE::TensorDesc& userTensorDesc, const IE::TensorDesc& deviceTensorDesc) {
    const auto userPrecision = userTensorDesc.getPrecision();
    const auto devicePrecision = deviceTensorDesc.getPrecision();
    if (userPrecision == devicePrecision) {
        const auto userLayout = userTensorDesc.getLayout();
        const auto deviceLayout = deviceTensorDesc.getLayout();
        // Equal layouts - no repacking
        if (userLayout == deviceLayout) {
            return false;
        }

        const auto userNumDims = getNumDims(userTensorDesc.getDims());
        const auto deviceNumDims = getNumDims(deviceTensorDesc.getDims());
        // Different 3D/4D/5D layouts - repacking required
        if (userNumDims == deviceNumDims) {
            return (userNumDims > 2);
        }
        const auto minNumDims = std::min(userNumDims, deviceNumDims);
        // Any 1D/2D layouts - no repacking
        if (minNumDims <= 2) {
            return false;
        }
        std::pair<IE::Layout, IE::Layout> layouts{userLayout, deviceLayout};
        if (userNumDims < deviceNumDims) {
            std::swap(layouts.first, layouts.second);
        }
        // Some 4D/3D layouts cases - no repacking
        return !((layouts.first == IE::Layout::NCHW && layouts.second == IE::Layout::CHW) ||
                 (layouts.first == IE::Layout::NHWC && layouts.second == IE::Layout::HWC));
    }
    return true;
}

}  // namespace

namespace vpux {

struct DiscretePipeline final : public Pipeline {
public:
    DiscretePipeline(const Config& config, const ze_device_handle_t& device_handle, const ze_context_handle_t context,
                     ze_graph_dditable_ext_t* graph_ddi_table_ext, const std::shared_ptr<ZeroExecutor::Graph>& graph,
                     ze_graph_profiling_query_handle_t profiling_handle,
                     const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& command_queues)
            : _config(config),
              _command_queues{command_queues},
              _command_list{{{device_handle, context, graph_ddi_table_ext, _config},
                             {device_handle, context, graph_ddi_table_ext, _config},
                             {device_handle, context, graph_ddi_table_ext, _config}}},
              _fence{{{*_command_queues[stage::UPLOAD], _config},
                      {*_command_queues[stage::EXECUTE], _config},
                      {*_command_queues[stage::READBACK], _config}}},
              _event_pool(device_handle, context, stage::COUNT, _config),
              _event{{{_event_pool.handle(), stage::UPLOAD, _config},
                      {_event_pool.handle(), stage::EXECUTE, _config},
                      {_event_pool.handle(), stage::READBACK, _config}}} {
        OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::DiscretePipeline::DiscretePipeline");
        for (const auto& desc : graph->inputs_desc_map()) {
            _inputs.appendArgument(desc.first, desc.second.info);
        }
        _inputs.allocate(device_handle, context);
        _command_list[stage::UPLOAD].appendMemoryCopy(_inputs.getDeviceMemRegion(), _inputs.getHostMemRegion(),
                                                      _inputs.getSize());
        for (const auto& desc : graph->inputs_desc_map()) {
            graph->setArgumentValue(desc.second.idx, _inputs.getDevicePtr(desc.first));
        }

        _command_list[stage::UPLOAD].appendBarrier();
        _event[stage::UPLOAD].AppendSignalEvent(_command_list[stage::UPLOAD]);

        for (const auto& desc : graph->outputs_desc_map()) {
            _outputs.appendArgument(desc.first, desc.second.info);
        }
        _outputs.allocate(device_handle, context);
        _command_list[stage::READBACK].appendMemoryCopy(_outputs.getHostMemRegion(), _outputs.getDeviceMemRegion(),
                                                        _outputs.getSize());
        for (const auto& desc : graph->outputs_desc_map()) {
            graph->setArgumentValue(desc.second.idx, _outputs.getDevicePtr(desc.first));
        }

        _event[stage::UPLOAD].AppendWaitOnEvent(_command_list[stage::EXECUTE]);

        _command_list[stage::EXECUTE].appendGraphExecute(graph->handle(), profiling_handle);

        _event[stage::UPLOAD].AppendEventReset(_command_list[stage::READBACK]);

        for (auto& commandList : _command_list) {
            commandList.close();
        }
    }

    DiscretePipeline(const DiscretePipeline&) = delete;
    DiscretePipeline& operator=(const DiscretePipeline&) = delete;
    virtual ~DiscretePipeline() = default;

    void push() override {
        OV_ITT_TASK_CHAIN(ZERO_INFER_REQUEST_DP_PUSH, itt::domains::LevelZeroBackend, "DiscretePipeline::push",
                          "UPLOAD");
        // Dispatch command to copy input data from upload heap to default heap
        _command_queues[stage::UPLOAD]->executeCommandList(_command_list[stage::UPLOAD]);

        OV_ITT_TASK_NEXT(ZERO_INFER_REQUEST_DP_PUSH, "EXECUTE");
        // Submit the command list for execute
        _command_queues[stage::EXECUTE]->executeCommandList(_command_list[stage::EXECUTE], _fence[stage::EXECUTE]);
    }
    void pull() override {
        OV_ITT_TASK_CHAIN(ZERO_INFER_REQUEST_DP_PULL, itt::domains::LevelZeroBackend, "DiscretePipeline::pull",
                          "EXECUTE");
        // Wait for execute to finish
        _fence[stage::EXECUTE].hostSynchronize();
        OV_ITT_TASK_NEXT(ZERO_INFER_REQUEST_DP_PULL, "READBACK");
        // Schedule the copy of outputs from zeDriverAllocDeviceMem to zeDriverAllocHostMem
        _command_queues[stage::READBACK]->executeCommandList(_command_list[stage::READBACK], _fence[stage::READBACK]);
        // Wait for output copy to finish execution for _fence from the host, to make sure that data
        // is available in the hostMem buffer of the output
        _fence[stage::READBACK].hostSynchronize();
    }
    void reset() const override {
        // Reset the fence objects
        for (auto& fence : _fence) {
            fence.reset();
        }
    }

private:
    const Config _config;
    const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& _command_queues;
    std::array<CommandList, stage::COUNT> _command_list;
    std::array<Fence, stage::COUNT> _fence;
    EventPool _event_pool;
    std::array<Event, stage::COUNT> _event;
};

struct IntegratedPipeline final : public Pipeline {
public:
    IntegratedPipeline(const Config& config, const ze_device_handle_t& device_handle, const ze_context_handle_t context,
                       ze_graph_dditable_ext_t* graph_ddi_table_ext, const std::shared_ptr<ZeroExecutor::Graph>& graph,
                       ze_graph_profiling_query_handle_t profiling_handle, CommandQueue& command_queue)
            : _config(config),
              _command_queue{command_queue},
              _command_list{device_handle, context, graph_ddi_table_ext, _config},
              _fence{_command_queue, _config},
              _event_pool{device_handle, context, 1, _config},
              _event{_event_pool.handle(), 0, _config} {
        OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend,
                           "Zero_infer_request::IntegratedPipeline::IntegratedPipeline");
        for (const auto& desc : graph->inputs_desc_map()) {
            _inputs.appendArgument(desc.first, desc.second.info);
        }
        _inputs.allocate(context, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);
        for (const auto& desc : graph->inputs_desc_map()) {
            graph->setArgumentValue(desc.second.idx, _inputs.getHostPtr(desc.first));
        }

        for (const auto& desc : graph->outputs_desc_map()) {
            _outputs.appendArgument(desc.first, desc.second.info);
        }
        _outputs.allocate(context);
        for (const auto& desc : graph->outputs_desc_map()) {
            graph->setArgumentValue(desc.second.idx, _outputs.getHostPtr(desc.first));
        }

        _command_list.appendGraphExecute(graph->handle(), profiling_handle);
        // appendBarrier used in L0 as well
        if (!sync_output_with_fences_) {
            _command_list.appendBarrier();
            _event.AppendSignalEvent(_command_list);
        }
        _command_list.close();
    }

    IntegratedPipeline(const IntegratedPipeline&) = delete;
    IntegratedPipeline& operator=(const IntegratedPipeline&) = delete;
    virtual ~IntegratedPipeline() = default;

    void push() override {
        OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_PUSH, itt::domains::LevelZeroBackend, "IntegratedPipeline", "push");
        if (sync_output_with_fences_) {
            _command_queue.executeCommandList(_command_list, _fence);
        } else {
            _command_queue.executeCommandList(_command_list);
        }
    }

    void pull() override {
        OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_PULL, itt::domains::LevelZeroBackend, "IntegratedPipeline", "pull");
        if (sync_output_with_fences_) {
            _fence.hostSynchronize();
        } else {
            _event.hostSynchronize();
        }
    }
    void reset() const override {
        if (sync_output_with_fences_) {
            _fence.reset();
        } else {
            _event.reset();
        }
    }

private:
    const Config _config;
    CommandQueue& _command_queue;
    CommandList _command_list;
    Fence _fence;
    EventPool _event_pool;
    Event _event;
    bool sync_output_with_fences_ = true;
};

}  // namespace vpux

//------------------------------------------------------------------------------
ZeroInferRequest::ZeroInferRequest(const IE::InputsDataMap& networkInputs, const IE::OutputsDataMap& networkOutputs,
                                   const Executor::Ptr& executor, const Config& config, const std::string& /*netName*/,
                                   const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                   const std::vector<std::shared_ptr<const ov::Node>>& results,
                                   const vpux::DataMap& networkStatesInfo,
                                   const std::shared_ptr<InferenceEngine::IAllocator>& allocator)
        : IInferRequest(networkInputs, networkOutputs),
          _executorPtr(executor),
          _config(config),
          _logger("ZeroInferRequest", config.get<LOG_LEVEL>()),
          _allocator(allocator),
          _statesInfo(networkStatesInfo),
          _profiling_pool(static_cast<ZeroExecutor*>(_executorPtr.get())->graph()->handle(), zeroProfiling::POOL_SIZE,
                          static_cast<ZeroExecutor*>(_executorPtr.get())->graph()->graph_profiling_ddi_table_ext()),
          _profiling_query(0, static_cast<ZeroExecutor*>(_executorPtr.get())->graph()->device(),
                           static_cast<ZeroExecutor*>(_executorPtr.get())->graph()->graph_profiling_ddi_table_ext()),
          _pipeline(makePipeline()) {
    if (_networkOutputs.empty() || _networkInputs.empty()) {
        IE_THROW() << "No information about network's output/input.";
    }

    _parameters = parameters;
    _results = results;

    const auto& deviceInputs = static_cast<ZeroExecutor*>(_executorPtr.get())->getNetworkDesc().getDeviceInputsInfo();
    for (const auto& networkInput : _networkInputs) {
        const std::string& inputName = networkInput.first;
        const IE::TensorDesc inputTensorDesc = networkInput.second->getTensorDesc();

        if (isRepackingRequired(inputTensorDesc, zeroUtils::mapArguments(deviceInputs, inputName)->getTensorDesc())) {
            _inputs[inputName] = allocateLocalBlob(inputTensorDesc, nullptr);
        } else {
            _inputs[inputName] = allocateLocalBlob(inputTensorDesc, _pipeline->inputs().getHostPtr(inputName));
        }
    }

    const auto& deviceOutputs = static_cast<ZeroExecutor*>(_executorPtr.get())->getNetworkDesc().getDeviceOutputsInfo();
    for (const auto& networkOutput : _networkOutputs) {
        const std::string& outputName = networkOutput.first;
        const IE::TensorDesc outputTensorDesc = networkOutput.second->getTensorDesc();

        if (isRepackingRequired(outputTensorDesc,
                                zeroUtils::mapArguments(deviceOutputs, outputName)->getTensorDesc())) {
            _outputs[outputName] = allocateLocalBlob(outputTensorDesc, nullptr);
        } else {
            _outputs[outputName] = allocateLocalBlob(outputTensorDesc, _pipeline->outputs().getHostPtr(outputName));
        }
    }
}

std::unique_ptr<Pipeline> ZeroInferRequest::makePipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Infer_request::makePipeline");
    if (_profiling_pool.create())
        _profiling_query.create(_profiling_pool._handle);

    ze_device_properties_t properties = {};
    zeroUtils::throwOnFail(
            "zeDeviceGetProperties",
            zeDeviceGetProperties(static_cast<ZeroExecutor*>(_executorPtr.get())->graph()->device(), &properties));

    const ze_device_handle_t device_handle = static_cast<ZeroExecutor*>(_executorPtr.get())->graph()->device();
    const ze_context_handle_t context = static_cast<ZeroExecutor*>(_executorPtr.get())->graph()->context();
    ze_graph_dditable_ext_t* graph_ddi_table_ext =
            static_cast<ZeroExecutor*>(_executorPtr.get())->graph()->graph_ddi_table_ext();
    auto& graph = static_cast<ZeroExecutor*>(_executorPtr.get())->graph();
    auto& command_queues = static_cast<ZeroExecutor*>(_executorPtr.get())->getCommandQueue();

    if (properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED)
        return std::make_unique<IntegratedPipeline>(_config, device_handle, context, graph_ddi_table_ext, graph,
                                                    _profiling_query.getHandle(), *command_queues[stage::EXECUTE]);

    return std::make_unique<DiscretePipeline>(_config, device_handle, context, graph_ddi_table_ext, graph,
                                              _profiling_query.getHandle(), command_queues);
}

void ZeroInferRequest::push(const IE::BlobMap& inputs) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::push");
    _logger.info("ZeroInferRequest::push started");
    const auto& deviceInputs = static_cast<ZeroExecutor*>(_executorPtr.get())->getNetworkDesc().getDeviceInputsInfo();
    const auto& quantParamsInfo = static_cast<ZeroExecutor*>(_executorPtr.get())->getNetworkDesc().getQuantParamsInfo();
    OV_ITT_TASK_CHAIN(ZERO_INFER_REQUEST_PUSH, itt::domains::LevelZeroBackend, "Zero_infer_request::push",
                      "PrepareInput");
    // Copy input data to staging buffer on Cpu (input always first argument)
    for (const auto& inferInput : inputs) {
        const auto& name = inferInput.first;
        const IE::Blob::Ptr& input = inferInput.second;

        const auto& desc = zeroUtils::mapArguments(
                static_cast<ZeroExecutor*>(_executorPtr.get())->graph()->inputs_desc_map(), name);
        const auto& deviceInput = deviceInputs.at(name);
        const auto noQuantParams = quantParamsInfo.find(name) == quantParamsInfo.end();
        const auto quantParams = noQuantParams ? vpux::None : quantParamsInfo.at(name);
        // TODO Currently L0 and Plugin might return different layouts which have dims like [1,1...]
        // They might be reinterpreted in different ways, so this check has been added to prevent that behavior
        if (std::max(getNumDims(desc.info.dims), getNumDims(deviceInput->getTensorDesc().getDims())) > 2) {
            if (!twoApiLayoutCouplingCheck(desc.info.deviceLayout, deviceInput->getLayout())) {
                IE_THROW() << "Parsing error: layouts are different for push blobs";
            }
        }
        if (desc.info.devicePrecision != zeroUtils::getZePrecision(deviceInput->getPrecision())) {
            IE_THROW() << "Parsing error: precisions are different for push blobs";
        }

        if (isRepackingRequired(input->getTensorDesc(), deviceInput->getTensorDesc())) {
            if (!isRepackingPossible(input->getTensorDesc(), deviceInput->getTensorDesc())) {
                IE_THROW() << "Push blobs: repacking is not possible";
            }
            void* hostMem = _pipeline->inputs().getHostPtr(name);
            prepareInputForInference(input, deviceInput->getTensorDesc(), hostMem, quantParams, _logger);
        } else {
            // we should check memory type: host memory or generic and copy if it's a generic
            const auto memInput = IE::as<IE::MemoryBlob>(input);
            VPUX_THROW_UNLESS(memInput != nullptr, "Input IE::Blob::Ptr cannot be cast to IE::MemoryBlob::Ptr");
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
    }

    _pipeline->push();
}

void ZeroInferRequest::pull(IE::BlobMap& outputs) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::pull");
    const auto& deviceOutputs = static_cast<ZeroExecutor*>(_executorPtr.get())->getNetworkDesc().getDeviceOutputsInfo();

    _pipeline->pull();
    // Copy output data to staging buffer on Cpu (input always first argument)
    for (auto& inferOutput : outputs) {
        const auto& name = inferOutput.first;
        IE::Blob::Ptr& output = inferOutput.second;

        const auto& desc = zeroUtils::mapArguments(
                static_cast<ZeroExecutor*>(_executorPtr.get())->graph()->outputs_desc_map(), name);
        const auto& deviceOutput = deviceOutputs.at(name);
        if (std::max(getNumDims(desc.info.dims), getNumDims(deviceOutput->getTensorDesc().getDims())) > 2) {
            if (!twoApiLayoutCouplingCheck(desc.info.deviceLayout, deviceOutput->getLayout())) {
                IE_THROW() << "Parsing error: layouts are different for pull blobs";
            }
        }
        if (desc.info.devicePrecision != zeroUtils::getZePrecision(deviceOutput->getPrecision())) {
            IE_THROW() << "Parsing error: precisions are different for pull blobs";
        }

        if (isRepackingRequired(output->getTensorDesc(), deviceOutput->getTensorDesc())) {
            if (!isRepackingPossible(output->getTensorDesc(), deviceOutput->getTensorDesc())) {
                IE_THROW() << "Pull blobs: repacking is not possible";
            }
            const void* hostMem = _pipeline->outputs().getHostPtr(name);
            getOutputAfterInference(output, deviceOutput->getTensorDesc(), hostMem, _logger);
        } else {
            // we should check memory type: host memory or generic and copy if it's a generic
            const auto memOutput = IE::as<IE::MemoryBlob>(output);
            VPUX_THROW_UNLESS(memOutput != nullptr, "Output IE::Blob::Ptr cannot be cast to IE::MemoryBlob::Ptr");
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
    }

    _pipeline->reset();
}

void ZeroInferRequest::InferImpl() {
    InferAsync();
    GetResult();
}

void ZeroInferRequest::InferAsync() {
    _logger.debug("InferRequest::InferAsync started");
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "InferAsync");

    execDataPreprocessing(_inputs);
    push(_inputs);
}

std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> ZeroInferRequest::QueryState() {
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
    pull(_outputs);
    _logger.debug("InferRequest::GetResult finished");
}

std::map<std::string, IE::InferenceEngineProfileInfo> ZeroInferRequest::GetPerformanceCounts() const {
    if (_config.get<PERF_COUNT>()) {
        return const_cast<ZeroInferRequest*>(this)->_profiling_query.getLayerStatistics(
                _config.get<COMPILER_TYPE>(), static_cast<ZeroExecutor*>(_executorPtr.get())->graph()->blob());
    } else {
        return {};
    }
}
