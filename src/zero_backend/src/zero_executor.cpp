//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "zero_executor.h"

#include "zero_allocator.h"

#include "vpux/utils/IE/blob.hpp"

#include <blob_factory.hpp>

#include "mcm/utils/profiling_parser.hpp"

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

using namespace vpux;

namespace {
void throwOnFail(const std::string& step, const ze_result_t result) {
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "throwOnFail: " << step << " result: 0x" << std::hex << uint64_t(result);
    }
}

size_t precisionToSize(const ze_graph_argument_precision_t val) {
    switch (val) {
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return 4;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return 2;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return 2;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return 1;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return 4;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return 2;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return 1;
    default:
        IE_THROW() << "precisionToSize switch->default reached";
    }
}

_ze_graph_argument_precision_t getZePrecision(InferenceEngine::Precision precision) {
    switch (precision) {
    case InferenceEngine::Precision::I8:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT8;
    case InferenceEngine::Precision::U8:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT8;
    case InferenceEngine::Precision::I16:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT16;
    case InferenceEngine::Precision::U16:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT16;
    case InferenceEngine::Precision::I32:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT32;
    case InferenceEngine::Precision::FP16:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP16;
    case InferenceEngine::Precision::FP32:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP32;
    case InferenceEngine::Precision::BIN:
        return ZE_GRAPH_ARGUMENT_PRECISION_BIN;
    default:
        return ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN;
    }
}

size_t layoutCount(const ze_graph_argument_layout_t val) {
    switch (val) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCHW:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW:
        return 5;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        return 5;
    case ZE_GRAPH_ARGUMENT_LAYOUT_OIHW:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_C:
        return 1;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CHW:
        return 3;
    case ZE_GRAPH_ARGUMENT_LAYOUT_HW:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NC:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        return 2;
    default:
        IE_THROW() << "layoutCount switch->default reached";
    }
}

// check that ie Layout and zeroApi layout are the same for some argument
bool twoApiLayoutCouplingCheck(const ze_graph_argument_layout_t zeroL, const ::InferenceEngine::Layout ieL) {
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

size_t getSizeIOBytes(const ze_graph_argument_properties_t& argument) {
    size_t num_elements = 1;
    for (size_t i = 0; i < layoutCount(argument.layout); ++i) {
        num_elements *= argument.dims[i];
    }
    const size_t size_in_bytes = num_elements * precisionToSize(argument.precision);
    return size_in_bytes;
}

template <typename Map>
auto mapArguments(Map& zero, const std::string& key) -> typename Map::mapped_type& {
    for (auto& p : zero) {
        if (std::string::npos != p.first.find(key)) {
            return p.second;
        }
    }

    IE_THROW() << "mapArguments: fail to map";
}

template <typename Map>
auto mapArguments(Map& zero, const std::string& key, std::size_t pos) -> typename Map::mapped_type& {
    for (auto& p : zero) {
        if (std::string::npos != p.first.find(key)) {
            return p.second;
        }
    }

    std::size_t zero_pos = 0;
    for (auto& p : zero) {
        if ((p.first == "profilingOutput") || (zero_pos == pos)) {
            return p.second;
        }
        zero_pos++;
    }

    IE_THROW() << "mapArguments: fail to map";
}
}  // namespace

ZeroExecutorCommon::ZeroExecutorCommon(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle,
                                       ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                                       ze_fence_dditable_ext_t* fence_ddi_table_ext,
                                       const vpux::NetworkDescription::Ptr& networkDescription,
                                       const ZeroConfig& config)
        : _config(config),
          _logger(std::make_shared<vpu::Logger>("ZeroExecutor", _config.logLevel(), vpu::consoleOutput())),
          _driver_handle(driver_handle),
          _device_handle(device_handle),
          _context(context),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _fence_ddi_table_ext(fence_ddi_table_ext),
          _push_count(0),
          _pull_count(0),
          _perf_count(0),
          _networkDesc(networkDescription),
          _graph(std::make_shared<graph>(driver_handle, device_handle, context, networkDescription, graph_ddi_table_ext,
                                         fence_ddi_table_ext)) {
}

ZeroExecutorCommon::ZeroExecutorCommon(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle,
                                       ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                                       ze_fence_dditable_ext_t* fence_ddi_table_ext,
                                       const vpux::NetworkDescription::Ptr& networkDescription,
                                       std::shared_ptr<graph> graph, const ZeroConfig& config)
        : _config(config),
          _logger(std::make_shared<vpu::Logger>("ZeroExecutor", _config.logLevel(), vpu::consoleOutput())),
          _driver_handle(driver_handle),
          _device_handle(device_handle),
          _context(context),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _fence_ddi_table_ext(fence_ddi_table_ext),
          _push_count(0),
          _pull_count(0),
          _perf_count(0),
          _networkDesc(networkDescription),
          _graph(graph) {
}

ZeroExecutorCommon::hostMem::hostMem(const ze_driver_handle_t driver_handle, const ze_context_handle_t context,
                                     const size_t size)
        : _driver_handle(driver_handle), _context(context), _size(size) {
    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    throwOnFail("zeMemAllocHost", zeMemAllocHost(_context, &desc, _size, _alignment, &_data));
}
ZeroExecutorCommon::hostMem::~hostMem() {
    if (_data) {
        throwOnFail("zeMemFree hostMem", zeMemFree(_context, _data));
    }
}

ZeroExecutorCommon::deviceMem::deviceMem(const ze_driver_handle_t driver_handle, const ze_device_handle_t deh_,
                                         ze_context_handle_t context, const size_t size)
        : _driver_handle(driver_handle), _context(context), _size(size) {
    ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};

    throwOnFail("zeDriverAllocDeviceMem", zeMemAllocDevice(_context, &desc, _size, _alignment, deh_, &_data));
}
ZeroExecutorCommon::deviceMem::~deviceMem() {
    if (_data) {
        throwOnFail("zeMemFree deviceMem", zeMemFree(_context, _data));
    }
}

ZeroExecutorCommon::commandList::commandList(const ze_device_handle_t& device_handle,
                                             const ze_context_handle_t& context,
                                             ze_graph_dditable_ext_t* graph_ddi_table_ext)
        : _context(context), _graph_ddi_table_ext(graph_ddi_table_ext) {
    ze_command_list_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
    throwOnFail("zeCommandListCreate", zeCommandListCreate(_context, device_handle, &desc, &_handle));
    reset();
}
void ZeroExecutorCommon::commandList::reset() {
    throwOnFail("zeCommandListReset", zeCommandListReset(_handle));
}
void ZeroExecutorCommon::commandList::appendMemoryCopy(void* dst, const void* src, size_t size) {
    throwOnFail("zeCommandListAppendMemoryCopy",
                zeCommandListAppendMemoryCopy(_handle, dst, src, size, nullptr, 0, nullptr));
}
void ZeroExecutorCommon::commandList::appendGraphInitialize(const ze_graph_handle_t& graph_handle) {
    throwOnFail("zeCommandListAppendGraphInitialize",
                _graph_ddi_table_ext->pfnAppendGraphInitialize(_handle, graph_handle));
}
void ZeroExecutorCommon::commandList::appendGraphExecute(const ze_graph_handle_t& graph_handle) {
    throwOnFail("zeCommandListAppendGraphExecute", _graph_ddi_table_ext->pfnAppendGraphExecute(_handle, graph_handle));
}
void ZeroExecutorCommon::commandList::close() {
    throwOnFail("zeCommandListClose", zeCommandListClose(_handle));
}
ZeroExecutorCommon::commandList::~commandList() {
    throwOnFail("zeCommandListDestroy", zeCommandListDestroy(_handle));
}

ZeroExecutorCommon::graph::graph(const ze_driver_handle_t& driver_handle, const ze_device_handle_t& device_handle,
                                 const ze_context_handle_t& context, const NetworkDescription::CPtr networkDesc,
                                 ze_graph_dditable_ext_t* graph_ddi_table_ext,
                                 ze_fence_dditable_ext_t* fence_ddi_table_ext)
        : _context(context),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _mem(driver_handle, _context, networkDesc->getCompiledNetwork().size()),
          _command_queue(std::make_shared<commandQueue>(device_handle, _context)),
          _command_list(device_handle, _context, graph_ddi_table_ext),
          _fence(std::make_shared<fence>(_command_queue, fence_ddi_table_ext)),
          _fence_value(0) {
    _mem.copyFrom(networkDesc->getCompiledNetwork());

    ze_graph_desc_t desc = {ZE_GRAPH_FORMAT_NATIVE, _mem.size(), static_cast<uint8_t*>(_mem.data())};
    throwOnFail("zeGraphCreate", _graph_ddi_table_ext->pfnCreate(device_handle, &desc, &_handle));

    throwOnFail("zeGraphGetProperties", _graph_ddi_table_ext->pfnGetProperties(_handle, &_props));
    for (uint32_t index = 0; index < _props.numGraphArgs; ++index) {
        ze_graph_argument_properties_t arg;
        throwOnFail("zeGraphGetArgumentProperties",
                    _graph_ddi_table_ext->pfnGetArgumentProperties(_handle, index, &arg));
        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            auto deviceInputs = networkDesc->getDeviceInputsInfo();

            // [Track number: S#49808]
            // hack for correct memory allocation on device
            arg.precision = getZePrecision(deviceInputs.at(arg.name)->getPrecision());

            _inputs_desc_map.emplace(std::make_pair(std::string(arg.name), argumentDescriptor{arg, index}));
        } else {
            _outputs_desc_map.emplace(std::make_pair(std::string(arg.name), argumentDescriptor{arg, index}));
        }
    }

    _command_list.appendGraphInitialize(_handle);
    _command_list.close();
}
void ZeroExecutorCommon::graph::init() {
    _command_queue->executeCommandList(_command_list);
    _fence->deviceSignal(++_fence_value);
    _fence->hostSynchronize(_fence_value);
}
void ZeroExecutorCommon::graph::setArgumentValue(uint32_t argi_, const void* argv_) const {
    throwOnFail("zeGraphSetArgumentValue", _graph_ddi_table_ext->pfnSetArgumentValue(_handle, argi_, argv_));
}
ZeroExecutorCommon::graph::~graph() {
    throwOnFail("zeGraphDestroy", _graph_ddi_table_ext->pfnDestroy(_handle));
}

ZeroExecutorCommon::commandQueue::commandQueue(const ze_device_handle_t& device_handle,
                                               const ze_context_handle_t& context)
        : _context(context) {
    ze_command_queue_desc_t queue_desc = {
            ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0, ZE_COMMAND_QUEUE_MODE_DEFAULT,
            ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    throwOnFail("zeCommandQueueCreate", zeCommandQueueCreate(_context, device_handle, &queue_desc, &_handle));
}
void ZeroExecutorCommon::commandQueue::executeCommandList(commandList& command_list) {
    throwOnFail("zeCommandQueueExecuteCommandLists",
                zeCommandQueueExecuteCommandLists(_handle, 1, &command_list._handle, nullptr));
}
ZeroExecutorCommon::commandQueue::~commandQueue() {
    throwOnFail("zeCommandQueueDestroy", zeCommandQueueDestroy(_handle));
}

ZeroExecutorCommon::pipelineCommon::pipelineCommon(const ze_driver_handle_t& driver_handle,
                                                   const ze_device_handle_t& device_handle,
                                                   const ze_context_handle_t context,
                                                   ze_graph_dditable_ext_t* graph_ddi_table_ext,
                                                   const std::shared_ptr<graph>& graph)
        : _command_list{{{device_handle, context, graph_ddi_table_ext},
                         {device_handle, context, graph_ddi_table_ext},
                         {device_handle, context, graph_ddi_table_ext}}} {
    for (const auto& desc : graph->_inputs_desc_map) {
        auto size = getSizeIOBytes(desc.second.info);
        _inputs_host_mem_map.try_emplace(desc.first, driver_handle, context, size);
        _inputs_device_mem_map.try_emplace(desc.first, driver_handle, device_handle, context, size);

        auto& hostMem = mapArguments(_inputs_host_mem_map, desc.first);
        auto& deviceMem = mapArguments(_inputs_device_mem_map, desc.first);
        _command_list[stage::UPLOAD].appendMemoryCopy(deviceMem.data(), hostMem.data(), size);

        graph->setArgumentValue(desc.second.idx, deviceMem.data());
    }
}

ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::ZeroExecutor(
        ze_driver_handle_t driver_handle, ze_device_handle_t device_handle, ze_context_handle_t context,
        ze_graph_dditable_ext_t* graph_ddi_table_ext, ze_fence_dditable_ext_t* fence_ddi_table_ext,
        const vpux::NetworkDescription::Ptr& networkDescription, const ZeroConfig& config)
        : ZeroExecutorCommon(driver_handle, device_handle, context, graph_ddi_table_ext, fence_ddi_table_ext,
                             networkDescription, config),
          _command_queue{{std::make_shared<commandQueue>(device_handle, context),
                          std::make_shared<commandQueue>(device_handle, context),
                          std::make_shared<commandQueue>(device_handle, context)}},
          _fence{{std::make_unique<fence>(_command_queue[stage::UPLOAD], fence_ddi_table_ext),
                  std::make_unique<fence>(_command_queue[stage::EXECUTE], fence_ddi_table_ext),
                  std::make_unique<fence>(_command_queue[stage::READBACK], fence_ddi_table_ext)}},
          _pipeline(std::make_unique<pipeline>(driver_handle, device_handle, _context, graph_ddi_table_ext,
                                               fence_ddi_table_ext, _graph)) {
    _graph->init();
}

ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::ZeroExecutor(
        ze_driver_handle_t driver_handle, ze_device_handle_t device_handle, ze_context_handle_t context,
        ze_graph_dditable_ext_t* graph_ddi_table_ext, ze_fence_dditable_ext_t* fence_ddi_table_ext,
        const vpux::NetworkDescription::Ptr& networkDescription,
        const std::array<std::shared_ptr<commandQueue>, stage::COUNT>& command_queue,
        const std::shared_ptr<graph>& graph, const ZeroConfig& config)
        : ZeroExecutorCommon(driver_handle, device_handle, context, graph_ddi_table_ext, fence_ddi_table_ext,
                             networkDescription, graph, config),
          _command_queue(command_queue),
          _fence{{std::make_unique<fence>(_command_queue[stage::UPLOAD], fence_ddi_table_ext),
                  std::make_unique<fence>(_command_queue[stage::EXECUTE], fence_ddi_table_ext),
                  std::make_unique<fence>(_command_queue[stage::READBACK], fence_ddi_table_ext)}},
          _pipeline(std::make_unique<pipeline>(driver_handle, device_handle, _context, graph_ddi_table_ext,
                                               fence_ddi_table_ext, _graph)) {
}

ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::fence::fence(
        const std::shared_ptr<commandQueue>& command_queue, ze_fence_dditable_ext_t* fence_ddi_table_ext)
        : _fence_ddi_table_ext(fence_ddi_table_ext) {
    ze_fence_desc_t fence_desc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
    throwOnFail("zeFenceCreate", zeFenceCreate(command_queue->_handle, &fence_desc, &_handle));
}
void ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::fence::reset() {
    throwOnFail("zeFenceReset", zeFenceReset(_handle));
}
void ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::fence::hostSynchronize(
        uint32_t fence_value) {
    throwOnFail("zeFenceHostSynchronize", zeFenceHostSynchronize(_handle, fence_value));
}
void ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::fence::deviceSynchronize(
        const std::shared_ptr<commandQueue>& queue, uint32_t fence_value) {
    throwOnFail("zeFenceDeviceSynchronize",
                _fence_ddi_table_ext->pfnDeviceSynchronize(queue->_handle, _handle, fence_value));
}
void ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::fence::deviceSignal(uint32_t fence_value) {
    throwOnFail("zeFenceDeviceSignal", _fence_ddi_table_ext->pfnDeviceSignal(_handle, fence_value));
}
ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::fence::~fence() {
    throwOnFail("zeFenceDestroy", zeFenceDestroy(_handle));
}

ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::pipeline::pipeline(
        const ze_driver_handle_t& driver_handle, const ze_device_handle_t& device_handle,
        const ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
        ze_fence_dditable_ext_t* fence_ddi_table_ext, const std::shared_ptr<graph>& graph)
        : pipelineCommon(driver_handle, device_handle, context, graph_ddi_table_ext, graph) {
    for (const auto& desc : graph->_outputs_desc_map) {
        auto size = getSizeIOBytes(desc.second.info);
        _outputs_host_mem_map.try_emplace(desc.first, driver_handle, context, size);
        _outputs_device_mem_map.try_emplace(desc.first, driver_handle, device_handle, context, size);

        auto& hostMem = mapArguments(_outputs_host_mem_map, desc.first);
        auto& deviceMem = mapArguments(_outputs_device_mem_map, desc.first);
        _command_list[stage::READBACK].appendMemoryCopy(hostMem.data(), deviceMem.data(), size);

        graph->setArgumentValue(desc.second.idx, deviceMem.data());
    }

    _command_list[stage::EXECUTE].appendGraphExecute(graph->_handle);

    for (auto& commandList : _command_list) {
        commandList.close();
    }
}

static void prepareInputForInference(const InferenceEngine::Blob::Ptr& actualInput,
                                     const InferenceEngine::Precision& expectedPrecision, void* dest_data = nullptr) {
    if (actualInput == nullptr) {
        IE_THROW() << "Actual input blob null pointer!";
    }
    if (actualInput->getTensorDesc().getPrecision() == expectedPrecision || dest_data == nullptr) {
        return;
    }

    vpux::toPrecision(InferenceEngine::as<InferenceEngine::MemoryBlob>(actualInput), expectedPrecision, nullptr,
                      dest_data);
}

void ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::push(
        const InferenceEngine::BlobMap& inputs) {
    _logger->info("ZeroExecutor::push started");
    const auto& deviceInputs = _networkDesc->getDeviceInputsInfo();

    const auto iteration = _push_count++;

    // Copy the inputs from host to zeDriverAllocHostMem buffer set up for copy
    for (const auto& inferInput : inputs) {
        const std::string& name = inferInput.first;
        const InferenceEngine::Blob::Ptr& input = inferInput.second;

        auto& desc = mapArguments(_graph->_inputs_desc_map, name);
        if (!twoApiLayoutCouplingCheck(desc.info.layout, input->getTensorDesc().getLayout()))
            IE_THROW() << "Layouts is different for push blobs";
        if (input->byteSize() != getSizeIOBytes(desc.info)) {
            _logger->info("Sizes are different for push blobs. Need precision convert");
        }

        auto& hostMem = mapArguments(_pipeline->_inputs_host_mem_map, name);
        if (input->getTensorDesc().getPrecision() == deviceInputs.at(name)->getPrecision()) {
            hostMem.copyFrom(input);
        } else {
            prepareInputForInference(input, deviceInputs.at(name)->getPrecision(), hostMem.data());
        }
    }

    // Schedule the copy of inputs from zeDriverAllocHostMem to zeDriverAllocDeviceMem
    _command_queue[stage::UPLOAD]->executeCommandList(_pipeline->_command_list[stage::UPLOAD]);

    // Signal fence from device after input copy is executed from upload command queue on device
    _fence[stage::UPLOAD]->deviceSignal(iteration + 1);

    // Wait for input copy to finish for iteration from the execute command queue on device,
    // before executing the inference to make sure that input data is available
    _fence[stage::UPLOAD]->deviceSynchronize(_command_queue[stage::EXECUTE], iteration + 1);

    // Schedule the inference, wait for completion will be in matching pull
    _command_queue[stage::EXECUTE]->executeCommandList(_pipeline->_command_list[stage::EXECUTE]);

    // Signal fence from device after inference is executed from execute command queue on device
    _fence[stage::EXECUTE]->deviceSignal(iteration + 1);

    // Wait for inference to finish for _pull_count from the readback command queue on device,
    // to make sure that output data is available
    _fence[stage::EXECUTE]->deviceSynchronize(_command_queue[READBACK], iteration + 1);

    // Schedule the copy of outputs from zeDriverAllocDeviceMem to zeDriverAllocHostMem
    _command_queue[stage::READBACK]->executeCommandList(_pipeline->_command_list[stage::READBACK]);

    // Signal fence from device after output copy is completed from readback command queue on device
    _fence[stage::READBACK]->deviceSignal(iteration + 1);

    _logger->info("ZeroExecutor::push finished");
}

void ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::pull(InferenceEngine::BlobMap& outputs) {
    _logger->info("ZeroExecutor::pull started");

    const auto iteration = _pull_count++;

    // Wait for output copy to finish execution for _pull_count from the host, to make sure that data
    // is available in the hostMem buffer of the output
    _fence[stage::READBACK]->hostSynchronize(iteration + 1);

    // Copy the outputs from set up zeDriverAllocHostMem to the host
    std::size_t out_id = 0;
    for (auto& inferOutput : outputs) {
        const auto& name = inferOutput.first;
        InferenceEngine::Blob::Ptr& output = inferOutput.second;

        auto& desc = mapArguments(_graph->_outputs_desc_map, name, out_id);
        if (!twoApiLayoutCouplingCheck(desc.info.layout, output->getTensorDesc().getLayout()))
            IE_THROW() << "Layouts is different for pull blobs";
        if (output->byteSize() != getSizeIOBytes(desc.info))
            IE_THROW() << "Sizes are different for pull blobs";

        auto& hostMem = mapArguments(_pipeline->_outputs_host_mem_map, name, out_id);
        hostMem.copyTo(output);
        out_id++;
    }

    _logger->info("ZeroExecutor::pull finished");
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>
ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::getLayerStatistics() {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfCounts;

    const auto blob = _graph->_mem.data();
    auto profilingOutputBlob = _pipeline->_outputs_host_mem_map.find("profilingOutput");
    if (profilingOutputBlob == _pipeline->_outputs_host_mem_map.end()) {
        _logger->warning("No profiling output. Blob was compiled without profiling enabled or do not contain "
                         "profiling info.");
        return perfCounts;
    }

    std::vector<mv::utils::ProfInfo> deviceProfiling;
    mv::utils::getProfilingInfo(blob, profilingOutputBlob->second.data(), deviceProfiling);

    int execution_index = 0;
    InferenceEngine::InferenceEngineProfileInfo info;
    for (const auto& profilingEntry : deviceProfiling) {
        info.status = InferenceEngine::InferenceEngineProfileInfo::EXECUTED;
        info.cpu_uSec = info.realTime_uSec = profilingEntry.time;
        info.execution_index = execution_index++;
        size_t typeLen = sizeof(info.layer_type) / sizeof(info.layer_type[0]);
        std::size_t length = profilingEntry.layer_type.copy(info.layer_type, typeLen, 0);
        info.layer_type[length] = '\0';
        typeLen = sizeof(info.exec_type) / sizeof(info.exec_type[0]);
        length = profilingEntry.exec_type.copy(info.exec_type, typeLen, 0);
        info.exec_type[length] = '\0';
        perfCounts[profilingEntry.name] = info;
    }

    return perfCounts;
}

Executor::Ptr ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>::clone() const {
    return std::make_shared<ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>>(
            _driver_handle, _device_handle, _context, _graph_ddi_table_ext, _fence_ddi_table_ext, _networkDesc,
            _command_queue, _graph, _config);
}

ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::ZeroExecutor(
        ze_driver_handle_t driver_handle, ze_device_handle_t device_handle, ze_context_handle_t context,
        ze_graph_dditable_ext_t* graph_ddi_table_ext, ze_fence_dditable_ext_t* fence_ddi_table_ext,
        const vpux::NetworkDescription::Ptr& networkDescription, const ZeroConfig& config)
        : ZeroExecutorCommon(driver_handle, device_handle, context, graph_ddi_table_ext, fence_ddi_table_ext,
                             networkDescription, config),
          _command_queue{{std::make_shared<commandQueue>(device_handle, context),
                          std::make_shared<commandQueue>(device_handle, context),
                          std::make_shared<commandQueue>(device_handle, context)}},
          _fence{{std::make_unique<fence>(_command_queue[stage::UPLOAD], fence_ddi_table_ext),
                  std::make_unique<fence>(_command_queue[stage::EXECUTE], fence_ddi_table_ext),
                  std::make_unique<fence>(_command_queue[stage::READBACK], fence_ddi_table_ext)}},
          _pipeline(std::make_unique<pipeline>(driver_handle, device_handle, context, graph_ddi_table_ext,
                                               fence_ddi_table_ext, _graph)) {
    _graph->init();
}

ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::ZeroExecutor(
        ze_driver_handle_t driver_handle, ze_device_handle_t device_handle, ze_context_handle_t context,
        ze_graph_dditable_ext_t* graph_ddi_table_ext, ze_fence_dditable_ext_t* fence_ddi_table_ext,
        const vpux::NetworkDescription::Ptr& networkDescription,
        const std::array<std::shared_ptr<commandQueue>, stage::COUNT>& command_queue,
        const std::shared_ptr<graph>& graph, const ZeroConfig& config)
        : ZeroExecutorCommon(driver_handle, device_handle, context, graph_ddi_table_ext, fence_ddi_table_ext,
                             networkDescription, graph, config),
          _command_queue(command_queue),
          _fence{{std::make_unique<fence>(_command_queue[stage::UPLOAD], fence_ddi_table_ext),
                  std::make_unique<fence>(_command_queue[stage::EXECUTE], fence_ddi_table_ext),
                  std::make_unique<fence>(_command_queue[stage::READBACK], fence_ddi_table_ext)}},
          _pipeline(std::make_unique<pipeline>(driver_handle, device_handle, context, graph_ddi_table_ext,
                                               fence_ddi_table_ext, graph)) {
}

ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::eventPool_t::eventPool_t(
        ze_device_handle_t device_handle, const ze_context_handle_t& context, uint32_t event_count)
        : _event_count(event_count) {
    ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
                                            event_count};
    throwOnFail("zeEventPoolCreate", zeEventPoolCreate(context, &event_pool_desc, 1, &device_handle, &_handle));
}
ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::event_t::event_t(
        ze_device_handle_t device_handle, const ze_context_handle_t& context, const ze_event_pool_handle_t& event_pool,
        uint32_t event_index)
        : _context(context), _device_t(device_handle) {
    ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, event_index, 0, 0};
    throwOnFail("zeEventCreate", zeEventCreate(event_pool, &event_desc, &_handle));
}
void ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::event_t::AppendSignalEvent(
        commandList& command_list) {
    throwOnFail("zeCommandListAppendSignalEvent", zeCommandListAppendSignalEvent(command_list._handle, _handle));
}
void ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::event_t::AppendWaitOnEvent(
        commandList& command_list) {
    throwOnFail("zeCommandListAppendWaitOnEvents", zeCommandListAppendWaitOnEvents(command_list._handle, 1, &_handle));
}
void ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::event_t::AppendEventReset(
        commandList& command_list) {
    throwOnFail("zeCommandListAppendWaitOnEvents", zeCommandListAppendEventReset(command_list._handle, _handle));
}

ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::pipeline::pipeline(
        const ze_driver_handle_t& driver_handle, const ze_device_handle_t& device_handle,
        const ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
        ze_fence_dditable_ext_t* fence_ddi_table_ext, const std::shared_ptr<graph>& graph)
        : pipelineCommon(driver_handle, device_handle, context, graph_ddi_table_ext, graph),
          _event_pool(device_handle, context, stage::COUNT),
          _event{{{device_handle, context, _event_pool._handle, stage::UPLOAD},
                  {device_handle, context, _event_pool._handle, stage::EXECUTE},
                  {device_handle, context, _event_pool._handle, stage::READBACK}}} {
    _event[stage::UPLOAD].AppendSignalEvent(_command_list[stage::UPLOAD]);

    for (const auto& desc : graph->_outputs_desc_map) {
        auto size = getSizeIOBytes(desc.second.info);
        _outputs_host_mem_map.try_emplace(desc.first, driver_handle, context, size);
        _outputs_device_mem_map.try_emplace(desc.first, driver_handle, device_handle, context, size);

        auto& hostMem = mapArguments(_outputs_host_mem_map, desc.first);
        auto& deviceMem = mapArguments(_outputs_device_mem_map, desc.first);
        _command_list[stage::READBACK].appendMemoryCopy(hostMem.data(), deviceMem.data(), size);

        graph->setArgumentValue(desc.second.idx, deviceMem.data());
    }

    _event[stage::UPLOAD].AppendWaitOnEvent(_command_list[stage::EXECUTE]);
    _command_list[stage::EXECUTE].appendGraphExecute(graph->_handle);

    _event[stage::UPLOAD].AppendEventReset(_command_list[stage::READBACK]);

    for (auto& commandList : _command_list) {
        commandList.close();
    }
}

ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::pipeline::~pipeline() {
    zeEventPoolDestroy(_event_pool._handle);
}

void ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::push(
        const InferenceEngine::BlobMap& inputs) {
    _logger->info("ZeroExecutor::push started");
    const auto& deviceInputs = _networkDesc->getDeviceInputsInfo();

    const auto iteration = _push_count++;

    // Copy input data to staging buffer on Cpu (input always first argument)
    for (const auto& inferInput : inputs) {
        const std::string& name = inferInput.first;
        const InferenceEngine::Blob::Ptr& input = inferInput.second;

        auto& desc = mapArguments(_graph->_inputs_desc_map, name);
        if (!twoApiLayoutCouplingCheck(desc.info.layout, input->getTensorDesc().getLayout()))
            IE_THROW() << "Layouts is different for push blobs";
        if (input->byteSize() != getSizeIOBytes(desc.info)) {
            _logger->info("Sizes are different for push blobs. Need precision convert");
        }

        auto& hostMem = mapArguments(_pipeline->_inputs_host_mem_map, name);
        if (input->getTensorDesc().getPrecision() == deviceInputs.at(name)->getPrecision()) {
            hostMem.copyFrom(input);
        } else {
            prepareInputForInference(input, deviceInputs.at(name)->getPrecision(), hostMem.data());
        }
    }

    // Dispatch command to copy input data from upload heap to default heap
    _command_queue[stage::UPLOAD]->executeCommandList(_pipeline->_command_list[stage::UPLOAD]);

    // Submit the command list for execute
    _command_queue[stage::EXECUTE]->executeCommandList(_pipeline->_command_list[stage::EXECUTE]);

    // Signal fence from device after inference is executed from execute command queue on device
    _fence[stage::EXECUTE]->deviceSignal(iteration + 1);

    // Wait for inference to finish for _pull_count from the readback command queue on device,
    // to make sure that output data is available
    _fence[stage::EXECUTE]->deviceSynchronize(_command_queue[READBACK], iteration + 1);

    // Schedule the copy of outputs from zeDriverAllocDeviceMem to zeDriverAllocHostMem
    _command_queue[stage::READBACK]->executeCommandList(_pipeline->_command_list[stage::READBACK]);

    // Signal fence from device after output copy is completed from readback command queue on device
    _fence[stage::READBACK]->deviceSignal(iteration + 1);
}

Executor::Ptr ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::clone() const {
    return std::make_shared<ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>>(
            _driver_handle, _device_handle, _context, _graph_ddi_table_ext, _fence_ddi_table_ext, _networkDesc,
            _command_queue, _graph, _config);
}

void ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::pull(InferenceEngine::BlobMap& outputs) {
    const auto iteration = _pull_count++;

    // Wait for output copy to finish execution for _pull_count from the host, to make sure that data
    // is available in the hostMem buffer of the output
    _fence[stage::READBACK]->hostSynchronize(iteration + 1);

    // Copy output data to staging buffer on Cpu (input always first argument)
    for (auto& inferOutput : outputs) {
        const auto& name = inferOutput.first;
        InferenceEngine::Blob::Ptr& output = inferOutput.second;

        auto& desc = mapArguments(_graph->_outputs_desc_map, name);
        if (!twoApiLayoutCouplingCheck(desc.info.layout, output->getTensorDesc().getLayout()))
            IE_THROW() << "Layouts is different for pull blobs";
        if (output->byteSize() != getSizeIOBytes(desc.info))
            IE_THROW() << "Sizes are different for pull blobs";

        auto& hostMem = mapArguments(_pipeline->_outputs_host_mem_map, name);
        hostMem.copyTo(output);
    }
}

InferenceEngine::Parameter ZeroExecutorCommon::getParameter(const std::string&) const {
    return InferenceEngine::Parameter();
}
void ZeroExecutorCommon::setup(const InferenceEngine::ParamMap&) {
    IE_THROW() << "Not implemented";
}
bool ZeroExecutorCommon::isPreProcessingSupported(const PreprocMap& preProcMap) const {
    return false;
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>
ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>::getLayerStatistics() {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfCounts;

    const auto blob = _graph->_mem.data();
    auto profilingOutputBlob = _pipeline->_outputs_host_mem_map.find("profilingOutput");
    if (profilingOutputBlob == _pipeline->_outputs_host_mem_map.end()) {
        _logger->warning("No profiling output. Blob was compiled without profiling enabled or do not contain "
                         "profiling info.");
        return perfCounts;
    }

    std::vector<mv::utils::ProfInfo> deviceProfiling;
    mv::utils::getProfilingInfo(blob, profilingOutputBlob->second.data(), deviceProfiling);

    int execution_index = 0;
    InferenceEngine::InferenceEngineProfileInfo info;
    for (const auto& profilingEntry : deviceProfiling) {
        info.status = InferenceEngine::InferenceEngineProfileInfo::EXECUTED;
        info.cpu_uSec = info.realTime_uSec = profilingEntry.time;
        info.execution_index = execution_index++;
        size_t typeLen = sizeof(info.layer_type) / sizeof(info.layer_type[0]);
        std::size_t length = profilingEntry.layer_type.copy(info.layer_type, typeLen, 0);
        info.layer_type[length] = '\0';
        typeLen = sizeof(info.exec_type) / sizeof(info.exec_type[0]);
        length = profilingEntry.exec_type.copy(info.exec_type, typeLen, 0);
        info.exec_type[length] = '\0';
        perfCounts[profilingEntry.name] = info;
    }

    return perfCounts;
}

void ZeroExecutorCommon::push(const InferenceEngine::BlobMap& /*inputs*/, const vpux::PreprocMap& /*preProcMap*/) {
    IE_THROW() << "Not implemented";
}
