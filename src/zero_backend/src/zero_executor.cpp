//
// Copyright 2020 Intel Corporation.
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

#include "zero_executor.h"

#include "zero_allocator.h"

#include "vpux/utils/IE/blob.hpp"

#include <blob_factory.hpp>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

using namespace vpux;

namespace {
void throwOnFail(const std::string& step, const ze_result_t result) {
    if (ZE_RESULT_SUCCESS != result) {
        THROW_IE_EXCEPTION << "throwOnFail: " << step << " result: 0x" << std::hex << uint64_t(result);
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
        THROW_IE_EXCEPTION << "precisionToSize switch->default reached";
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
        THROW_IE_EXCEPTION << "layoutCount switch->default reached";
    }
}

// check that ie Layout and zeroApi layout are the same for some argument
bool twoApiLayoutCouplingCheck(const ze_graph_argument_layout_t zeroL, const ::InferenceEngine::Layout ieL) {
    using namespace ::InferenceEngine;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_ANY == zeroL && ANY == ieL) return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_NCHW == zeroL && NCHW == ieL) return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_NHWC == zeroL && NHWC == ieL) return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW == zeroL && NCDHW == ieL) return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC == zeroL && NDHWC == ieL) return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_OIHW == zeroL && OIHW == ieL) return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_C == zeroL && C == ieL) return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_CHW == zeroL && CHW == ieL) return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_HW == zeroL && HW == ieL) return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_NC == zeroL && NC == ieL) return true;
    if (ZE_GRAPH_ARGUMENT_LAYOUT_CN == zeroL && CN == ieL) return true;

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
    THROW_IE_EXCEPTION << "mapArguments: fail to map";
}
}  // namespace

ZeroExecutor::hostMem::hostMem(const ze_driver_handle_t drh_, const size_t sz_)
    : _drh(drh_),
      _sz(sz_) {
    ze_host_mem_alloc_desc_t desc = { ZE_HOST_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_HOST_MEM_ALLOC_FLAG_DEFAULT };
    throwOnFail("zeDriverAllocHostMem", zeDriverAllocHostMem(_drh, &desc, _sz, _alignment, &_data));
}
ZeroExecutor::hostMem::~hostMem() {
    if (_data) {
        throwOnFail("zeDriverFreeMem hostMem", zeDriverFreeMem(_drh, _data));
    }
}

ZeroExecutor::deviceMem::deviceMem(const ze_driver_handle_t drh_, const ze_device_handle_t deh_, const size_t sz_)
    : _drh(drh_),
      _sz(sz_) {
    ze_device_mem_alloc_desc_t desc = {
        ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT, 0 };
    throwOnFail("zeDriverAllocDeviceMem", zeDriverAllocDeviceMem(_drh, &desc, _sz, _alignment, deh_, &_data));
}
ZeroExecutor::deviceMem::~deviceMem() {
    if (_data) {
        throwOnFail("zeDriverFreeMem deviceMem", zeDriverFreeMem(_drh, _data));
    }
}

ZeroExecutor::commandList::commandList(const ze_device_handle_t& deh_) {
    ze_command_list_desc_t desc = { ZE_COMMAND_LIST_DESC_VERSION_CURRENT, ZE_COMMAND_LIST_FLAG_NONE };
    throwOnFail("zeCommandListCreate", zeCommandListCreate(deh_, &desc, &_handle));
    reset();
}
void ZeroExecutor::commandList::reset() {
    throwOnFail("zeCommandListReset", zeCommandListReset(_handle));
}
void ZeroExecutor::commandList::appendMemoryCopy(void* dst, const void* src, size_t sz) {
    throwOnFail("zeCommandListAppendMemoryCopy", zeCommandListAppendMemoryCopy(_handle, dst, src, sz, nullptr));
}
void ZeroExecutor::commandList::appendGraphInitialize(const ze_graph_handle_t& graph_handle_) {
    throwOnFail("zeCommandListAppendGraphInitialize", zeCommandListAppendGraphInitialize(_handle, graph_handle_));
}
void ZeroExecutor::commandList::appendGraphExecute(const ze_graph_handle_t& graph_handle_) {
    throwOnFail("zeCommandListAppendGraphExecute", zeCommandListAppendGraphExecute(_handle, graph_handle_));
}
void ZeroExecutor::commandList::close() {
    throwOnFail("zeCommandListClose", zeCommandListClose(_handle));
}
ZeroExecutor::commandList::~commandList() {
    throwOnFail("zeCommandListDestroy", zeCommandListDestroy(_handle));
}

ZeroExecutor::fence::fence(const commandQueue& cq_) {
    ze_fence_desc_t desc = { ZE_FENCE_DESC_VERSION_CURRENT, ZE_FENCE_FLAG_NONE };
    throwOnFail("zeFenceCreate", zeFenceCreate(cq_._handle, &desc, &_handle));
}
void ZeroExecutor::fence::reset() {
    throwOnFail("zeFenceReset", zeFenceReset(_handle));
}
void ZeroExecutor::fence::hostSynchronize(uint32_t fence_value_) {
    throwOnFail("zeFenceHostSynchronize", zeFenceHostSynchronize(_handle, fence_value_));
}
void ZeroExecutor::fence::deviceSynchronize(const commandQueue& queue_, uint32_t fence_value_) {
    throwOnFail("zeFenceDeviceSynchronize", zeFenceDeviceSynchronize(queue_._handle, _handle, fence_value_));
}
void ZeroExecutor::fence::deviceSignal(uint32_t fence_value_) {
    throwOnFail("zeFenceDeviceSignal", zeFenceDeviceSignal(_handle, fence_value_));
}
ZeroExecutor::fence::~fence() {
    throwOnFail("zeFenceDestroy", zeFenceDestroy(_handle));
}

ZeroExecutor::commandQueue::commandQueue(const ze_device_handle_t& deh_) {
    ze_command_queue_desc_t desc = { ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT, ZE_COMMAND_QUEUE_FLAG_NONE,
        ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL };
    throwOnFail("zeCommandQueueCreate", zeCommandQueueCreate(deh_, &desc, &_handle));
}
void ZeroExecutor::commandQueue::executeCommandList(commandList& cl_) {
    throwOnFail("zeCommandQueueExecuteCommandLists",
                zeCommandQueueExecuteCommandLists(_handle, 1, &cl_._handle, nullptr));
}
ZeroExecutor::commandQueue::~commandQueue() {
    throwOnFail("zeCommandQueueDestroy", zeCommandQueueDestroy(_handle));
}

ZeroExecutor::graph::graph(const ze_driver_handle_t& drh_, const ze_device_handle_t& deh_,
                           const NetworkDescription::CPtr _networkDesc)
        : _mem(drh_, _networkDesc->getCompiledNetwork().size()),
          _command_queue(deh_),
          _command_list(deh_),
          _fence(_command_queue) {
    _mem.copyFrom(_networkDesc->getCompiledNetwork());

    ze_graph_desc_t desc = { ZE_GRAPH_DESC_VERSION_CURRENT, ZE_GRAPH_FORMAT_NATIVE,
        _mem.size(), static_cast<uint8_t*>(_mem.data()) };
    throwOnFail("zeGraphCreate", zeGraphCreate(deh_, &desc, &_handle));

    throwOnFail("zeGraphGetProperties", zeGraphGetProperties(_handle, &_props));
    for (uint32_t index = 0; index < _props.numGraphArgs; ++index)
    {
        ze_graph_argument_properties_t arg;
        throwOnFail("zeGraphGetArgumentProperties", zeGraphGetArgumentProperties(_handle, index, &arg));
        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type)
        {
            auto deviceInputs = _networkDesc->getDeviceInputsInfo();

            // [Track number: S#49808]
            // hack for correct memory allocation on device
            arg.precision = getZePrecision(deviceInputs.at(arg.name)->getPrecision());

            _inputs_desc_map.emplace(std::make_pair(std::string(arg.name), argumentDescriptor{ arg, index }));
        }
        else
        {
            _outputs_desc_map.emplace(std::make_pair(std::string(arg.name), argumentDescriptor{ arg, index }));
        }
    }

    _command_list.appendGraphInitialize(_handle);
    _command_list.close();
}
void ZeroExecutor::graph::init() {
    _command_queue.executeCommandList(_command_list);
    _fence.deviceSignal(1);
}
void ZeroExecutor::graph::setArgumentValue(uint32_t argi_, const void* argv_) const {
    throwOnFail("zeGraphSetArgumentValue", zeGraphSetArgumentValue(_handle, argi_, argv_));
}
ZeroExecutor::graph::~graph() {
    throwOnFail("zeGraphDestroy", zeGraphDestroy(_handle));
}

ZeroExecutor::pipeline::pipeline(const ze_driver_handle_t& drh_, const ze_device_handle_t& deh_,
    const std::array<commandQueue, stage::COUNT>& cq_, const graph& graph_)
    : _command_list{ deh_, deh_, deh_ } {
    for (const auto& desc : graph_._inputs_desc_map) {
        auto size = getSizeIOBytes(desc.second.info);
        _inputs_host_mem_map.try_emplace(desc.first, drh_, size);
        _inputs_device_mem_map.try_emplace(desc.first, drh_, deh_, size);

        auto& hostMem = mapArguments(_inputs_host_mem_map, desc.first);
        auto& deviceMem = mapArguments(_inputs_device_mem_map, desc.first);
        _command_list[stage::UPLOAD].appendMemoryCopy(
            deviceMem.data(), hostMem.data(), size);

        graph_.setArgumentValue(desc.second.idx, deviceMem.data());
    }
    _command_list[stage::UPLOAD].close();

    for (const auto& desc : graph_._outputs_desc_map) {
        auto size = getSizeIOBytes(desc.second.info);
        _outputs_host_mem_map.try_emplace(desc.first, drh_, size);
        _outputs_device_mem_map.try_emplace(desc.first, drh_, deh_, size);

        auto& hostMem = mapArguments(_outputs_host_mem_map, desc.first);
        auto& deviceMem = mapArguments(_outputs_device_mem_map, desc.first);
        _command_list[stage::READBACK].appendMemoryCopy(
            hostMem.data(), deviceMem.data(), size);

        graph_.setArgumentValue(desc.second.idx, deviceMem.data());
    }

    _command_list[stage::EXECUTE].appendGraphExecute(graph_._handle);

    for (auto& commandList: _command_list) {
        commandList.close();
    }
}
ZeroExecutor::pipeline::~pipeline() {

}

ZeroExecutor::ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle,
    const vpux::NetworkDescription::Ptr& networkDescription, const VPUXConfig& config)
    : _config(config),
      _logger(std::make_shared<vpu::Logger>("ZeroExecutor", _config.logLevel(), vpu::consoleOutput())),
      _driver_handle(driver_handle),
      _device_handle(device_handle),
      _graph(driver_handle, device_handle, networkDescription),
      _push_count(0),
      _pull_count(0),
      _networkDesc(networkDescription),
      _command_queue{ device_handle, device_handle, device_handle },
      _fence{ _command_queue[stage::UPLOAD], _command_queue[stage::EXECUTE], _command_queue[stage::READBACK] },
      _pipeline_depth(8) {
    for (uint32_t index = 0; index < _pipeline_depth; ++index) {
        _pipeline.emplace_back(std::make_unique<pipeline>(driver_handle, device_handle, _command_queue, _graph));
    }

    _graph.init();
}

static void prepareInputForInference(
        const InferenceEngine::Blob::Ptr& actualInput, const InferenceEngine::Precision& expectedPrecision, void* dest_data=nullptr) {
    if (actualInput == nullptr) {
        THROW_IE_EXCEPTION << "Actual input blob null pointer!";
    }
    if (actualInput->getTensorDesc().getPrecision() == expectedPrecision || dest_data == nullptr) {
        return;
    }

    vpux::toPrecision(InferenceEngine::as<InferenceEngine::MemoryBlob>(actualInput), expectedPrecision, nullptr, dest_data);
}

void ZeroExecutor::push(const InferenceEngine::BlobMap& inputs) {
    _logger->info("ZeroExecutor::push started");
    const auto& deviceInputs = _networkDesc->getDeviceInputsInfo();

    const auto iteration = _push_count++;
    const auto depth = iteration % _pipeline_depth;

    // Wait for input copy to finish for iteration - _pipeline_depth from the host, before overwriting the
    // hostMem buffer for the input
    if (iteration >= _pipeline_depth) {
        _fence[stage::UPLOAD].hostSynchronize(iteration - _pipeline_depth + 1);
    }

    // Copy the inputs from host to zeDriverAllocHostMem buffer set up for copy
    for (const auto& inferInput : inputs) {
        const std::string& name = inferInput.first;
        const InferenceEngine::Blob::Ptr& input = inferInput.second;

        auto& desc = mapArguments(_graph._inputs_desc_map, name);
        if (!twoApiLayoutCouplingCheck(desc.info.layout, input->getTensorDesc().getLayout()))
            THROW_IE_EXCEPTION << "Layouts is different for push blobs";
        if (input->byteSize() != getSizeIOBytes(desc.info)) {
            _logger->info("Sizes are different for push blobs. Need precision convert");
        }

        auto& hostMem = mapArguments(_pipeline[depth]->_inputs_host_mem_map, name);
        if (input->getTensorDesc().getPrecision() == deviceInputs.at(name)->getPrecision()) {
            hostMem.copyFrom(input);
        } else {
            prepareInputForInference(input, deviceInputs.at(name)->getPrecision(), hostMem.data());
        }
    }

    // Wait for execute to finish for iteration - _pipeline_depth from the upload command queue on device,
    // before overwriting the deviceMem buffer for input
    if (iteration >= _pipeline_depth) {
        _fence[stage::EXECUTE].deviceSynchronize(
            _command_queue[stage::UPLOAD], iteration - _pipeline_depth + 1);
    }

    // Schedule the copy of inputs from zeDriverAllocHostMem to zeDriverAllocDeviceMem
    _command_queue[stage::UPLOAD].executeCommandList(_pipeline[depth]->_command_list[stage::UPLOAD]);

    // Signal fence from device after input copy is executed from upload command queue on device
    _fence[stage::UPLOAD].deviceSignal(iteration + 1);

    // For the very first inference, wait for graph init to complete execution on the device
    if (iteration == 0) {
        _graph._fence.hostSynchronize(1);
    }

    // Wait for readback to finish for iteration - _pipeline_depth from the execute command queue on device,
    // before executing the inference and potentially overwriting the deviceMem buffer for output
    if (iteration >= _pipeline_depth) {
        _fence[stage::READBACK].deviceSynchronize(
            _command_queue[stage::EXECUTE], iteration - _pipeline_depth + 1);
    }

    // Wait for input copy to finish for iteration from the execute command queue on device,
    // before executing the inference to make sure that input data is available
    _fence[stage::UPLOAD].deviceSynchronize(_command_queue[stage::EXECUTE], iteration + 1);

    // Schedule the inference, wait for completion will be in matching pull
    _command_queue[stage::EXECUTE].executeCommandList(_pipeline[depth]->_command_list[stage::EXECUTE]);

    // Signal fence from device after inference is executed from execute command queue on device
    _fence[stage::EXECUTE].deviceSignal(iteration + 1);

    _logger->info("ZeroExecutor::push finished");
}

void ZeroExecutor::pull(InferenceEngine::BlobMap& outputs) {
    _logger->info("ZeroExecutor::pull started");

    const auto iteration = _pull_count++;
    const auto depth = iteration % _pipeline_depth;

    // Wait for inference to finish for _pull_count from the readback command queue on device,
    // to make sure that output data is available
    _fence[stage::EXECUTE].deviceSynchronize(_command_queue[READBACK], iteration + 1);

    // Schedule the copy of outputs from zeDriverAllocDeviceMem to zeDriverAllocHostMem
    _command_queue[stage::READBACK].executeCommandList(_pipeline[depth]->_command_list[stage::READBACK]);

    // Signal fence from device after output copy is completed from readback command queue on device
    _fence[stage::READBACK].deviceSignal(iteration + 1);

    // Wait for output copy to finish execution for _pull_count from the host, to make sure that data
    // is available in the hostMem buffer of the output
    _fence[stage::READBACK].hostSynchronize(iteration + 1);

    // Copy the outputs from set up zeDriverAllocHostMem to the host
    for (auto& inferOutput : outputs) {
        const auto& name = inferOutput.first;
        InferenceEngine::Blob::Ptr& output = inferOutput.second;

        auto& desc = mapArguments(_graph._outputs_desc_map, name);
        if (!twoApiLayoutCouplingCheck(desc.info.layout, output->getTensorDesc().getLayout()))
            THROW_IE_EXCEPTION << "Layouts is different for pull blobs";
        if (output->byteSize() != getSizeIOBytes(desc.info))
            THROW_IE_EXCEPTION << "Sizes are different for pull blobs";

        auto& hostMem = mapArguments(_pipeline[depth]->_outputs_host_mem_map, name);
        hostMem.copyTo(output);
    }

    _logger->info("ZeroExecutor::pull finished");
}

ZeroExecutor::~ZeroExecutor() {

}

InferenceEngine::Parameter ZeroExecutor::getParameter(const std::string&) const { return InferenceEngine::Parameter(); }
void ZeroExecutor::setup(const InferenceEngine::ParamMap&) { THROW_IE_EXCEPTION << "Not implemented"; }
bool ZeroExecutor::isPreProcessingSupported(const PreprocMap& preProcMap) const { return false; }
std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> ZeroExecutor::getLayerStatistics() {
    THROW_IE_EXCEPTION << "Not implemented";
    return std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>();
}
void ZeroExecutor::push(const InferenceEngine::BlobMap& /*inputs*/, const vpux::PreprocMap& /*preProcMap*/) {
    THROW_IE_EXCEPTION << "Not implemented";
}
