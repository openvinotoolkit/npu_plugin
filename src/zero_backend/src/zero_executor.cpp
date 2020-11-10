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

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "zero_allocator.h"

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

void ZeroExecutor::hostMem::init(const ze_driver_handle_t h) {
    if (drh) THROW_IE_EXCEPTION << "hostMem::init double init!";
    if (!h) THROW_IE_EXCEPTION << "hostMem::init arg == 0!";
    drh = h;
}
void ZeroExecutor::hostMem::resize(const size_t size) {
    if (!drh) THROW_IE_EXCEPTION << "hostMem::resize not init!";
    if (!size) THROW_IE_EXCEPTION << "hostMem::resize size is 0";
    if (size != sz) {
        free();
        ze_host_mem_alloc_desc_t desc = {ZE_HOST_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_HOST_MEM_ALLOC_FLAG_DEFAULT};
        throwOnFail("zeDriverAllocHostMem", zeDriverAllocHostMem(drh, &desc, size, alignment, &mem));
        sz = size;
    }
}
void ZeroExecutor::hostMem::free() {
    if (mem) {
        throwOnFail("zeDriverFreeMem hostMem", zeDriverFreeMem(drh, mem));
        mem = nullptr;
        sz = 0;
    }
}

void ZeroExecutor::deviceMem::init(
    const ze_driver_handle_t drh_, const ze_device_handle_t deh_, const ze_command_list_handle_t clh_) {
    if (drh) THROW_IE_EXCEPTION << "deviceMem::init double init!";
    if (!drh_) THROW_IE_EXCEPTION << "deviceMem::init drh_ == 0!";
    if (!deh_) THROW_IE_EXCEPTION << "deviceMem::init deh_ == 0!";
    if (!clh_) THROW_IE_EXCEPTION << "deviceMem::init clh_ == 0!";
    drh = drh_;
    deh = deh_;
    clh = clh_;
}
void ZeroExecutor::deviceMem::resize(const size_t size) {
    if (!drh) THROW_IE_EXCEPTION << "deviceMem::resize not init!";
    if (!size) THROW_IE_EXCEPTION << "deviceMem::resize size is 0";
    if (size != sz) {
        free();
        ze_device_mem_alloc_desc_t desc = {
            ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT, 0};
        sz = size;
        throwOnFail("zeDriverAllocDeviceMem", zeDriverAllocDeviceMem(drh, &desc, size, alignment, deh, &mem));
    }
}
void ZeroExecutor::deviceMem::free() {
    if (mem) {
        throwOnFail("zeDriverFreeMem deviceMem", zeDriverFreeMem(drh, mem));
        mem = nullptr;
        sz = 0;
    }
}
void ZeroExecutor::deviceMem::copyFromImpl(const void* src) {
    throwOnFail(
        "zeCommandListAppendMemoryCopy copyFromHost", zeCommandListAppendMemoryCopy(clh, mem, src, sz, nullptr));
}
void ZeroExecutor::deviceMem::copyToImpl(void* dst) {
    throwOnFail("zeCommandListAppendMemoryCopy copyToHost", zeCommandListAppendMemoryCopy(clh, dst, mem, sz, nullptr));
}

void ZeroExecutor::graph_raii::init(const ze_device_handle_t& device_handle, hostMem& graphMemory) {
    ze_graph_desc_t graph_desc = {ZE_GRAPH_DESC_VERSION_CURRENT, ZE_GRAPH_FORMAT_NATIVE, graphMemory.size(),
        static_cast<uint8_t*>(graphMemory.data())};
    throwOnFail("zeGraphCreate", zeGraphCreate(device_handle, &graph_desc, &g));
}
void ZeroExecutor::graph_raii::getProperties(ze_graph_properties_t& property) {
    throwOnFail("zeGraphGetProperties", zeGraphGetProperties(g, &property));
}
void ZeroExecutor::graph_raii::getArgumentProperties(const uint32_t index, ze_graph_argument_properties_t& arg) {
    throwOnFail("zeGraphGetArgumentProperties", zeGraphGetArgumentProperties(g, index, &arg));
}
void ZeroExecutor::graph_raii::setArgumentValue(const uint32_t index, const void* data) {
    throwOnFail("zeGraphSetArgumentValue", zeGraphSetArgumentValue(g, index, data));
}
void ZeroExecutor::graph_raii::commandListAppendGraphInitialize(const ze_command_list_handle_t& list) {
    throwOnFail("zeCommandListAppendGraphInitialize", zeCommandListAppendGraphInitialize(list, g));
}
void ZeroExecutor::graph_raii::commandListAppendGraphExecute(const ze_command_list_handle_t& list) {
    throwOnFail("zeCommandListAppendGraphExecute", zeCommandListAppendGraphExecute(list, g));
}
ZeroExecutor::graph_raii::~graph_raii() {
    if (ZE_RESULT_SUCCESS != zeGraphDestroy(g)) {
        std::cerr << "Error in dtor: zeGraphDestroy failed" << std::endl;
    }
}

ZeroExecutor::ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle,
    const vpux::NetworkDescription::Ptr& networkDescription, const VPUXConfig& config)
    : _config(config),
      _logger(std::make_shared<vpu::Logger>("ZeroExecutor", _config.logLevel(), vpu::consoleOutput())),
      _driver_handle(driver_handle),
      _device_handle(device_handle) {
    // Create our command queue
    ze_command_queue_desc_t queue_desc = {ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT, ZE_COMMAND_QUEUE_FLAG_NONE,
        ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    throwOnFail("zeCommandQueueCreate", zeCommandQueueCreate(_device_handle, &queue_desc, &_command_queue_handle));

    // Create our command list
    ze_command_list_desc_t list_desc = {ZE_COMMAND_LIST_DESC_VERSION_CURRENT, ZE_COMMAND_LIST_FLAG_NONE};
    throwOnFail("zeCommandListCreate", zeCommandListCreate(_device_handle, &list_desc, &_command_list_handle));

    ze_fence_desc_t fence_desc = {ZE_FENCE_DESC_VERSION_CURRENT, ZE_FENCE_FLAG_NONE};
    throwOnFail("zeFenceCreate", zeFenceCreate(_command_queue_handle, &fence_desc, &_fence_handle));

    graphMemory.init(_driver_handle);
    graphMemory.copyFrom(networkDescription->getCompiledNetwork());

    _graph_handle.init(_device_handle, graphMemory);
    memory_init();
}

void ZeroExecutor::memory_init() {
    // Query the graph arguments
    _graph_handle.getProperties(_graph_properties);

    for (uint32_t index = 0; _graph_properties.numGraphArgs > index; ++index) {
        ze_graph_argument_properties_t cur_arg;
        _graph_handle.getArgumentProperties(index, cur_arg);

        deviceMem mem(_driver_handle, _device_handle, _command_list_handle);
        mem.resize(getSizeIOBytes(cur_arg));

        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == cur_arg.type) {
            inputs_map.emplace(
                std::make_pair(std::string(cur_arg.name), argumentDescriptor{std::move(mem), cur_arg, index}));
        } else {
            outputs_map.emplace(
                std::make_pair(std::string(cur_arg.name), argumentDescriptor{std::move(mem), cur_arg, index}));
        }
    }

    // Initialize the graph
    _graph_handle.commandListAppendGraphInitialize(_command_list_handle);
    commit();
}

void ZeroExecutor::commit() {
    // Close and execute the command list
    _logger->info("ZeroExecutor::commit");
    throwOnFail("\tzeCommandListClose", zeCommandListClose(_command_list_handle));
    throwOnFail("\tzeCommandQueueExecuteCommandLists",
        zeCommandQueueExecuteCommandLists(_command_queue_handle, 1, &_command_list_handle, _fence_handle));
    throwOnFail("\tzeFenceHostSynchronize", zeFenceHostSynchronize(_fence_handle, 0));
    throwOnFail("\tzeCommandListReset", zeCommandListReset(_command_list_handle));
    throwOnFail("\tzeFenceReset", zeFenceReset(_fence_handle));
}

void ZeroExecutor::push(const InferenceEngine::BlobMap& inputs) {
    _logger->info("ZeroExecutor::push started");

    // we should repack input args to zero-allocated host memory if it's regular memory
    std::vector<hostMem> hm;

    // we should copy through host-memory but it's a deferred operation so we should be carefull with lifetime of
    // objects (for KMB, on MTL will be shared memory) InferenceEngine::BlobMap& inputs -> (deferred)deviceMemory
    for (const auto& inferInput : inputs) {
        const std::string& name = inferInput.first;
        const InferenceEngine::Blob::Ptr& input = inferInput.second;

        argumentDescriptor& arg = mapArguments(inputs_map, name);
        if (!twoApiLayoutCouplingCheck(arg.info.layout, input->getTensorDesc().getLayout()))
            THROW_IE_EXCEPTION << "Layouts is different for push blobs";
        if (input->byteSize() != getSizeIOBytes(arg.info)) THROW_IE_EXCEPTION << "Sizes are different for push blobs";

        // we could use (base) Blob* here cause we just need check pointer value;
        // temporary LockedMemory (result of cbuffer) object be alive till the end of expression
        if (ZeroAllocator::isZeroPtr(input->cbuffer().as<const uint8_t*>())) {
            arg.memory.copyFrom(input);
        } else {
            _logger->info("[Performance] inputs passed in non-zero api allocated memory. Memory repacking are performed");
            hm.emplace_back(_driver_handle);
            auto& currentHostMem = hm.back();
            currentHostMem.copyFrom(input);  // copy from regular to host memory
            arg.memory.copyFrom(currentHostMem);
        }

        _graph_handle.setArgumentValue(arg.idx, arg.memory.data());
    }

    // bind outputs to graph
    for (auto& cur_output : outputs_map) {
        argumentDescriptor& output = cur_output.second;
        _graph_handle.setArgumentValue(output.idx, output.memory.data());
    }
    commit();

    // run graph
    _graph_handle.commandListAppendGraphExecute(_command_list_handle);
    // Close and execute the command list
    commit();
    _logger->info("ZeroExecutor::push finished");
}

void ZeroExecutor::pull(InferenceEngine::BlobMap& outputs) {
    _logger->info("ZeroExecutor::pull started");

    // we should transfer output args from device memory to zero-allocated host memory if it's regular memory
    std::vector<hostMem> hostmemTransfer;
    std::vector<std::reference_wrapper<InferenceEngine::Blob::Ptr>> outputBlobTarget;

    // we should copy through host_memory but it's a deferred operation soo we should be carefull with lifetime of
    // objects (for KMB, on MTL will be shared memory)
    // copy from device memory to outputs(which should be allocated /w zero api)
    for (auto& inferOutput : outputs) {
        const auto& name = inferOutput.first;
        InferenceEngine::Blob::Ptr& output = inferOutput.second;

        argumentDescriptor& copyFromDeviceMem = mapArguments(outputs_map, name);
        if (!twoApiLayoutCouplingCheck(copyFromDeviceMem.info.layout, output->getTensorDesc().getLayout()))
            THROW_IE_EXCEPTION << "Layouts is different for pull blobs";
        if (output->byteSize() != getSizeIOBytes(copyFromDeviceMem.info))
            THROW_IE_EXCEPTION << "Sizes are different for pull blobs";

        // check comment in push method
        if (ZeroAllocator::isZeroPtr(output->cbuffer().as<const uint8_t*>())) {
            copyFromDeviceMem.memory.copyTo(output);
        } else {
            _logger->info("[Performance] output passed in non-zero api allocated memory. Memory repacking are performed");
            hostmemTransfer.emplace_back(_driver_handle);
            outputBlobTarget.emplace_back(output);

            copyFromDeviceMem.memory.copyTo(hostmemTransfer.back());
        }
    }
    commit();

    IE_ASSERT(hostmemTransfer.size() == outputBlobTarget.size());

    for (std::size_t i = 0; i < hostmemTransfer.size(); ++i) {
        hostmemTransfer[i].copyTo(outputBlobTarget[i]);
    }

    _logger->info("ZeroExecutor::pull finished");
}

InferenceEngine::Parameter ZeroExecutor::getParameter(const std::string&) const { return InferenceEngine::Parameter(); }
void ZeroExecutor::setup(const InferenceEngine::ParamMap&) { THROW_IE_EXCEPTION << "Not implemented"; }
bool ZeroExecutor::isPreProcessingSupported(const InferenceEngine::PreProcessInfo&) const { return false; }
std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> ZeroExecutor::getLayerStatistics() {
    THROW_IE_EXCEPTION << "Not implemented";
    return std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>();
}
void ZeroExecutor::push(const InferenceEngine::BlobMap& /*inputs*/, const vpux::PreprocMap& /*preProcMap*/) {
    THROW_IE_EXCEPTION << "Not implemented";
}
