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

namespace IE = InferenceEngine;

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

_ze_graph_argument_precision_t getZePrecision(const IE::Precision precision) {
    switch (precision) {
    case IE::Precision::I8:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT8;
    case IE::Precision::U8:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT8;
    case IE::Precision::I16:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT16;
    case IE::Precision::U16:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT16;
    case IE::Precision::I32:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT32;
    case IE::Precision::FP16:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP16;
    case IE::Precision::FP32:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP32;
    case IE::Precision::BIN:
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
auto mapArguments(Map& zero, const std::string& key, const std::size_t pos) -> typename Map::mapped_type& {
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

template <typename T>
size_t getNumDims(const T& dims) {
    return std::count_if(std::begin(dims), std::end(dims), [](const size_t& dim) -> bool {
        return (dim > 1);
    });
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
                              std::shared_ptr<vpu::Logger>& logger) {
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

    IE::Blob::Ptr expectedInput = userInput;
    if (userPrecision != devicePrecision) {
        logger->info("Different precisions of push blobs. Conversion required");
        expectedInput = toPrecision(IE::as<IE::MemoryBlob>(expectedInput), devicePrecision);
        if (expectedInput == nullptr) {
            IE_THROW() << "Blob data null pointer";
        }
    }
    if (userLayout != deviceLayout) {
        logger->info("Different layouts of push blobs. Conversion required");
        toLayout(IE::as<IE::MemoryBlob>(expectedInput), deviceLayout, nullptr, destData);
    }
}

void getOutputAfterInference(IE::Blob::Ptr& userOutput, const IE::TensorDesc& deviceTensorDesc, const void* srcData,
                             std::shared_ptr<vpu::Logger>& logger) {
    if (userOutput == nullptr) {
        IE_THROW() << "User output blob null pointer";
    }
    if (srcData == nullptr) {
        IE_THROW() << "Source data null pointer";
    }
    const auto userPrecision = userOutput->getTensorDesc().getPrecision();
    const auto userLayout = userOutput->getTensorDesc().getLayout();
    const auto devicePrecision = deviceTensorDesc.getPrecision();
    const auto deviceLayout = deviceTensorDesc.getLayout();

    // [OV design flaw] OV API make_blob_with_precision doesn't have any version with const source data
    IE::Blob::Ptr expectedOutput = makeBlob(deviceTensorDesc, nullptr, const_cast<void*>(srcData));
    if (userPrecision != devicePrecision) {
        logger->info("Different precisions of pull blobs. Conversion required");
        expectedOutput = toPrecision(IE::as<IE::MemoryBlob>(expectedOutput), userPrecision);
        if (expectedOutput == nullptr) {
            IE_THROW() << "Blob data null pointer";
        }
    }
    if (userLayout != deviceLayout) {
        logger->info("Different layouts of pull blobs. Conversion required");
        expectedOutput = toLayout(IE::as<IE::MemoryBlob>(expectedOutput), userLayout);
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

}  // namespace

ZeroExecutor::ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle,
                           ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                           const vpux::NetworkDescription::Ptr& networkDescription, const VPUXConfig& config)
        : _config(config),
          _logger(std::make_shared<vpu::Logger>("ZeroExecutor", _config.logLevel(), vpu::consoleOutput())),
          _driver_handle(driver_handle),
          _device_handle(device_handle),
          _context(context),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _networkDesc(networkDescription),
          _graph(std::make_shared<graph>(_device_handle, _context, _networkDesc, _graph_ddi_table_ext)),
          _command_queue{{std::make_shared<commandQueue>(device_handle, context),
                          std::make_shared<commandQueue>(device_handle, context),
                          std::make_shared<commandQueue>(device_handle, context)}},
          _pipeline(std::make_unique<pipeline>(driver_handle, device_handle, context, graph_ddi_table_ext, _graph,
                                               _command_queue)) {
    _graph->init();
}

ZeroExecutor::ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle,
                           ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                           const vpux::NetworkDescription::Ptr& networkDescription,
                           const std::array<std::shared_ptr<commandQueue>, stage::COUNT>& command_queue,
                           const std::shared_ptr<graph>& graph, const VPUXConfig& config)
        : _config(config),
          _logger(std::make_shared<vpu::Logger>("ZeroExecutor", _config.logLevel(), vpu::consoleOutput())),
          _driver_handle(driver_handle),
          _device_handle(device_handle),
          _context(context),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _networkDesc(networkDescription),
          _graph(graph),
          _command_queue(command_queue),
          _pipeline(std::make_unique<pipeline>(driver_handle, device_handle, context, graph_ddi_table_ext, graph,
                                               _command_queue)) {
}

ZeroExecutor::hostMem::hostMem(const ze_driver_handle_t driver_handle, const ze_context_handle_t context,
                               const size_t size)
        : _size(size), _driver_handle(driver_handle), _context(context) {
    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    throwOnFail("zeMemAllocHost", zeMemAllocHost(_context, &desc, _size, _alignment, &_data));
}
ZeroExecutor::hostMem::~hostMem() {
    if (_data) {
        throwOnFail("zeMemFree hostMem", zeMemFree(_context, _data));
    }
}

ZeroExecutor::deviceMem::deviceMem(const ze_driver_handle_t driver_handle, const ze_device_handle_t deh_,
                                   ze_context_handle_t context, const size_t size)
        : _size(size), _driver_handle(driver_handle), _context(context) {
    ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};

    throwOnFail("zeMemAllocDevice", zeMemAllocDevice(_context, &desc, _size, _alignment, deh_, &_data));
}
ZeroExecutor::deviceMem::~deviceMem() {
    if (_data) {
        throwOnFail("zeMemFree deviceMem", zeMemFree(_context, _data));
    }
}

ZeroExecutor::commandList::commandList(const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
                                       ze_graph_dditable_ext_t* graph_ddi_table_ext)
        : _context(context), _graph_ddi_table_ext(graph_ddi_table_ext) {
    ze_command_list_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
    throwOnFail("zeCommandListCreate", zeCommandListCreate(_context, device_handle, &desc, &_handle));
    reset();
}
void ZeroExecutor::commandList::reset() {
    throwOnFail("zeCommandListReset", zeCommandListReset(_handle));
}
void ZeroExecutor::commandList::appendMemoryCopy(void* dst, const void* src, size_t size) {
    throwOnFail("zeCommandListAppendMemoryCopy",
                zeCommandListAppendMemoryCopy(_handle, dst, src, size, nullptr, 0, nullptr));
}
void ZeroExecutor::commandList::appendGraphInitialize(const ze_graph_handle_t& graph_handle) {
    throwOnFail("pfnAppendGraphInitialize", _graph_ddi_table_ext->pfnAppendGraphInitialize(_handle, graph_handle));
}
void ZeroExecutor::commandList::appendGraphExecute(const ze_graph_handle_t& graph_handle) {
    throwOnFail("pfnAppendGraphExecute", _graph_ddi_table_ext->pfnAppendGraphExecute(_handle, graph_handle));
}
// TODO This is a work-around due to bug on ARM side
// ARM sends signal before all necessary copying operations are completed
// Should be removed when the bug is resolved
// [Track number: E#13355]
// [Track number: E#16690]
void ZeroExecutor::commandList::appendBarrier() {
    throwOnFail("zeCommandListAppendBarrier", zeCommandListAppendBarrier(_handle, nullptr, 0, nullptr));
}
void ZeroExecutor::commandList::close() {
    throwOnFail("zeCommandListClose", zeCommandListClose(_handle));
}
ZeroExecutor::commandList::~commandList() {
    throwOnFail("zeCommandListDestroy", zeCommandListDestroy(_handle));
}

ZeroExecutor::graph::graph(const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
                           const NetworkDescription::CPtr networkDesc, ze_graph_dditable_ext_t* graph_ddi_table_ext)
        : _context(context),
          _blob(networkDesc->getCompiledNetwork()),
          _command_queue(std::make_shared<commandQueue>(device_handle, _context)),
          _command_list(device_handle, _context, graph_ddi_table_ext),
          _fence(std::make_shared<fence>(_command_queue)),
          _graph_ddi_table_ext(graph_ddi_table_ext) {
    ze_graph_desc_t desc = {ZE_GRAPH_FORMAT_NATIVE, _blob.size(), reinterpret_cast<const uint8_t*>(_blob.data())};
    throwOnFail("pfnCreate", _graph_ddi_table_ext->pfnCreate(device_handle, &desc, &_handle));

    throwOnFail("pfnGetProperties", _graph_ddi_table_ext->pfnGetProperties(_handle, &_props));
    for (uint32_t index = 0; index < _props.numGraphArgs; ++index) {
        ze_graph_argument_properties_t arg;
        throwOnFail("pfnGetArgumentProperties", _graph_ddi_table_ext->pfnGetArgumentProperties(_handle, index, &arg));
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
void ZeroExecutor::graph::init() {
    _command_queue->executeCommandList(_command_list, *_fence);
    _fence->hostSynchronize();
}
void ZeroExecutor::graph::setArgumentValue(uint32_t argi_, const void* argv_) const {
    throwOnFail("zeGraphSetArgumentValue", _graph_ddi_table_ext->pfnSetArgumentValue(_handle, argi_, argv_));
}
ZeroExecutor::graph::~graph() {
    throwOnFail("pfnDestroy", _graph_ddi_table_ext->pfnDestroy(_handle));
}

ZeroExecutor::commandQueue::commandQueue(const ze_device_handle_t& device_handle, const ze_context_handle_t& context)
        : _context(context) {
    ze_command_queue_desc_t queue_desc = {
            ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0, ZE_COMMAND_QUEUE_MODE_DEFAULT,
            ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    throwOnFail("zeCommandQueueCreate", zeCommandQueueCreate(_context, device_handle, &queue_desc, &_handle));
}
void ZeroExecutor::commandQueue::executeCommandList(commandList& command_list) {
    throwOnFail("zeCommandQueueExecuteCommandLists",
                zeCommandQueueExecuteCommandLists(_handle, 1, &command_list._handle, nullptr));
}
void ZeroExecutor::commandQueue::executeCommandList(commandList& command_list, fence& fence) {
    throwOnFail("zeCommandQueueExecuteCommandLists",
                zeCommandQueueExecuteCommandLists(_handle, 1, &command_list._handle, fence._handle));
}
ZeroExecutor::commandQueue::~commandQueue() {
    throwOnFail("zeCommandQueueDestroy", zeCommandQueueDestroy(_handle));
}

ZeroExecutor::pipeline::pipeline(const ze_driver_handle_t& driver_handle, const ze_device_handle_t& device_handle,
                                 const ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                                 const std::shared_ptr<graph>& graph,
                                 const std::array<std::shared_ptr<commandQueue>, stage::COUNT>& command_queue)
        : _command_list{{{device_handle, context, graph_ddi_table_ext},
                         {device_handle, context, graph_ddi_table_ext},
                         {device_handle, context, graph_ddi_table_ext}}},
          _fence{{{command_queue[stage::UPLOAD]}, {command_queue[stage::EXECUTE]}, {command_queue[stage::READBACK]}}},
          _event_pool(device_handle, context, stage::COUNT),
          _event{{{device_handle, context, _event_pool._handle, stage::UPLOAD},
                  {device_handle, context, _event_pool._handle, stage::EXECUTE},
                  {device_handle, context, _event_pool._handle, stage::READBACK}}} {
    for (const auto& desc : graph->_inputs_desc_map) {
        auto size = getSizeIOBytes(desc.second.info);
        _inputs_host_mem_map.emplace(std::make_pair(desc.first, hostMem{driver_handle, context, size}));
        _inputs_device_mem_map.emplace(
                std::make_pair(desc.first, deviceMem{driver_handle, device_handle, context, size}));

        const auto& hostMem = mapArguments(_inputs_host_mem_map, desc.first);
        auto& deviceMem = mapArguments(_inputs_device_mem_map, desc.first);
        _command_list[stage::UPLOAD].appendMemoryCopy(deviceMem.data(), hostMem.data(), size);

        graph->setArgumentValue(desc.second.idx, deviceMem.data());
    }
    _command_list[stage::UPLOAD].appendBarrier();
    _event[stage::UPLOAD].AppendSignalEvent(_command_list[stage::UPLOAD]);

    for (const auto& desc : graph->_outputs_desc_map) {
        const auto size = getSizeIOBytes(desc.second.info);
        _outputs_host_mem_map.emplace(std::make_pair(desc.first, hostMem{driver_handle, context, size}));
        _outputs_device_mem_map.emplace(
                std::make_pair(desc.first, deviceMem{driver_handle, device_handle, context, size}));

        auto& hostMem = mapArguments(_outputs_host_mem_map, desc.first);
        const auto& deviceMem = mapArguments(_outputs_device_mem_map, desc.first);
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

ZeroExecutor::pipeline::~pipeline() {
    zeEventPoolDestroy(_event_pool._handle);
}

ZeroExecutor::fence::fence(const std::shared_ptr<commandQueue>& command_queue) {
    ze_fence_desc_t fence_desc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
    throwOnFail("zeFenceCreate", zeFenceCreate(command_queue->_handle, &fence_desc, &_handle));
}
void ZeroExecutor::fence::reset() {
    throwOnFail("zeFenceReset", zeFenceReset(_handle));
}
void ZeroExecutor::fence::hostSynchronize() {
    throwOnFail("zeFenceHostSynchronize", zeFenceHostSynchronize(_handle, UINT64_MAX));
}
ZeroExecutor::fence::~fence() {
    throwOnFail("zeFenceDestroy", zeFenceDestroy(_handle));
}

ZeroExecutor::eventPool_t::eventPool_t(ze_device_handle_t device_handle, const ze_context_handle_t& context,
                                       uint32_t event_count)
        : _event_count(event_count) {
    ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
                                            event_count};
    throwOnFail("zeEventPoolCreate", zeEventPoolCreate(context, &event_pool_desc, 1, &device_handle, &_handle));
}
ZeroExecutor::event_t::event_t(ze_device_handle_t device_handle, const ze_context_handle_t& context,
                               const ze_event_pool_handle_t& event_pool, uint32_t event_index)
        : _device_t(device_handle), _context(context) {
    ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, event_index, 0, 0};
    throwOnFail("zeEventCreate", zeEventCreate(event_pool, &event_desc, &_handle));
}
void ZeroExecutor::event_t::AppendSignalEvent(commandList& command_list) {
    throwOnFail("zeCommandListAppendSignalEvent", zeCommandListAppendSignalEvent(command_list._handle, _handle));
}
void ZeroExecutor::event_t::AppendWaitOnEvent(commandList& command_list) {
    throwOnFail("zeCommandListAppendWaitOnEvents", zeCommandListAppendWaitOnEvents(command_list._handle, 1, &_handle));
}
void ZeroExecutor::event_t::AppendEventReset(commandList& command_list) {
    throwOnFail("zeCommandListAppendEventReset", zeCommandListAppendEventReset(command_list._handle, _handle));
}

void ZeroExecutor::push(const IE::BlobMap& inputs) {
    _logger->info("ZeroExecutor::push started");
    const auto& deviceInputs = _networkDesc->getDeviceInputsInfo();

    // Copy input data to staging buffer on Cpu (input always first argument)
    for (const auto& inferInput : inputs) {
        const auto& name = inferInput.first;
        const IE::Blob::Ptr& input = inferInput.second;

        const auto& desc = mapArguments(_graph->_inputs_desc_map, name);
        const auto& deviceInput = deviceInputs.at(name);
        // TODO Currently L0 and Plugin might return different layouts which have dims like [1,1...]
        // They might be reinterpreted in different ways, so this check has been added to prevent that behavior
        if (std::max(getNumDims(desc.info.dims), getNumDims(deviceInput->getTensorDesc().getDims())) > 2) {
            if (!twoApiLayoutCouplingCheck(desc.info.layout, deviceInput->getLayout())) {
                IE_THROW() << "Parsing error: layouts are different for push blobs";
            }
        }
        if (desc.info.precision != getZePrecision(deviceInput->getPrecision())) {
            IE_THROW() << "Parsing error: precisions are different for push blobs";
        }

        auto& hostMem = mapArguments(_pipeline->_inputs_host_mem_map, name);
        if (!isRepackingRequired(input->getTensorDesc(), deviceInput->getTensorDesc())) {
            hostMem.copyFrom(input);
        } else {
            if (!isRepackingPossible(input->getTensorDesc(), deviceInput->getTensorDesc())) {
                IE_THROW() << "Push blobs: repacking is not possible";
            }
            prepareInputForInference(input, deviceInput->getTensorDesc(), hostMem.data(), _logger);
        }
    }

    // Dispatch command to copy input data from upload heap to default heap
    _command_queue[stage::UPLOAD]->executeCommandList(_pipeline->_command_list[stage::UPLOAD]);

    // Submit the command list for execute
    _command_queue[stage::EXECUTE]->executeCommandList(_pipeline->_command_list[stage::EXECUTE],
                                                       _pipeline->_fence[stage::EXECUTE]);
}

Executor::Ptr ZeroExecutor::clone() const {
    return std::make_shared<ZeroExecutor>(_driver_handle, _device_handle, _context, _graph_ddi_table_ext, _networkDesc,
                                          _command_queue, _graph, _config);
}

void ZeroExecutor::pull(IE::BlobMap& outputs) {
    const auto& deviceOutputs = _networkDesc->getDeviceOutputsInfo();

    // Wait for execute to finish
    _pipeline->_fence[stage::EXECUTE].hostSynchronize();

    // Schedule the copy of outputs from zeDriverAllocDeviceMem to zeDriverAllocHostMem
    _command_queue[stage::READBACK]->executeCommandList(_pipeline->_command_list[stage::READBACK],
                                                        _pipeline->_fence[stage::READBACK]);
    // Wait for output copy to finish execution for _fence from the host, to make sure that data
    // is available in the hostMem buffer of the output
    _pipeline->_fence[stage::READBACK].hostSynchronize();

    // Copy output data to staging buffer on Cpu (input always first argument)
    for (auto& inferOutput : outputs) {
        const auto& name = inferOutput.first;
        IE::Blob::Ptr& output = inferOutput.second;

        const auto& desc = mapArguments(_graph->_outputs_desc_map, name);
        const auto& deviceOutput = deviceOutputs.at(name);
        if (std::max(getNumDims(desc.info.dims), getNumDims(deviceOutput->getTensorDesc().getDims())) > 2) {
            if (!twoApiLayoutCouplingCheck(desc.info.layout, deviceOutput->getLayout())) {
                IE_THROW() << "Parsing error: layouts are different for pull blobs";
            }
        }
        if (desc.info.precision != getZePrecision(deviceOutput->getPrecision())) {
            IE_THROW() << "Parsing error: precisions are different for pull blobs";
        }

        const auto& hostMem = mapArguments(_pipeline->_outputs_host_mem_map, name);
        if (!isRepackingRequired(output->getTensorDesc(), deviceOutput->getTensorDesc())) {
            hostMem.copyTo(output);
        } else {
            if (!isRepackingPossible(output->getTensorDesc(), deviceOutput->getTensorDesc())) {
                IE_THROW() << "Pull blobs: repacking is not possible";
            }
            getOutputAfterInference(output, deviceOutput->getTensorDesc(), hostMem.data(), _logger);
        }
    }

    // Reset the fence objects
    for (auto& fence : _pipeline->_fence) {
        fence.reset();
    }
}

IE::Parameter ZeroExecutor::getParameter(const std::string&) const {
    return IE::Parameter();
}
void ZeroExecutor::setup(const IE::ParamMap&) {
    IE_THROW() << "Not implemented";
}
bool ZeroExecutor::isPreProcessingSupported(const PreprocMap&) const {
    return false;
}

std::map<std::string, IE::InferenceEngineProfileInfo> ZeroExecutor::getLayerStatistics() {
    std::map<std::string, IE::InferenceEngineProfileInfo> perfCounts;

    const auto blob = _graph->_blob.data();
    auto profilingOutputBlob = _pipeline->_outputs_host_mem_map.find("profilingOutput");
    if (profilingOutputBlob == _pipeline->_outputs_host_mem_map.end()) {
        _logger->warning("No profiling output. Blob was compiled without profiling enabled or do not contain "
                         "profiling info.");
        return perfCounts;
    }

    std::vector<mv::utils::ProfInfo> deviceProfiling;
    mv::utils::getProfilingInfo(blob, profilingOutputBlob->second.data(), deviceProfiling);

    unsigned execution_index = 0;
    IE::InferenceEngineProfileInfo info;
    for (const auto& profilingEntry : deviceProfiling) {
        info.status = IE::InferenceEngineProfileInfo::EXECUTED;
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

void ZeroExecutor::push(const IE::BlobMap& /*inputs*/, const vpux::PreprocMap& /*preProcMap*/) {
    IE_THROW() << "Not implemented";
}
