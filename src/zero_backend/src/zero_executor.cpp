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

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"

#include "vpux/utils/IE/blob.hpp"

#include <blob_factory.hpp>

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/plugin/profiling_parser.hpp"

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

ze_graph_argument_precision_t getZePrecision(const IE::Precision precision) {
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

ze_command_queue_priority_t toZeQueuePriority(const ov::hint::Priority& val) {
    switch (val) {
    case ov::hint::Priority::LOW:
        return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
    case ov::hint::Priority::MEDIUM:
        return ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    case ov::hint::Priority::HIGH:
        return ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
    default:
        IE_THROW() << "Incorrect queue priority.";
    }
}

size_t getSizeIOBytes(const ze_graph_argument_properties_t& argument) {
    size_t num_elements = 1;
    for (size_t i = 0; i < layoutCount(argument.deviceLayout); ++i) {
        num_elements *= argument.dims[i];
    }
    const size_t size_in_bytes = num_elements * precisionToSize(argument.devicePrecision);
    return size_in_bytes;
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
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::prepareInputForInference");
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
    const bool isPrecisionMatched = userPrecision == devicePrecision;
    const bool isLayoutMatched = userLayout == deviceLayout;
    if (isPrecisionMatched && isLayoutMatched) {
        IE_THROW() << "There is nothing to repack";
    }

    if (!isPrecisionMatched) {
        logger.info("Different precisions of user and device input blobs. Conversion required from {0} to {1}",
                    userPrecision, devicePrecision);
        if (!isLayoutMatched) {
            expectedInput = toPrecision(IE::as<IE::MemoryBlob>(expectedInput), devicePrecision, quantParam);
            logger.info("Different layouts of user and device input blobs. Conversion required from {0} to {1}",
                        userLayout, deviceLayout);
            toLayout(IE::as<IE::MemoryBlob>(expectedInput), deviceLayout, nullptr, destData);
        } else {
            toPrecision(IE::as<IE::MemoryBlob>(expectedInput), devicePrecision, quantParam, nullptr, destData);
        }
    } else if (!isLayoutMatched) {
        logger.info("Different layouts of user and device input blobs. Conversion required from {0} to {1}", userLayout,
                    deviceLayout);
        toLayout(IE::as<IE::MemoryBlob>(expectedInput), deviceLayout, nullptr, destData);
    }
}

void getOutputAfterInference(IE::Blob::Ptr& userOutput, const IE::TensorDesc& deviceTensorDesc, const void* srcData,
                             Logger logger) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::getOutputsAfterInference");
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
        logger.info("Different precisions of pull blobs. Conversion required");
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
        logger.info("Different layouts of pull blobs. Conversion required");
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

}  // namespace

namespace vpux {
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
}  // namespace vpux

ZeroExecutor::ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle,
                           ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                           const vpux::NetworkDescription::Ptr& networkDescription, const Config& config)
        : _config(config),
          _logger("ZeroExecutor", _config.get<LOG_LEVEL>()),
          _driver_handle(driver_handle),
          _device_handle(device_handle),
          _context(context),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _networkDesc(networkDescription),
          _graph(std::make_shared<Graph>(_device_handle, _context, _networkDesc, _graph_ddi_table_ext)),
          _command_queue{{std::make_shared<CommandQueue>(device_handle, context,
                                                         toZeQueuePriority(_config.get<MODEL_PRIORITY>())),
                          std::make_shared<CommandQueue>(device_handle, context,
                                                         toZeQueuePriority(_config.get<MODEL_PRIORITY>())),
                          std::make_shared<CommandQueue>(device_handle, context,
                                                         toZeQueuePriority(_config.get<MODEL_PRIORITY>()))}},
          _pipeline(std::make_unique<Pipeline>(device_handle, context, graph_ddi_table_ext, _graph, _command_queue)) {
    _graph->init();
}

ZeroExecutor::ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle,
                           ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                           const vpux::NetworkDescription::Ptr& networkDescription,
                           const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& command_queue,
                           const std::shared_ptr<Graph>& graph, const Config& config)
        : _config(config),
          _logger("ZeroExecutor", _config.get<LOG_LEVEL>()),
          _driver_handle(driver_handle),
          _device_handle(device_handle),
          _context(context),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _networkDesc(networkDescription),
          _graph(graph),
          _command_queue(command_queue),
          _pipeline(std::make_unique<Pipeline>(device_handle, context, graph_ddi_table_ext, graph, _command_queue)) {
}

ZeroExecutor::HostMem::HostMem(const ze_context_handle_t context, const size_t size): _size(size), _context(context) {
    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    throwOnFail("zeMemAllocHost", zeMemAllocHost(_context, &desc, _size, _alignment, &_data));
}
ZeroExecutor::HostMem& ZeroExecutor::HostMem::operator=(HostMem&& other) {
    free();
    _size = other._size;
    _data = other._data;
    _context = other._context;
    other._size = 0;
    other._data = nullptr;
    return *this;
}
void ZeroExecutor::HostMem::free() {
    if (_size) {
        _size = 0;
        throwOnFail("zeMemFree HostMem", zeMemFree(_context, _data));
        _data = nullptr;
    }
}
ZeroExecutor::HostMem::~HostMem() {
    free();
}

ZeroExecutor::DeviceMem::DeviceMem(const ze_device_handle_t deh, ze_context_handle_t context, const size_t size)
        : _size(size), _context(context) {
    ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};

    throwOnFail("zeMemAllocDevice", zeMemAllocDevice(_context, &desc, _size, _alignment, deh, &_data));
}
ZeroExecutor::DeviceMem& ZeroExecutor::DeviceMem::operator=(DeviceMem&& other) {
    free();
    _size = other._size;
    _data = other._data;
    _context = other._context;
    other._size = 0;
    other._data = nullptr;
    return *this;
}
void ZeroExecutor::DeviceMem::free() {
    if (0 != _size) {
        _size = 0;
        throwOnFail("zeMemFree DeviceMem", zeMemFree(_context, _data));
        _data = nullptr;
    }
}
ZeroExecutor::DeviceMem::~DeviceMem() {
    free();
}

void ZeroExecutor::MemoryManagementUnit::appendArgument(const std::string& name,
                                                        const ze_graph_argument_properties_t& argument) {
    _offsets.emplace(std::make_pair(name, _size));

    const std::size_t argSize = getSizeIOBytes(argument);
    _size += argSize + alignment -
             (argSize % alignment);  // is this really necessary? if 0==argSize%alignment -> add 1 * alignment
}

void ZeroExecutor::MemoryManagementUnit::allocate(const ze_device_handle_t device_handle,
                                                  const ze_context_handle_t context) {
    if (0 != _host.size())
        IE_THROW() << "Memory already allocated";
    if (0 == _size)
        IE_THROW() << "Can't allocate empty buffer";

    _host = HostMem(context, _size);
    _device = DeviceMem(device_handle, context, _size);
}
std::size_t ZeroExecutor::MemoryManagementUnit::getSize() const {
    return _size;
}
const void* ZeroExecutor::MemoryManagementUnit::getHostMemRegion() const {
    return _host.data();
}
const void* ZeroExecutor::MemoryManagementUnit::getDeviceMemRegion() const {
    return _device.data();
}
void* ZeroExecutor::MemoryManagementUnit::getHostMemRegion() {
    return _host.data();
}
void* ZeroExecutor::MemoryManagementUnit::getDeviceMemRegion() {
    return _device.data();
}
void* ZeroExecutor::MemoryManagementUnit::getHostPtr(const std::string& name) {
    uint8_t* from = static_cast<uint8_t*>(_host.data());
    if (!from)
        IE_THROW() << "Host memory not allocated yet";

    return mapArguments(_offsets, name) + from;
}
void* ZeroExecutor::MemoryManagementUnit::getDevicePtr(const std::string& name) {
    uint8_t* from = static_cast<uint8_t*>(_device.data());
    if (!from)
        IE_THROW() << "Device memory not allocated yet";

    return mapArguments(_offsets, name) + from;
}
bool ZeroExecutor::MemoryManagementUnit::checkHostPtr(const void* ptr) const {
    const uint8_t* from = static_cast<const uint8_t*>(_host.data());
    return (ptr >= from && (from + _size) > ptr);
}

ZeroExecutor::CommandList::CommandList(const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
                                       ze_graph_dditable_ext_t* graph_ddi_table_ext)
        : _context(context), _graph_ddi_table_ext(graph_ddi_table_ext) {
    ze_command_list_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
    throwOnFail("zeCommandListCreate", zeCommandListCreate(_context, device_handle, &desc, &_handle));
    reset();
}
void ZeroExecutor::CommandList::reset() {
    throwOnFail("zeCommandListReset", zeCommandListReset(_handle));
}
void ZeroExecutor::CommandList::appendMemoryCopy(void* dst, const void* src, size_t size) {
    throwOnFail("zeCommandListAppendMemoryCopy",
                zeCommandListAppendMemoryCopy(_handle, dst, src, size, nullptr, 0, nullptr));
}
void ZeroExecutor::CommandList::appendGraphInitialize(const ze_graph_handle_t& graph_handle) {
    throwOnFail("pfnAppendGraphInitialize",
                _graph_ddi_table_ext->pfnAppendGraphInitialize(_handle, graph_handle, nullptr, 0, nullptr));
}
void ZeroExecutor::CommandList::appendGraphExecute(const ze_graph_handle_t& graph_handle) {
    throwOnFail("pfnAppendGraphExecute",
                _graph_ddi_table_ext->pfnAppendGraphExecute(_handle, graph_handle, nullptr, nullptr, 0, nullptr));
}
// TODO This is a work-around due to bug on ARM side
// ARM sends signal before all necessary copying operations are completed
// Should be removed when the bug is resolved
// [Track number: E#13355]
// [Track number: E#16690]
void ZeroExecutor::CommandList::appendBarrier() {
    throwOnFail("zeCommandListAppendBarrier", zeCommandListAppendBarrier(_handle, nullptr, 0, nullptr));
}
void ZeroExecutor::CommandList::close() {
    throwOnFail("zeCommandListClose", zeCommandListClose(_handle));
}
ZeroExecutor::CommandList::~CommandList() {
    throwOnFail("zeCommandListDestroy", zeCommandListDestroy(_handle));
}

ZeroExecutor::Graph::Graph(const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
                           const NetworkDescription::CPtr networkDesc, ze_graph_dditable_ext_t* graph_ddi_table_ext)
        : _context(context),
          _blob(networkDesc->getCompiledNetwork()),
          _command_queue(std::make_shared<CommandQueue>(device_handle, _context, ZE_COMMAND_QUEUE_PRIORITY_NORMAL)),
          _command_list(device_handle, _context, graph_ddi_table_ext),
          _fence(std::make_shared<Fence>(_command_queue)),
          _graph_ddi_table_ext(graph_ddi_table_ext) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::Graph::Graph");
    ze_graph_desc_t desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,        nullptr, ZE_GRAPH_FORMAT_NATIVE, _blob.size(),
                         reinterpret_cast<const uint8_t*>(_blob.data()), nullptr};
    throwOnFail("pfnCreate", _graph_ddi_table_ext->pfnCreate(_context, device_handle, &desc, &_handle));

    throwOnFail("pfnGetProperties", _graph_ddi_table_ext->pfnGetProperties(_handle, &_props));
    for (uint32_t index = 0; index < _props.numGraphArgs; ++index) {
        ze_graph_argument_properties_t arg;
        throwOnFail("pfnGetArgumentProperties", _graph_ddi_table_ext->pfnGetArgumentProperties(_handle, index, &arg));
        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            _inputs_desc_map.emplace(std::make_pair(std::string(arg.name), ArgumentDescriptor{arg, index}));
        } else {
            _outputs_desc_map.emplace(std::make_pair(std::string(arg.name), ArgumentDescriptor{arg, index}));
        }
    }
    _command_list.appendGraphInitialize(_handle);
    _command_list.close();
}
void ZeroExecutor::Graph::init() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::Graph::init");
    _command_queue->executeCommandList(_command_list, *_fence);
    _fence->hostSynchronize();
}
void ZeroExecutor::Graph::setArgumentValue(uint32_t argi_, const void* argv_) const {
    throwOnFail("zeGraphSetArgumentValue", _graph_ddi_table_ext->pfnSetArgumentValue(_handle, argi_, argv_));
}
ZeroExecutor::Graph::~Graph() {
    throwOnFail("pfnDestroy", _graph_ddi_table_ext->pfnDestroy(_handle));
}

ZeroExecutor::CommandQueue::CommandQueue(const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
                                         const ze_command_queue_priority_t& priority)
        : _context(context) {
    ze_command_queue_desc_t queue_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
                                          ZE_COMMAND_QUEUE_MODE_DEFAULT,        priority};
    throwOnFail("zeCommandQueueCreate", zeCommandQueueCreate(_context, device_handle, &queue_desc, &_handle));
}
void ZeroExecutor::CommandQueue::executeCommandList(CommandList& command_list) {
    throwOnFail("zeCommandQueueExecuteCommandLists",
                zeCommandQueueExecuteCommandLists(_handle, 1, &command_list._handle, nullptr));
}
void ZeroExecutor::CommandQueue::executeCommandList(CommandList& command_list, Fence& fence) {
    throwOnFail("zeCommandQueueExecuteCommandLists",
                zeCommandQueueExecuteCommandLists(_handle, 1, &command_list._handle, fence._handle));
}
ZeroExecutor::CommandQueue::~CommandQueue() {
    throwOnFail("zeCommandQueueDestroy", zeCommandQueueDestroy(_handle));
}

ZeroExecutor::Pipeline::Pipeline(const ze_device_handle_t& device_handle, const ze_context_handle_t context,
                                 ze_graph_dditable_ext_t* graph_ddi_table_ext, const std::shared_ptr<Graph>& graph,
                                 const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& command_queue)
        : _command_list{{{device_handle, context, graph_ddi_table_ext},
                         {device_handle, context, graph_ddi_table_ext},
                         {device_handle, context, graph_ddi_table_ext}}},
          _fence{{{command_queue[stage::UPLOAD]}, {command_queue[stage::EXECUTE]}, {command_queue[stage::READBACK]}}},
          _event_pool(device_handle, context, stage::COUNT),
          _event{{{device_handle, context, _event_pool._handle, stage::UPLOAD},
                  {device_handle, context, _event_pool._handle, stage::EXECUTE},
                  {device_handle, context, _event_pool._handle, stage::READBACK}}} {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::Pipeline::Pipeline");
    for (const auto& desc : graph->_inputs_desc_map) {
        _inputs.appendArgument(desc.first, desc.second.info);
    }
    _inputs.allocate(device_handle, context);
    _command_list[stage::UPLOAD].appendMemoryCopy(_inputs.getDeviceMemRegion(), _inputs.getHostMemRegion(),
                                                  _inputs.getSize());
    for (const auto& desc : graph->_inputs_desc_map) {
        graph->setArgumentValue(desc.second.idx, _inputs.getDevicePtr(desc.first));
    }

    _command_list[stage::UPLOAD].appendBarrier();
    _event[stage::UPLOAD].AppendSignalEvent(_command_list[stage::UPLOAD]);

    for (const auto& desc : graph->_outputs_desc_map) {
        _outputs.appendArgument(desc.first, desc.second.info);
    }
    _outputs.allocate(device_handle, context);
    _command_list[stage::READBACK].appendMemoryCopy(_outputs.getHostMemRegion(), _outputs.getDeviceMemRegion(),
                                                    _outputs.getSize());
    for (const auto& desc : graph->_outputs_desc_map) {
        graph->setArgumentValue(desc.second.idx, _outputs.getDevicePtr(desc.first));
    }

    _event[stage::UPLOAD].AppendWaitOnEvent(_command_list[stage::EXECUTE]);
    _command_list[stage::EXECUTE].appendGraphExecute(graph->_handle);

    _event[stage::UPLOAD].AppendEventReset(_command_list[stage::READBACK]);

    for (auto& commandList : _command_list) {
        commandList.close();
    }
}

ZeroExecutor::Fence::Fence(const std::shared_ptr<CommandQueue>& command_queue) {
    ze_fence_desc_t fence_desc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
    throwOnFail("zeFenceCreate", zeFenceCreate(command_queue->_handle, &fence_desc, &_handle));
}
void ZeroExecutor::Fence::reset() {
    throwOnFail("zeFenceReset", zeFenceReset(_handle));
}
void ZeroExecutor::Fence::hostSynchronize() {
    throwOnFail("zeFenceHostSynchronize", zeFenceHostSynchronize(_handle, UINT64_MAX));
}
ZeroExecutor::Fence::~Fence() {
    throwOnFail("zeFenceDestroy", zeFenceDestroy(_handle));
}

ZeroExecutor::EventPool::EventPool(ze_device_handle_t device_handle, const ze_context_handle_t& context,
                                   uint32_t event_count)
        : _event_count(event_count) {
    ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
                                            event_count};
    throwOnFail("zeEventPoolCreate", zeEventPoolCreate(context, &event_pool_desc, 1, &device_handle, &_handle));
}
ZeroExecutor::Event::Event(ze_device_handle_t device_handle, const ze_context_handle_t& context,
                           const ze_event_pool_handle_t& event_pool, uint32_t event_index)
        : _device_t(device_handle), _context(context) {
    ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, event_index, 0, 0};
    throwOnFail("zeEventCreate", zeEventCreate(event_pool, &event_desc, &_handle));
}
void ZeroExecutor::Event::AppendSignalEvent(CommandList& command_list) {
    throwOnFail("zeCommandListAppendSignalEvent", zeCommandListAppendSignalEvent(command_list._handle, _handle));
}
void ZeroExecutor::Event::AppendWaitOnEvent(CommandList& command_list) {
    throwOnFail("zeCommandListAppendWaitOnEvents", zeCommandListAppendWaitOnEvents(command_list._handle, 1, &_handle));
}
void ZeroExecutor::Event::AppendEventReset(CommandList& command_list) {
    throwOnFail("zeCommandListAppendEventReset", zeCommandListAppendEventReset(command_list._handle, _handle));
}

void ZeroExecutor::push(const IE::BlobMap& inputs) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::push");
    _logger.info("ZeroExecutor::push started");
    const auto& deviceInputs = _networkDesc->getDeviceInputsInfo();
    const auto quantParamsInfo = _networkDesc->getQuantParamsInfo();
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_PUSH, itt::domains::LevelZeroBackend, "Executor::push", "PrepareInput");
    // Copy input data to staging buffer on Cpu (input always first argument)
    for (const auto& inferInput : inputs) {
        const auto& name = inferInput.first;
        const IE::Blob::Ptr& input = inferInput.second;

        const auto& desc = mapArguments(_graph->_inputs_desc_map, name);
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
        if (desc.info.devicePrecision != getZePrecision(deviceInput->getPrecision())) {
            IE_THROW() << "Parsing error: precisions are different for push blobs";
        }

        if (isRepackingRequired(input->getTensorDesc(), deviceInput->getTensorDesc())) {
            if (!isRepackingPossible(input->getTensorDesc(), deviceInput->getTensorDesc())) {
                IE_THROW() << "Push blobs: repacking is not possible";
            }
            void* hostMem = _pipeline->_inputs.getHostPtr(name);
            prepareInputForInference(input, deviceInput->getTensorDesc(), hostMem, quantParams, _logger);
        } else {
            // we should check memory type: host memory or generic and copy if it's a generic
            const uint8_t* inputPtr = input->buffer().as<const uint8_t*>();
            if (!_pipeline->_inputs.checkHostPtr(inputPtr)) {
                void* hostMem = _pipeline->_inputs.getHostPtr(name);
                if (0 != ie_memcpy(hostMem, input->byteSize(), inputPtr, input->byteSize())) {
                    IE_THROW() << "memcpy error for push blob " << name;
                }
            }
        }
    }
    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_PUSH, "UPLOAD");
    // Dispatch command to copy input data from upload heap to default heap
    _command_queue[stage::UPLOAD]->executeCommandList(_pipeline->_command_list[stage::UPLOAD]);

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_PUSH, "EXECUTE");
    // Submit the command list for execute
    _command_queue[stage::EXECUTE]->executeCommandList(_pipeline->_command_list[stage::EXECUTE],
                                                       _pipeline->_fence[stage::EXECUTE]);
    OV_ITT_TASK_SKIP(ZERO_EXECUTOR_PUSH);
}

Executor::Ptr ZeroExecutor::clone() const {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::clone");
    return std::make_shared<ZeroExecutor>(_driver_handle, _device_handle, _context, _graph_ddi_table_ext, _networkDesc,
                                          _command_queue, _graph, _config);
}

void ZeroExecutor::pull(IE::BlobMap& outputs) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::pull");
    const auto& deviceOutputs = _networkDesc->getDeviceOutputsInfo();

    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_PULL, itt::domains::LevelZeroBackend, "Executor::pull", "EXECUTE");
    // Wait for execute to finish
    _pipeline->_fence[stage::EXECUTE].hostSynchronize();
    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_PULL, "READBACK");
    // Schedule the copy of outputs from zeDriverAllocDeviceMem to zeDriverAllocHostMem
    _command_queue[stage::READBACK]->executeCommandList(_pipeline->_command_list[stage::READBACK],
                                                        _pipeline->_fence[stage::READBACK]);
    // Wait for output copy to finish execution for _fence from the host, to make sure that data
    // is available in the hostMem buffer of the output
    _pipeline->_fence[stage::READBACK].hostSynchronize();
    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_PULL, "GetOutput");
    // Copy output data to staging buffer on Cpu (input always first argument)
    for (auto& inferOutput : outputs) {
        const auto& name = inferOutput.first;
        IE::Blob::Ptr& output = inferOutput.second;

        const auto& desc = mapArguments(_graph->_outputs_desc_map, name);
        const auto& deviceOutput = deviceOutputs.at(name);
        if (std::max(getNumDims(desc.info.dims), getNumDims(deviceOutput->getTensorDesc().getDims())) > 2) {
            if (!twoApiLayoutCouplingCheck(desc.info.deviceLayout, deviceOutput->getLayout())) {
                IE_THROW() << "Parsing error: layouts are different for pull blobs";
            }
        }
        if (desc.info.devicePrecision != getZePrecision(deviceOutput->getPrecision())) {
            IE_THROW() << "Parsing error: precisions are different for pull blobs";
        }

        if (isRepackingRequired(output->getTensorDesc(), deviceOutput->getTensorDesc())) {
            if (!isRepackingPossible(output->getTensorDesc(), deviceOutput->getTensorDesc())) {
                IE_THROW() << "Pull blobs: repacking is not possible";
            }
            const void* hostMem = _pipeline->_outputs.getHostPtr(name);
            getOutputAfterInference(output, deviceOutput->getTensorDesc(), hostMem, _logger);
        } else {
            // we should check memory type: host memory or generic and copy if it's a generic
            uint8_t* outputPtr = output->buffer().as<uint8_t*>();
            if (!_pipeline->_outputs.checkHostPtr(outputPtr)) {
                const void* hostMem = _pipeline->_outputs.getHostPtr(name);
                if (0 != ie_memcpy(outputPtr, output->byteSize(), hostMem, output->byteSize())) {
                    IE_THROW() << "memcpy error for pull blob " << name;
                }
            }
        }
    }

    // Reset the fence objects
    for (auto& fence : _pipeline->_fence) {
        fence.reset();
    }
    OV_ITT_TASK_SKIP(ZERO_EXECUTOR_PULL);
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

    // [Track number: E#26528]
    // Enable profiling in L0 backend

    return perfCounts;
}

void ZeroExecutor::push(const IE::BlobMap& /*inputs*/, const vpux::PreprocMap& /*preProcMap*/) {
    IE_THROW() << "Not implemented";
}
