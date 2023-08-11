//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "zero_executor.h"

#include "ze_api.h"
#include "zero_allocator.h"
#include "zero_device.h"
#include "zero_utils.h"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/al/config/runtime.hpp"

#include "vpux/utils/IE/itt.hpp"

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

using namespace vpux;

namespace IE = InferenceEngine;

ZeroExecutor::ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle,
                           ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                           ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext,
                           const vpux::NetworkDescription::Ptr& networkDescription, const Config& config)
        : _config(config),
          _logger("ZeroExecutor", _config.get<LOG_LEVEL>()),
          _driver_handle(driver_handle),
          _device_handle(device_handle),
          _context(context),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _graph_profiling_ddi_table_ext(graph_profiling_ddi_table_ext),
          _networkDesc(networkDescription),
          _graph(std::make_shared<Graph>(_config, _device_handle, _context, _networkDesc, _graph_ddi_table_ext,
                                         graph_profiling_ddi_table_ext)),
          _command_queues{
                  {std::make_shared<CommandQueue>(device_handle, context,
                                                  zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()), _config),
                   std::make_shared<CommandQueue>(device_handle, context,
                                                  zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()), _config),
                   std::make_shared<CommandQueue>(device_handle, context,
                                                  zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                  _config)}} {
    _graph->init();
}

ZeroExecutor::ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle,
                           ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                           ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext,
                           const vpux::NetworkDescription::Ptr& networkDescription,
                           const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& command_queues,
                           const std::shared_ptr<Graph>& graph, const Config& config)
        : _config(config),
          _logger("ZeroExecutor", _config.get<LOG_LEVEL>()),
          _driver_handle(driver_handle),
          _device_handle(device_handle),
          _context(context),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _graph_profiling_ddi_table_ext(graph_profiling_ddi_table_ext),
          _networkDesc(networkDescription),
          _graph(graph),
          _command_queues{command_queues} {
}

ZeroExecutor::Graph::Graph(const Config& config, const ze_device_handle_t& device_handle,
                           const ze_context_handle_t& context, const NetworkDescription::CPtr networkDesc,
                           ze_graph_dditable_ext_t* graph_ddi_table_ext,
                           ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext)
        : _config(config),
          _logger("Graph", _config.get<LOG_LEVEL>()),
          _device(device_handle),
          _context(context),
          _blob(networkDesc->getCompiledNetwork()),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _graph_profiling_ddi_table_ext(graph_profiling_ddi_table_ext),
          _command_list(std::make_unique<CommandList>(device_handle, _context, graph_ddi_table_ext, _config)) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::Graph::Graph");
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_GRAPH, itt::domains::LevelZeroBackend, "Executor::Graph::Graph", "pfnCreate");
    ze_graph_desc_t desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,        nullptr, ZE_GRAPH_FORMAT_NATIVE, _blob.size(),
                         reinterpret_cast<const uint8_t*>(_blob.data()), nullptr};
    zeroUtils::throwOnFail("pfnCreate", _graph_ddi_table_ext->pfnCreate(_context, device_handle, &desc, &_handle));

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGetProperties");
    zeroUtils::throwOnFail("pfnGetProperties", _graph_ddi_table_ext->pfnGetProperties(_handle, &_props));

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGetArgumentProperties");
    for (uint32_t index = 0; index < _props.numGraphArgs; ++index) {
        ze_graph_argument_properties_t arg;
        zeroUtils::throwOnFail("pfnGetArgumentProperties",
                               _graph_ddi_table_ext->pfnGetArgumentProperties(_handle, index, &arg));
        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            _inputs_desc_map.emplace(std::make_pair(std::string(arg.name), ArgumentDescriptor{arg, index}));
        } else {
            _outputs_desc_map.emplace(std::make_pair(std::string(arg.name), ArgumentDescriptor{arg, index}));
        }
    }
    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "appendGraphInitialize");
    _command_list->appendGraphInitialize(_handle);
    _command_list->close();
}
void ZeroExecutor::Graph::init() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::Graph::init");

    CommandQueue command_queue(_device, _context, ZE_COMMAND_QUEUE_PRIORITY_NORMAL, _config);
    Fence fence(command_queue, _config);

    OV_ITT_TASK_CHAIN(QUEUE_EXECUTE, itt::domains::LevelZeroBackend, "Executor::Graph::init", "queue_execute");
    command_queue.executeCommandList(*_command_list, fence);
    fence.hostSynchronize();
}
void ZeroExecutor::Graph::setArgumentValue(uint32_t argi_, const void* argv_) const {
    zeroUtils::throwOnFail("zeGraphSetArgumentValue", _graph_ddi_table_ext->pfnSetArgumentValue(_handle, argi_, argv_));
}
ZeroExecutor::Graph::~Graph() {
    auto result = _graph_ddi_table_ext->pfnDestroy(_handle);
    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("_graph_ddi_table_ext->pfnDestroy failed {0:X+}", uint64_t(result));
    }
}
Executor::Ptr ZeroExecutor::clone() const {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::clone");
    return std::make_shared<ZeroExecutor>(_driver_handle, _device_handle, _context, _graph_ddi_table_ext,
                                          _graph_profiling_ddi_table_ext, _networkDesc, _command_queues, _graph,
                                          _config);
}
