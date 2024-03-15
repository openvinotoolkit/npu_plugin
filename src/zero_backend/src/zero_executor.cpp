//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "zero_executor.h"

#include "ze_api.h"
#include "zero_device.h"
#include "zero_utils.h"

#include "vpux/al/config/common.hpp"

#include "vpux/utils/IE/itt.hpp"

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

using namespace vpux;

ZeroExecutor::ZeroExecutor(ze_driver_handle_t /*driver*/, ze_device_handle_t device, ze_context_handle_t context,
                           ze_graph_dditable_ext_curr_t* graph_ddi_table_ext,
                           ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext,
                           const NetworkDescription::CPtr networkDescription, const Config& config,
                           const uint32_t& group_ordinal)
        : _config(config),
          _logger("Graph", _config.get<LOG_LEVEL>()),
          _networkDesc(networkDescription),
          _device(device),
          _context(context),
          _graph_ddi_table_ext(graph_ddi_table_ext),
          _graph_profiling_ddi_table_ext(graph_profiling_ddi_table_ext),
          _group_ordinal(group_ordinal),
          _command_queues{{std::make_shared<CommandQueue>(_device, _context,
                                                          zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                          _config, group_ordinal),
                           std::make_shared<CommandQueue>(_device, _context,
                                                          zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                          _config, group_ordinal),
                           std::make_shared<CommandQueue>(_device, _context,
                                                          zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                          _config, group_ordinal)}} {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::ZeroExecutor");
    CommandList graph_command_list(_device, _context, graph_ddi_table_ext, _config, _group_ordinal);
    CommandQueue graph_command_queue(_device, _context, ZE_COMMAND_QUEUE_PRIORITY_NORMAL, _config, _group_ordinal);
    Fence fence(graph_command_queue, _config);
    ze_device_properties_t properties = {};
    zeroUtils::throwOnFail("zeDeviceGetProperties", zeDeviceGetProperties(_device, &properties));

    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_GRAPH, itt::domains::LevelZeroBackend, "Executor::ZeroExecutor", "graphCreate");
    ze_graph_desc_t desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                         nullptr,
                         ZE_GRAPH_FORMAT_NATIVE,
                         _networkDesc->getCompiledNetwork().size(),
                         reinterpret_cast<const uint8_t*>(_networkDesc->getCompiledNetwork().data()),
                         nullptr};
    zeroUtils::throwOnFail("pfnCreate", _graph_ddi_table_ext->pfnCreate(_context, _device, &desc, &_graph));

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGetProperties");
    zeroUtils::throwOnFail("pfnGetProperties", _graph_ddi_table_ext->pfnGetProperties(_graph, &_props));

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGetArgumentProperties");
    for (uint32_t index = 0; index < _props.numGraphArgs; ++index) {
        ze_graph_argument_properties_t arg;
        zeroUtils::throwOnFail("pfnGetArgumentProperties",
                               _graph_ddi_table_ext->pfnGetArgumentProperties(_graph, index, &arg));
        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            _inputs_desc_map.emplace(std::make_pair(std::string(arg.name), ArgumentDescriptor{arg, index}));
        } else {
            _outputs_desc_map.emplace(std::make_pair(std::string(arg.name), ArgumentDescriptor{arg, index}));
        }
    }
    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "appendGraphInitialize");
    graph_command_list.appendGraphInitialize(_graph);
    graph_command_list.close();

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "queue_execute");
    graph_command_queue.executeCommandList(graph_command_list, fence);
    fence.hostSynchronize();
}

void ZeroExecutor::setArgumentValue(uint32_t argi_, const void* argv_) const {
    zeroUtils::throwOnFail("zeGraphSetArgumentValue", _graph_ddi_table_ext->pfnSetArgumentValue(_graph, argi_, argv_));
}

ZeroExecutor::~ZeroExecutor() {
    auto result = _graph_ddi_table_ext->pfnDestroy(_graph);
    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("_graph_ddi_table_ext->pfnDestroy failed {0:X+}", uint64_t(result));
    }
}
