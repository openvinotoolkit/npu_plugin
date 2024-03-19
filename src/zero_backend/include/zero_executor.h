//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"
#include "zero_wrappers.h"

#include <ze_api.h>
#include <ze_graph_ext.h>

namespace vpux {

class ZeroExecutor final : public Executor {
public:
    ZeroExecutor(ze_driver_handle_t driver, ze_device_handle_t device, ze_context_handle_t context,
                 ze_graph_dditable_ext_curr_t* graph_ddi_table_ext,
                 ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext,
                 const NetworkDescription::CPtr networkDescription, const Config& config,
                 const uint32_t& group_ordinal);

    ZeroExecutor(const ZeroExecutor&) = delete;
    ZeroExecutor(ZeroExecutor&&) = delete;
    ZeroExecutor& operator=(const ZeroExecutor&) = delete;
    ZeroExecutor& operator=(ZeroExecutor&&) = delete;
    ~ZeroExecutor();

    struct ArgumentDescriptor {
        ze_graph_argument_properties_t info;
        uint32_t idx;
    };

    void setArgumentValue(uint32_t argi_, const void* argv_) const;
    inline ze_graph_handle_t graph() const {
        return _graph;
    };
    inline ze_device_handle_t device() const {
        return _device;
    };
    inline ze_context_handle_t context() const {
        return _context;
    };
    inline ze_graph_dditable_ext_curr_t* graph_ddi_table_ext() const {
        return _graph_ddi_table_ext;
    };
    inline ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext() const {
        return _graph_profiling_ddi_table_ext;
    };
    inline const NetworkDescription& getNetworkDesc() const {
        return *_networkDesc.get();
    }
    inline const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& getCommandQueue() const {
        return _command_queues;
    }
    inline const uint32_t& get_group_ordinal() const {
        return _group_ordinal;
    };
    inline const std::unordered_map<std::string, ArgumentDescriptor>& inputs_desc_map() const {
        return _inputs_desc_map;
    };
    inline const std::unordered_map<std::string, ArgumentDescriptor>& outputs_desc_map() const {
        return _outputs_desc_map;
    };

private:
    const Config _config;
    Logger _logger;

    NetworkDescription::CPtr _networkDesc;

    ze_device_handle_t _device = nullptr;
    ze_context_handle_t _context = nullptr;
    ze_graph_dditable_ext_curr_t* _graph_ddi_table_ext = nullptr;
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;

    const uint32_t _group_ordinal;

    ze_graph_handle_t _graph = nullptr;
    ze_graph_properties_t _props{};
    std::unordered_map<std::string, ArgumentDescriptor> _inputs_desc_map;
    std::unordered_map<std::string, ArgumentDescriptor> _outputs_desc_map;

    std::array<std::shared_ptr<CommandQueue>, stage::COUNT> _command_queues;
};

}  // namespace vpux
