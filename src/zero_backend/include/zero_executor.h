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
    struct Graph;

    ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle, ze_context_handle_t context,
                 ze_graph_dditable_ext_t* graph_ddi_table_ext,
                 ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext,
                 const vpux::NetworkDescription::Ptr& networkDescription, const Config& config);

    ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle, ze_context_handle_t context,
                 ze_graph_dditable_ext_t* graph_ddi_table_ext,
                 ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext,
                 const vpux::NetworkDescription::Ptr& networkDescription,
                 const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& command_queues,
                 const std::shared_ptr<Graph>& graph, const Config& config);

    ZeroExecutor(const ZeroExecutor&) = delete;
    ZeroExecutor(ZeroExecutor&&) = delete;
    ZeroExecutor& operator=(const ZeroExecutor&) = delete;
    ZeroExecutor& operator=(ZeroExecutor&&) = delete;

    ZeroExecutor::Ptr clone() const override;

    const std::shared_ptr<Graph>& graph() const {
        return _graph;
    }

    NetworkDescription& getNetworkDesc() {
        return *_networkDesc.get();
    }

    const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& getCommandQueue() const {
        return _command_queues;
    }

    ~ZeroExecutor() = default;

    struct ArgumentDescriptor {
        ze_graph_argument_properties_t info;
        uint32_t idx;
    };

    struct Graph {
        Graph() = delete;
        Graph(const Config& config, const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
              const NetworkDescription::CPtr networkDesc, ze_graph_dditable_ext_t* graph_ddi_table_ext,
              ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext);
        Graph(const Graph&) = delete;
        Graph(Graph&&) = delete;
        Graph& operator=(const Graph&) = delete;
        Graph& operator=(Graph&&) = delete;
        ~Graph();

        void init();
        void setArgumentValue(uint32_t argi_, const void* argv_) const;
        inline ze_graph_handle_t handle() const {
            return _handle;
        };
        inline ze_device_handle_t device() const {
            return _device;
        };
        inline ze_context_handle_t context() const {
            return _context;
        };
        inline ze_graph_dditable_ext_t* graph_ddi_table_ext() const {
            return _graph_ddi_table_ext;
        };
        inline ze_graph_profiling_dditable_ext_t* graph_profiling_ddi_table_ext() const {
            return _graph_profiling_ddi_table_ext;
        };

        inline const std::map<std::string, ArgumentDescriptor>& inputs_desc_map() const {
            return _inputs_desc_map;
        };
        inline const std::map<std::string, ArgumentDescriptor>& outputs_desc_map() const {
            return _outputs_desc_map;
        };
        inline const std::vector<char>& blob() const {
            return _blob;
        };

    private:
        const Config _config;
        Logger _logger;

        ze_device_handle_t _device = nullptr;
        ze_context_handle_t _context = nullptr;
        const std::vector<char>& _blob;
        ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
        ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;

        ze_graph_handle_t _handle = nullptr;
        ze_graph_properties_t _props{};
        std::map<std::string, ArgumentDescriptor> _inputs_desc_map;
        std::map<std::string, ArgumentDescriptor> _outputs_desc_map;

        std::unique_ptr<CommandList> _command_list;
    };

private:
    const Config _config;
    Logger _logger;

    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;

    NetworkDescription::Ptr _networkDesc;

    std::shared_ptr<Graph> _graph;

    std::array<std::shared_ptr<CommandQueue>, stage::COUNT> _command_queues;
};

}  // namespace vpux
