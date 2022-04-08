//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ie_memcpy.h>

#include <cstring>  // std::memcpy for pointer-only args
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"
#include "zero_memory.h"
#include "zero_profiling.h"
#include "zero_utils.h"

#include <ze_api.h>
#include <ze_graph_ext.h>

namespace vpux {

class ZeroExecutor final : public Executor {
    // NB: originally, it was protected as an implementation detail
    // made public for InferRequest to make accessible Pipeline and its details (HostMem)
    // protected:
public:
    struct Graph;
    struct CommandQueue;
    struct Pipeline;

    enum stage {
        UPLOAD,
        EXECUTE,
        READBACK,

        COUNT
    };

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

    void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) override;
    void push(const InferenceEngine::BlobMap& inputs) override;
    void pull(InferenceEngine::BlobMap& outputs) override;

    ZeroExecutor(const ZeroExecutor&) = delete;
    ZeroExecutor(ZeroExecutor&&) = delete;
    ZeroExecutor& operator=(const ZeroExecutor&) = delete;
    ZeroExecutor& operator=(ZeroExecutor&&) = delete;

    // TODO: not implemented
    void setup(const InferenceEngine::ParamMap& params) override;
    bool isPreProcessingSupported(const PreprocMap& preProcessMap) const override;
    InferenceEngine::Parameter getParameter(const std::string& paramName) const override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;

    ZeroExecutor::Ptr clone() const override;

    Pipeline& getPipeline() {
        return *_pipeline.get();
    }

    NetworkDescription& getNetworkDesc() {
        return *_networkDesc.get();
    }

    ~ZeroExecutor() = default;
    // NB: originally, it was protected as an implementation detail
    // made public for InferRequest to make accessible Pipeline and its details (HostMem)
    // protected:
    struct CommandList {
        friend class CommandQueue;
        CommandList() = delete;
        CommandList(const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
                    ze_graph_dditable_ext_t* graph_ddi_table_ext);
        CommandList(const CommandList&) = delete;
        CommandList(CommandList&&) = delete;
        CommandList& operator=(const CommandList&) = delete;
        CommandList& operator=(CommandList&&) = delete;

        void reset() const;
        void appendMemoryCopy(void* dst, const void* src, const std::size_t size) const;
        void appendGraphInitialize(const ze_graph_handle_t& graph_handle) const;
        void appendGraphExecute(const ze_graph_handle_t& graph_handle,
                                const ze_graph_profiling_query_handle_t& profiling_query_handle) const;
        void appendBarrier() const;
        void close() const;
        ~CommandList();

        inline ze_command_list_handle_t handle() const {
            return _handle;
        };

    private:
        ze_command_list_handle_t _handle = nullptr;
        const ze_context_handle_t _context = nullptr;
        ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    };

    struct Fence {
        Fence() = delete;
        Fence(const CommandQueue& command_queue);
        Fence(const Fence&) = delete;
        Fence(Fence&&) = delete;
        Fence& operator=(const Fence&) = delete;
        Fence& operator=(Fence&&) = delete;

        void reset() const;
        void hostSynchronize() const;
        ~Fence();
        inline ze_fence_handle_t handle() const {
            return _handle;
        };

    private:
        ze_fence_handle_t _handle = nullptr;
    };

    struct CommandQueue {
        CommandQueue() = delete;
        CommandQueue(const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
                     const ze_command_queue_priority_t& priority);
        CommandQueue(const CommandQueue&) = delete;
        CommandQueue(CommandQueue&&) = delete;
        CommandQueue& operator=(const CommandQueue&) = delete;
        CommandQueue& operator=(CommandQueue&&) = delete;

        void executeCommandList(CommandList& command_list) const;
        void executeCommandList(CommandList& command_list, Fence& fence) const;
        ~CommandQueue();
        inline ze_command_queue_handle_t handle() const {
            return _handle;
        };

    private:
        ze_command_queue_handle_t _handle = nullptr;
        ze_context_handle_t _context = nullptr;
    };

    struct EventPool {
        EventPool() = delete;
        EventPool(ze_device_handle_t device_handle, const ze_context_handle_t& context, uint32_t event_count);
        EventPool(const EventPool&) = delete;
        EventPool(EventPool&&) = delete;
        EventPool& operator=(const EventPool&) = delete;
        EventPool& operator=(EventPool&&) = delete;
        ~EventPool();
        inline ze_event_pool_handle_t handle() const {
            return _handle;
        };

    private:
        ze_event_pool_handle_t _handle = nullptr;
    };

    struct Event {
        Event() = delete;
        Event(ze_device_handle_t device_handle, const ze_context_handle_t& context,
              const ze_event_pool_handle_t& event_pool, uint32_t event_index);
        Event(const Event&) = delete;
        Event(Event&&) = delete;
        Event& operator=(const Event&) = delete;
        Event& operator=(Event&&) = delete;

        void AppendSignalEvent(CommandList& command_list) const;
        void AppendWaitOnEvent(CommandList& command_list);
        void AppendEventReset(CommandList& command_list) const;
        void hostSynchronize() const;
        void reset() const;
        ~Event();

    private:
        ze_device_handle_t _device_t = nullptr;
        ze_context_handle_t _context = nullptr;
        ze_event_handle_t _handle = nullptr;
    };

    struct ArgumentDescriptor {
        ze_graph_argument_properties_t info;
        uint32_t idx;
    };

    struct Graph {
        Graph() = delete;
        Graph(const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
              const NetworkDescription::CPtr networkDesc, ze_graph_dditable_ext_t* graph_ddi_table_ext);
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
        ze_device_handle_t _device = nullptr;
        ze_context_handle_t _context = nullptr;
        const std::vector<char>& _blob;
        ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;

        ze_graph_handle_t _handle = nullptr;
        ze_graph_properties_t _props{};
        std::map<std::string, ArgumentDescriptor> _inputs_desc_map;
        std::map<std::string, ArgumentDescriptor> _outputs_desc_map;

        std::unique_ptr<CommandList> _command_list;
    };

    struct Pipeline {
        Pipeline() = default;
        Pipeline(const Pipeline&) = delete;
        Pipeline(Pipeline&&) = delete;
        Pipeline& operator=(const Pipeline&) = delete;
        Pipeline& operator=(Pipeline&&) = delete;
        virtual ~Pipeline() = default;

        virtual void push() = 0;
        virtual void pull() = 0;
        virtual void reset() const = 0;

        inline zeroMemory::MemoryManagementUnit& inputs() {
            return _inputs;
        };
        inline zeroMemory::MemoryManagementUnit& outputs() {
            return _outputs;
        };

    protected:
        zeroMemory::MemoryManagementUnit _inputs;
        zeroMemory::MemoryManagementUnit _outputs;
    };

    struct DiscretePipeline final : public Pipeline {
        DiscretePipeline(const ze_device_handle_t& device_handle, const ze_context_handle_t context,
                         ze_graph_dditable_ext_t* graph_ddi_table_ext, const std::shared_ptr<Graph>& graph,
                         ze_graph_profiling_query_handle_t profiling_handle,
                         const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& command_queues);
        DiscretePipeline(const DiscretePipeline&) = delete;
        DiscretePipeline& operator=(const DiscretePipeline&) = delete;
        virtual ~DiscretePipeline() = default;

        void push() override;
        void pull() override;
        void reset() const override;

    private:
        const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& _command_queues;
        std::array<CommandList, stage::COUNT> _command_list;
        std::array<Fence, stage::COUNT> _fence;
        EventPool _event_pool;
        std::array<Event, stage::COUNT> _event;
    };

    struct IntegratedPipeline final : public Pipeline {
        IntegratedPipeline(const ze_device_handle_t& device_handle, const ze_context_handle_t context,
                           ze_graph_dditable_ext_t* graph_ddi_table_ext, const std::shared_ptr<Graph>& graph,
                           ze_graph_profiling_query_handle_t profiling_handle, CommandQueue& command_queue);
        IntegratedPipeline(const IntegratedPipeline&) = delete;
        IntegratedPipeline& operator=(const IntegratedPipeline&) = delete;
        virtual ~IntegratedPipeline() = default;

        void push() override;
        void pull() override;
        void reset() const override;

    private:
        CommandQueue& _command_queue;
        CommandList _command_list;
        Fence _fence;
        EventPool _event_pool;
        Event _event;
        bool sync_output_with_fences_ = false;
    };

private:
    std::unique_ptr<Pipeline> makePipeline();

    const Config _config;
    Logger _logger;

    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    ze_graph_profiling_dditable_ext_t* _graph_profiling_ddi_table_ext = nullptr;

    NetworkDescription::Ptr _networkDesc;

    std::shared_ptr<Graph> _graph;
    zeroProfiling::ProfilingPool _profiling_pool;
    zeroProfiling::ProfilingQuery _profiling_query;
    std::array<std::shared_ptr<CommandQueue>, stage::COUNT> _command_queues;
    std::unique_ptr<Pipeline> _pipeline;
};

bool isRepackingRequired(const InferenceEngine::TensorDesc& userTensorDesc,
                         const InferenceEngine::TensorDesc& deviceTensorDesc);
}  // namespace vpux
