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

#include <ze_api.h>
#include <ze_graph_ext.h>

namespace vpux {

class ZeroExecutor final : public Executor {
protected:
    struct Graph;
    struct CommandQueue;
    enum stage {
        UPLOAD,
        EXECUTE,
        READBACK,

        COUNT
    };

public:
    ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle, ze_context_handle_t context,
                 ze_graph_dditable_ext_t* graph_ddi_table_ext, const vpux::NetworkDescription::Ptr& networkDescription,
                 const Config& config);

    ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle, ze_context_handle_t context,
                 ze_graph_dditable_ext_t* graph_ddi_table_ext, const vpux::NetworkDescription::Ptr& networkDescription,
                 const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& command_queue,
                 const std::shared_ptr<Graph>& graph, const Config& config);

    void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) override;
    void push(const InferenceEngine::BlobMap& inputs) override;
    void pull(InferenceEngine::BlobMap& outputs) override;

    // TODO: not implemented
    void setup(const InferenceEngine::ParamMap& params) override;
    bool isPreProcessingSupported(const PreprocMap& preProcessMap) const override;
    InferenceEngine::Parameter getParameter(const std::string& paramName) const override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;

    ZeroExecutor::Ptr clone() const override;

    struct Pipeline;
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
    struct HostMem {
        HostMem() = default;
        HostMem(const ze_driver_handle_t driver_handle, const ze_context_handle_t context, const size_t size);
        HostMem(const HostMem&) = delete;
        HostMem(HostMem&& other)
                : _size(other._size),
                  _data(other._data),
                  _driver_handle{other._driver_handle},
                  _context(other._context) {
            other._size = 0;
            other._data = nullptr;
        }
        HostMem& operator=(const HostMem&) = delete;
        HostMem& operator=(HostMem&&) = delete;

        const void* data() const {
            return _data;
        }
        void* data() {
            return _data;
        }
        size_t size() const {
            return _size;
        }
        template <typename T>
        void copyFrom(const std::vector<T>& vec) {
            const auto inSz = vec.size() * sizeof(T);
            if (inSz != _size)
                IE_THROW() << "HostMem::copyFrom sizes mismatch";
            if (0 != ie_memcpy(_data, _size, vec.data(), inSz))
                IE_THROW() << "HostMem::copyFrom::ie_memcpy return != 0";
        }
        void copyFrom(const InferenceEngine::Blob::Ptr& blob) {
            const InferenceEngine::MemoryBlob::CPtr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
            if (!mblob)
                IE_THROW() << "HostMem::copyFrom failing of casting blob to MemoryBlob";
            if (mblob->byteSize() != _size)
                IE_THROW() << "HostMem::copyFrom sizes mismatch";
            if (0 != ie_memcpy(_data, _size, mblob->rmap().as<const uint8_t*>(), mblob->byteSize()))
                IE_THROW() << "HostMem::copyFrom::ie_memcpy* return != 0";
        }
        void copyTo(InferenceEngine::Blob::Ptr& blob) const {
            InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
            if (!mblob)
                IE_THROW() << "HostMem::copyTo failing of casting blob to MemoryBlob";

            if (mblob->byteSize() != _size)
                IE_THROW() << "HostMem::copyTo sizes mismatch";
            if (0 != ie_memcpy(mblob->buffer().as<uint8_t*>(), mblob->byteSize(), _data, _size))
                IE_THROW() << "HostMem::copyTo::ie_memcpy return != 0";
        }
        ~HostMem();

    private:
        size_t _size = 0;
        void* _data = nullptr;
        const ze_driver_handle_t _driver_handle = nullptr;
        const ze_context_handle_t _context = nullptr;
        const static size_t _alignment = 4096;
    };

    struct DeviceMem {
        DeviceMem() = default;
        DeviceMem(const ze_driver_handle_t driver_handle, const ze_device_handle_t device_handle,
                  const ze_context_handle_t context, const size_t size);
        DeviceMem(const DeviceMem&) = delete;
        DeviceMem(DeviceMem&& other)
                : _size(other._size),
                  _data(other._data),
                  _driver_handle(other._driver_handle),
                  _context(other._context) {
            other._size = 0;
            other._data = nullptr;
        }
        DeviceMem& operator=(const DeviceMem&) = delete;
        DeviceMem& operator=(DeviceMem&&) = delete;

        const void* data() const {
            return _data;
        }
        void* data() {
            return _data;
        }
        size_t size() const {
            return _size;
        }
        ~DeviceMem();

    private:
        size_t _size = 0;
        void* _data = nullptr;
        const ze_driver_handle_t _driver_handle = nullptr;
        const ze_context_handle_t _context = nullptr;
        const static size_t _alignment = 4096;
    };

    struct CommandList {
        CommandList() = default;
        CommandList(const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
                    ze_graph_dditable_ext_t* graph_ddi_table_ext);
        CommandList(const CommandList&) = delete;
        CommandList& operator=(const CommandList&) = delete;
        void reset();
        void appendMemoryCopy(void* dst, const void* src, size_t size);
        void appendGraphInitialize(const ze_graph_handle_t& graph_handle);
        void appendGraphExecute(const ze_graph_handle_t& graph_handle);
        void appendBarrier();
        void close();
        ~CommandList();
        ze_command_list_handle_t _handle = nullptr;
        const ze_context_handle_t _context = nullptr;
        ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    };

    struct Fence {
        Fence() = default;
        Fence(const std::shared_ptr<CommandQueue>& command_queue);
        Fence(const Fence&) = delete;
        Fence& operator=(const Fence&) = delete;
        void reset();
        void hostSynchronize();
        ~Fence();
        ze_fence_handle_t _handle = nullptr;
    };

    struct CommandQueue {
        CommandQueue() = default;
        CommandQueue(const ze_device_handle_t& device_handle, const ze_context_handle_t& context);
        CommandQueue(const CommandQueue&) = delete;
        CommandQueue& operator=(const CommandQueue&) = delete;
        void executeCommandList(CommandList& command_list);
        void executeCommandList(CommandList& command_list, Fence& fence);
        ~CommandQueue();
        ze_command_queue_handle_t _handle = nullptr;
        ze_context_handle_t _context = nullptr;
    };

    struct EventPool {
        EventPool() = default;
        EventPool(ze_device_handle_t device_handle, const ze_context_handle_t& context, uint32_t event_count);
        EventPool(const EventPool&) = delete;
        EventPool& operator=(const EventPool&) = delete;

        ~EventPool() {
            zeEventPoolDestroy(_handle);
        }

        const uint32_t _event_count;
        ze_event_pool_handle_t _handle = nullptr;
    };

    struct Event {
        Event(ze_device_handle_t device_handle, const ze_context_handle_t& context,
              const ze_event_pool_handle_t& event_pool, uint32_t event_index);
        Event(const Event&) = delete;
        Event& operator=(const Event&) = delete;
        void AppendSignalEvent(CommandList& command_list);
        void AppendWaitOnEvent(CommandList& command_list);
        void AppendEventReset(CommandList& command_list);

        ~Event() {
            zeEventDestroy(_handle);
        }
        ze_device_handle_t _device_t = nullptr;
        ze_context_handle_t _context = nullptr;

        ze_event_handle_t _handle = nullptr;
    };

    struct ArgumentDescriptor {
        ze_graph_argument_properties_t info;
        uint32_t idx;
    };

    struct Graph {
        Graph(const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
              const NetworkDescription::CPtr networkDesc, ze_graph_dditable_ext_t* graph_ddi_table_ext);
        Graph(const Graph&) = delete;
        Graph& operator=(const Graph&) = delete;
        void init();
        void setArgumentValue(uint32_t argi_, const void* argv_) const;
        ~Graph();
        ze_graph_handle_t _handle = nullptr;
        ze_context_handle_t _context = nullptr;
        const std::vector<char>& _blob;
        ze_graph_properties_t _props{};
        std::map<std::string, ArgumentDescriptor> _inputs_desc_map;
        std::map<std::string, ArgumentDescriptor> _outputs_desc_map;
        std::shared_ptr<CommandQueue> _command_queue;
        CommandList _command_list;
        std::shared_ptr<Fence> _fence;

        ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    };

    struct Pipeline {
        Pipeline(const ze_driver_handle_t& driver_handle, const ze_device_handle_t& device_handle,
                 const ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                 const std::shared_ptr<Graph>& graph,
                 const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& command_queue);
        Pipeline(const Pipeline&) = delete;
        Pipeline& operator=(const Pipeline&) = delete;
        ~Pipeline() = default;

        std::map<std::string, HostMem> _inputs_host_mem_map;
        std::map<std::string, DeviceMem> _inputs_device_mem_map;
        std::map<std::string, HostMem> _outputs_host_mem_map;
        std::map<std::string, DeviceMem> _outputs_device_mem_map;

        std::array<CommandList, stage::COUNT> _command_list;
        std::array<Fence, stage::COUNT> _fence;

        EventPool _event_pool;
        std::array<Event, stage::COUNT> _event;
    };

private:
    const Config _config;
    Logger _logger;

    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;

    NetworkDescription::Ptr _networkDesc;

    std::shared_ptr<Graph> _graph;

    std::array<std::shared_ptr<CommandQueue>, stage::COUNT> _command_queue;

    std::unique_ptr<Pipeline> _pipeline;
};

bool isRepackingRequired(const InferenceEngine::TensorDesc& userTensorDesc,
                         const InferenceEngine::TensorDesc& deviceTensorDesc);

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
auto mapArguments(const Map& zero, const std::string& key) -> typename const Map::mapped_type& {
    for (auto& p : zero) {
        if (std::string::npos != p.first.find(key)) {
            return p.second;
        }
    }

    IE_THROW() << "mapArguments: fail to map";
}

}  // namespace vpux
