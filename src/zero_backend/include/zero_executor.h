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

#include <condition_variable>
#include <cstring>  // std::memcpy for pointer-only args
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vpux.hpp"
#include "ze_api.h"

#include "ze_fence_ext.h"
#include "ze_graph_ext.h"

#include "zero_config.h"
#include "zero_private_config.h"

namespace vpux {

class ZeroExecutorCommon : public Executor {
public:
    ZeroExecutorCommon(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle, ze_context_handle_t context,
                       ze_graph_dditable_ext_t* graph_ddi_table_ext, ze_fence_dditable_ext_t* fence_ddi_table_ext,
                       const vpux::NetworkDescription::Ptr& networkDescription, const ZeroConfig& config);

    void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) override;
    // TODO: not implemented
    void setup(const InferenceEngine::ParamMap& params) override;
    bool isPreProcessingSupported(const PreprocMap& preProcessMap) const override;
    InferenceEngine::Parameter getParameter(const std::string& paramName) const override;

    ~ZeroExecutorCommon() = default;

protected:
    struct hostMem {
        hostMem() = default;
        hostMem(const ze_driver_handle_t driver_handle, const ze_context_handle_t context, const size_t size);
        hostMem(const hostMem&) = delete;
        hostMem& operator=(const hostMem&) = delete;
        hostMem& operator=(hostMem&&) = delete;

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
                IE_THROW() << "hostMem::copyFrom sizes mismatch";
            if (0 != memcpy_s(_data, _size, vec.data(), inSz))
                IE_THROW() << "hostMem::copyFrom::ie_memcpy return != 0";
        }
        void copyFrom(const InferenceEngine::Blob::Ptr& blob) {
            const InferenceEngine::MemoryBlob::CPtr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
            if (!mblob)
                IE_THROW() << "deviceMem::copyFrom failing of casting blob to MemoryBlob";
            if (mblob->byteSize() != _size)
                IE_THROW() << "hostMem::copyFrom sizes mismatch";
            if (0 != memcpy_s(_data, _size, mblob->rmap().as<const uint8_t*>(), mblob->byteSize()))
                IE_THROW() << "hostMem::copyFrom::ie_memcpy* return != 0";
        }
        void copyTo(InferenceEngine::Blob::Ptr& blob) const {
            InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
            if (!mblob)
                IE_THROW() << "hostMem::copyTo failing of casting blob to MemoryBlob";

            if (mblob->byteSize() != _size)
                IE_THROW() << "hostMem::copyTo sizes mismatch";
            if (0 != memcpy_s(mblob->buffer().as<uint8_t*>(), mblob->byteSize(), _data, _size))
                IE_THROW() << "hostMem::copyTo::ie_memcpy return != 0";
        }
        ~hostMem();

    private:
        size_t _size = 0;
        void* _data = nullptr;
        const ze_driver_handle_t _driver_handle = nullptr;
        const ze_context_handle_t _context = nullptr;
        const static size_t _alignment = 4096;
    };

    struct deviceMem {
        deviceMem() = default;
        deviceMem(const ze_driver_handle_t driver_handle, const ze_device_handle_t device_handle,
                  const ze_context_handle_t context, const size_t size);
        deviceMem(const deviceMem&) = delete;
        deviceMem& operator=(const deviceMem&) = delete;
        deviceMem& operator=(deviceMem&&) = delete;

        const void* data() const {
            return _data;
        }
        void* data() {
            return _data;
        }
        size_t size() const {
            return _size;
        }
        ~deviceMem();

    private:
        size_t _size = 0;
        void* _data = nullptr;
        const ze_driver_handle_t _driver_handle = nullptr;
        const ze_context_handle_t _context = nullptr;
        const static size_t _alignment = 4096;
    };

    struct commandList {
        commandList() = default;
        commandList(const ze_device_handle_t& device_handle, const ze_context_handle_t& context,
                    ze_graph_dditable_ext_t* graph_ddi_table_ext);
        commandList(const commandList&) = delete;
        commandList& operator=(const commandList&) = delete;
        void reset();
        void appendMemoryCopy(void* dst, const void* src, size_t size);
        void appendGraphInitialize(const ze_graph_handle_t& graph_handle);
        void appendGraphExecute(const ze_graph_handle_t& graph_handle);
        void close();
        ~commandList();
        ze_command_list_handle_t _handle = nullptr;
        const ze_context_handle_t _context = nullptr;
        ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    };

    struct commandQueue {
        commandQueue() = default;
        commandQueue(const ze_device_handle_t& device_handle, const ze_context_handle_t& context);
        commandQueue(const commandQueue&) = delete;
        commandQueue& operator=(const commandQueue&) = delete;
        void executeCommandList(commandList& command_list);
        ~commandQueue();
        ze_command_queue_handle_t _handle = nullptr;
        ze_context_handle_t _context = nullptr;
    };

    struct fence {
        fence() = default;
        fence(const commandQueue& command_queue, ze_fence_dditable_ext_t* fence_ddi_table_ext);
        fence(const fence&) = delete;
        fence& operator=(const fence&) = delete;
        void reset();
        void hostSynchronize(uint32_t fence_value);
        void deviceSynchronize(const commandQueue& queue, uint32_t fence_value);
        void deviceSignal(uint32_t fence_value);
        ~fence();
        ze_fence_handle_t _handle = nullptr;
        ze_fence_dditable_ext_t* _fence_ddi_table_ext = nullptr;
    };

    struct argumentDescriptor {
        ze_graph_argument_properties_t info;
        uint32_t idx;
    };

    struct graph {
        graph(const ze_driver_handle_t& driver_handle, const ze_device_handle_t& device_handle,
              const ze_context_handle_t& context, const NetworkDescription::CPtr networkDescm,
              ze_graph_dditable_ext_t* graph_ddi_table_ext, ze_fence_dditable_ext_t* fence_ddi_table_ext);
        graph(const graph&) = delete;
        graph& operator=(const graph&) = delete;
        void init();
        void setArgumentValue(uint32_t argi_, const void* argv_) const;
        ~graph();
        ze_graph_handle_t _handle = nullptr;
        ze_context_handle_t _context = nullptr;
        hostMem _mem;
        ze_graph_properties_t _props{};
        std::map<std::string, argumentDescriptor> _inputs_desc_map;
        std::map<std::string, argumentDescriptor> _outputs_desc_map;
        commandQueue _command_queue;
        commandList _command_list;
        fence _fence;
        uint32_t _fence_value;

        ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    };

    enum stage {
        UPLOAD,
        EXECUTE,
        READBACK,

        COUNT
    };

    struct pipelineCommon {
        pipelineCommon(const ze_driver_handle_t& driver_handle, const ze_device_handle_t& device_handle,
                       const ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                       const graph& graph_);
        pipelineCommon(const pipelineCommon&) = delete;
        pipelineCommon& operator=(const pipelineCommon&) = delete;
        ~pipelineCommon() = default;

        std::map<std::string, hostMem> _inputs_host_mem_map;
        std::map<std::string, deviceMem> _inputs_device_mem_map;
        std::map<std::string, hostMem> _outputs_host_mem_map;
        std::map<std::string, deviceMem> _outputs_device_mem_map;
        std::array<commandList, stage::COUNT> _command_list;
    };

protected:
    const ZeroConfig& _config;
    vpu::Logger::Ptr _logger;

    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_t* _graph_ddi_table_ext = nullptr;
    ze_fence_dditable_ext_t* _fence_ddi_table_ext = nullptr;

    uint32_t _push_count;
    uint32_t _pull_count;
    uint32_t _perf_count;

    NetworkDescription::CPtr _networkDesc;

    graph _graph;

    const uint32_t _pipeline_depth;
};

template <InferenceEngine::VPUXConfigParams::ze_syncType mode_t>
class ZeroExecutor final : public Executor {};

template <>
class ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE> final : public ZeroExecutorCommon {
public:
    using Ptr = std::shared_ptr<ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>>;
    using CPtr = std::shared_ptr<const ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_FENCE>>;

    ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle, ze_context_handle_t context,
                 ze_graph_dditable_ext_t* graph_ddi_table_ext, ze_fence_dditable_ext_t* fence_ddi_table_ext,
                 const vpux::NetworkDescription::Ptr& networkDescription, const ZeroConfig& config);

    void push(const InferenceEngine::BlobMap& inputs) override;
    void pull(InferenceEngine::BlobMap& outputs) override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;

    ~ZeroExecutor() = default;

private:
    struct pipeline : public ZeroExecutorCommon::pipelineCommon {
        pipeline(const ze_driver_handle_t& driver_handle, const ze_device_handle_t& device_handle,
                 const ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                 ze_fence_dditable_ext_t* fence_ddi_table_ext, const graph& graph);
        pipeline(const pipeline&) = delete;
        pipeline& operator=(const pipeline&) = delete;
        ~pipeline() = default;
    };

    std::array<commandQueue, stage::COUNT> _command_queue;
    std::array<fence, stage::COUNT> _fence;

    std::vector<std::unique_ptr<pipeline>> _pipeline;
};

template <>
class ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT> final : public ZeroExecutorCommon {
public:
    using Ptr = std::shared_ptr<ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>>;
    using CPtr = std::shared_ptr<const ZeroExecutor<InferenceEngine::VPUXConfigParams::ze_syncType::ZE_EVENT>>;

    ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle, ze_context_handle_t context,
                 ze_graph_dditable_ext_t* graph_ddi_table_ext, ze_fence_dditable_ext_t* fence_ddi_table_ext,
                 const vpux::NetworkDescription::Ptr& networkDescription, const ZeroConfig& config);

    void push(const InferenceEngine::BlobMap& inputs) override;
    void pull(InferenceEngine::BlobMap& outputs) override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;

    ~ZeroExecutor() = default;

private:
    struct eventPool_t {
        eventPool_t() = default;
        eventPool_t(ze_device_handle_t device_handle, const ze_context_handle_t& context, uint32_t event_count);
        eventPool_t(const eventPool_t&) = delete;
        eventPool_t& operator=(const eventPool_t&) = delete;

        ~eventPool_t() {
            zeEventPoolDestroy(_handle);
        };

        const uint32_t _event_count;
        ze_event_pool_handle_t _handle = nullptr;
    };

    struct event_t {
        event_t(ze_device_handle_t device_handle, const ze_context_handle_t& context,
                const ze_event_pool_handle_t& event_pool, uint32_t event_index);
        event_t(const event_t&) = delete;
        event_t& operator=(const event_t&) = delete;
        void AppendSignalEvent(commandList& command_list);
        void AppendWaitOnEvent(commandList& command_list);
        void AppendEventReset(commandList& command_list);

        ~event_t() {
            zeEventDestroy(_handle);
        };

        ze_device_handle_t _device_t = nullptr;
        ze_context_handle_t _context = nullptr;

        ze_event_handle_t _handle = nullptr;
    };

    struct pipeline : public ZeroExecutorCommon::pipelineCommon {
        pipeline(const ze_driver_handle_t& driver_handle, const ze_device_handle_t& device_handle,
                 const ze_context_handle_t context, ze_graph_dditable_ext_t* graph_ddi_table_ext,
                 ze_fence_dditable_ext_t* fence_ddi_table_ext, const graph& graph);
        pipeline(const pipeline&) = delete;
        pipeline& operator=(const pipeline&) = delete;
        ~pipeline();

        std::condition_variable _cond_var;
        std::mutex _mutex;
        bool _available;

        eventPool_t _event_pool;
        std::array<event_t, stage::COUNT> _event;
    };

    std::array<commandQueue, stage::COUNT> _command_queue;
    std::array<fence, stage::COUNT> _fence;

    std::vector<std::unique_ptr<pipeline>> _pipeline;
};

}  // namespace vpux
