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

#pragma once

#include <ie_memcpy.h>

#include <cstring>  // std::memcpy for pointer-only args
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <condition_variable>

#include "vpux.hpp"
#include "ze_api.h"

namespace vpux {

class ZeroExecutor final : public Executor {
public:
    using Ptr = std::shared_ptr<ZeroExecutor>;
    using CPtr = std::shared_ptr<const ZeroExecutor>;

    ZeroExecutor(ze_driver_handle_t driver_handle, ze_device_handle_t device_handle,
        const vpux::NetworkDescription::Ptr& networkDescription, const VPUXConfig& config);

    void push(const InferenceEngine::BlobMap& inputs) override;
    void push(const InferenceEngine::BlobMap& inputs, const PreprocMap& preProcMap) override;

    void pull(InferenceEngine::BlobMap& outputs) override;
    // TODO: not implemented
    void setup(const InferenceEngine::ParamMap& params) override;
    bool isPreProcessingSupported(const PreprocMap& preProcMap) const override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> getLayerStatistics() override;
    InferenceEngine::Parameter getParameter(const std::string& paramName) const override;

    ~ZeroExecutor();

private:

    struct hostMem {
        hostMem() = default;
        hostMem(const ze_driver_handle_t drh_, const size_t sz_);
        hostMem(const hostMem&) = delete;
        hostMem& operator=(const hostMem&) = delete;
        hostMem& operator=(hostMem&&) = delete;

        const void* data() const { return _data; }
        void* data() { return _data; }
        size_t size() const { return _sz; }
        template <typename T>
        void copyFrom(const std::vector<T>& vec) {
            const auto inSz = vec.size() * sizeof(T);
            if (inSz != _sz)
                IE_THROW() << "hostMem::copyFrom sizes mismatch";
            if (0 != memcpy_s(_data, _sz, vec.data(), inSz))
                IE_THROW() << "hostMem::copyFrom::ie_memcpy return != 0";
        }
        void copyFrom(const InferenceEngine::Blob::Ptr& blob) {
            const InferenceEngine::MemoryBlob::CPtr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
            if (!mblob) IE_THROW() << "deviceMem::copyFrom failing of casting blob to MemoryBlob";
            if (mblob->byteSize() != _sz)
                IE_THROW() << "hostMem::copyFrom sizes mismatch";
            if (0 != memcpy_s(_data, _sz, mblob->rmap().as<const uint8_t*>(), mblob->byteSize()))
                IE_THROW() << "hostMem::copyFrom::ie_memcpy* return != 0";
        }
        void copyTo(InferenceEngine::Blob::Ptr& blob) const {
            InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
            if (!mblob) IE_THROW() << "hostMem::copyTo failing of casting blob to MemoryBlob";

            if (mblob->byteSize() != _sz) IE_THROW() << "hostMem::copyTo sizes mismatch";
            if (0 != memcpy_s(mblob->buffer().as<uint8_t*>(), mblob->byteSize(), _data, _sz))
                IE_THROW() << "hostMem::copyTo::ie_memcpy return != 0";
        }
        ~hostMem();

    private:
        size_t _sz = 0;
        void* _data = nullptr;
        const ze_driver_handle_t _drh = nullptr;
        const static size_t _alignment = 4096;
    };

    struct deviceMem {
        deviceMem() = default;
        deviceMem(const ze_driver_handle_t drh_, const ze_device_handle_t deh_, const size_t sz);
        deviceMem(const deviceMem&) = delete;
        deviceMem& operator=(const deviceMem&) = delete;
        deviceMem& operator=(deviceMem&&) = delete;

        const void* data() const { return _data; }
        void* data() { return _data; }
        size_t size() const { return _sz; }
        ~deviceMem();

    private:
        size_t _sz = 0;
        void* _data = nullptr;
        const ze_driver_handle_t _drh = nullptr;

        const static size_t _alignment = 4096;
    };

    struct commandList {
        commandList() = default;
        commandList(const ze_device_handle_t& deh_);
        commandList(const commandList&) = delete;
        commandList& operator=(const commandList&) = delete;
        void reset();
        void appendMemoryCopy(void* dst, const void* src, size_t sz);
        void appendGraphInitialize(const ze_graph_handle_t& gh_);
        void appendGraphExecute(const ze_graph_handle_t& gh_);
        void close();
        ~commandList();
        ze_command_list_handle_t _handle = nullptr;
    };

    struct commandQueue;

    struct fence {
        fence() = default;
        fence(const commandQueue& cq_);
        fence(const fence&) = delete;
        fence& operator=(const fence&) = delete;
        void reset();
        void hostSynchronize(uint32_t fence_value_);
        void deviceSynchronize(const commandQueue& queue_, uint32_t fence_value_);
        void deviceSignal(uint32_t fence_value_);
        ~fence();
        ze_fence_handle_t _handle = nullptr;
    };

    struct commandQueue {
        commandQueue() = default;
        commandQueue(const ze_device_handle_t& deh_);
        commandQueue(const commandQueue&) = delete;
        commandQueue& operator=(const commandQueue&) = delete;
        void executeCommandList(commandList& cl_);
        ~commandQueue();
        ze_command_queue_handle_t _handle = nullptr;
    };

    struct argumentDescriptor {
        ze_graph_argument_properties_t info;
        uint32_t idx;
    };

    struct graph {
        graph() = default;
        graph(const ze_driver_handle_t& drh_, const ze_device_handle_t& deh_,
            const NetworkDescription::CPtr _networkDesc);
        graph(const graph&) = delete;
        graph& operator=(const graph&) = delete;
        void init();
        void setArgumentValue(uint32_t argi_, const void* argv_) const;
        ~graph();
        ze_graph_handle_t _handle = nullptr;
        hostMem _mem;
        ze_graph_properties_t _props{ };
        std::map<std::string, argumentDescriptor> _inputs_desc_map;
        std::map<std::string, argumentDescriptor> _outputs_desc_map;
        commandQueue _command_queue;
        commandList _command_list;
        fence _fence;
    };

    enum stage {
        UPLOAD,
        EXECUTE,
        READBACK,

        COUNT
    };

    struct pipeline {
        pipeline() = default;
        pipeline(const ze_driver_handle_t& drh_, const ze_device_handle_t& deh_,
                 const std::array<commandQueue, stage::COUNT>& cq_, const graph& graph_);
        pipeline(const pipeline&) = delete;
        pipeline& operator=(const pipeline&) = delete;
        ~pipeline();

        std::map<std::string, hostMem> _inputs_host_mem_map;
        std::map<std::string, deviceMem> _inputs_device_mem_map;
        std::map<std::string, hostMem> _outputs_host_mem_map;
        std::map<std::string, deviceMem> _outputs_device_mem_map;
        std::array<commandList, stage::COUNT> _command_list;
    };

    const VPUXConfig& _config;
    vpu::Logger::Ptr _logger;

    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;

    graph _graph;

    uint32_t _push_count;
    uint32_t _pull_count;

    NetworkDescription::CPtr _networkDesc;

    std::array<commandQueue, stage::COUNT> _command_queue;
    std::array<fence, stage::COUNT> _fence;

    std::vector<std::unique_ptr<pipeline>> _pipeline;
    const uint32_t _pipeline_depth;
};

}  // namespace vpux
