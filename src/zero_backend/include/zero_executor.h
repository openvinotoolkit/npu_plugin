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
    void memory_init();
    void commit();

    struct hostMem {
        hostMem() = default;
        explicit hostMem(const ze_driver_handle_t h): drh(h) {}
        hostMem(const hostMem&) = delete;
        hostMem& operator=(const hostMem&) = delete;
        hostMem(hostMem&& other): sz(other.sz), mem(other.mem), drh(other.drh) {
            other.sz = 0;
            other.mem = nullptr;
        }
        hostMem& operator=(hostMem&&) = delete;

        void init(const ze_driver_handle_t h);
        void resize(const size_t);
        void* data() { return mem; }
        const void* data() const { return mem; }
        size_t size() const { return sz; }
        template <typename T>
        void copyFrom(const std::vector<T>& vec) {
            const auto inSz = vec.size() * sizeof(T);
            resize(inSz);
            if (0 != ie_memcpy(mem, sz, vec.data(), inSz))
                THROW_IE_EXCEPTION << "hostMem::copyFrom::ie_memcpy return != 0";
        }
        void copyFrom(const InferenceEngine::Blob::Ptr& blob) {
            const InferenceEngine::MemoryBlob::CPtr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
            if (!mblob) THROW_IE_EXCEPTION << "deviceMem::copyFrom failing of casting blob to MemoryBlob";

            resize(mblob->byteSize());
            if (0 != ie_memcpy(mem, sz, mblob->rmap().as<const uint8_t*>(), mblob->byteSize()))
                THROW_IE_EXCEPTION << "hostMem::copyFrom::ie_memcpy* return != 0";
        }
        void copyTo(InferenceEngine::Blob::Ptr& blob) const {
            InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
            if (!mblob) THROW_IE_EXCEPTION << "deviceMem::copyTo failing of casting blob to MemoryBlob";

            if (mblob->byteSize() != sz) THROW_IE_EXCEPTION << "deviceMem::copyTo sizes mismatch";
            if (0 != ie_memcpy(mblob->buffer().as<uint8_t*>(), mblob->byteSize(), mem, sz))
                THROW_IE_EXCEPTION << "hostMem::copyTo::ie_memcpy return != 0";
        }
        void free();
        ~hostMem() {
            try {
                free();
            } catch (...) {
                std::cerr << "Error in dtor: zeGraphDestroy failed at hostMem" << std::endl;
            }
        }

    private:
        size_t sz = 0;
        void* mem = nullptr;
        ze_driver_handle_t drh = nullptr;

        const static size_t alignment = 4096;
    };

    // defferred operations. be carefull
    struct deviceMem {
        deviceMem() = default;
        deviceMem(const ze_driver_handle_t drh_, const ze_device_handle_t deh_, const ze_command_list_handle_t clh_)
            : drh(drh_), deh(deh_), clh(clh_) {}
        deviceMem(const deviceMem&) = delete;
        deviceMem& operator=(const deviceMem&) = delete;
        deviceMem(deviceMem&& other): sz(other.sz), mem(other.mem), drh(other.drh), deh(other.deh), clh(other.clh) {
            other.sz = 0;
            other.mem = nullptr;
        }
        deviceMem& operator=(deviceMem&&) = delete;

        void init(const ze_driver_handle_t drh_, const ze_device_handle_t deh_, const ze_command_list_handle_t clh_);
        void resize(const size_t);
        void* data() { return mem; }
        const void* data() const { return mem; }
        size_t size() const { return sz; }
        // actually it puts command to copy in a command_list afaiu
        // for KMB we must copy through hostMem! real copy will appear when sync on fence!
        void copyFrom(const InferenceEngine::Blob::Ptr& blob) {
            const InferenceEngine::MemoryBlob::CPtr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
            if (!mblob) THROW_IE_EXCEPTION << "deviceMem::copyFrom failing of casting blob to MemoryBlob";

            resize(mblob->byteSize());
            copyFromImpl(mblob->rmap().as<const uint8_t*>());
        }
        void copyFrom(const hostMem& hm) {
            resize(hm.size());
            copyFromImpl(hm.data());
        }
        void copyTo(InferenceEngine::Blob::Ptr& blob) {
            InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
            if (!mblob) THROW_IE_EXCEPTION << "deviceMem::copyTo failing of casting blob to MemoryBlob";
            if (mblob->byteSize() != sz) THROW_IE_EXCEPTION << "deviceMem::copyTo sizes mismatch";
            copyToImpl(mblob->wmap().as<uint8_t*>());
        }
        void copyTo(hostMem& hm) {
            hm.resize(sz);
            copyToImpl(hm.data());
        }
        void free();
        ~deviceMem() { free(); }

    private:
        void copyToImpl(void* dst);
        void copyFromImpl(const void* src);

        size_t sz = 0;
        void* mem = nullptr;
        ze_driver_handle_t drh = nullptr;
        ze_device_handle_t deh = nullptr;
        ze_command_list_handle_t clh = nullptr;

        const static size_t alignment = 4096;
    };

    struct graph_raii {
        graph_raii() = default;
        void init(const ze_device_handle_t& device_handle, hostMem& graphMemory);
        void getProperties(ze_graph_properties_t& property);
        void getArgumentProperties(const uint32_t index, ze_graph_argument_properties_t& arg);
        void setArgumentValue(const uint32_t index, const void* data);
        void commandListAppendGraphInitialize(const ze_command_list_handle_t& list);
        void commandListAppendGraphExecute(const ze_command_list_handle_t& list);
        ~graph_raii();
        graph_raii& operator=(const graph_raii&) = delete;
        graph_raii(const graph_raii&) = delete;
        ze_graph_handle_t g = nullptr;
    };

    const VPUXConfig& _config;
    vpu::Logger::Ptr _logger;

    ze_driver_handle_t _driver_handle = nullptr;
    ze_device_handle_t _device_handle = nullptr;
    ze_command_queue_handle_t _command_queue_handle = nullptr;
    ze_command_list_handle_t _command_list_handle = nullptr;
    ze_fence_handle_t _fence_handle = nullptr;

    graph_raii _graph_handle;
    ze_graph_properties_t _graph_properties;

    hostMem graphMemory;

    struct argumentDescriptor {
        deviceMem memory;
        ze_graph_argument_properties_t info;
        uint32_t idx;
    };
    std::map<std::string, argumentDescriptor> inputs_map;
    std::map<std::string, argumentDescriptor> outputs_map;
};

}  // namespace vpux
