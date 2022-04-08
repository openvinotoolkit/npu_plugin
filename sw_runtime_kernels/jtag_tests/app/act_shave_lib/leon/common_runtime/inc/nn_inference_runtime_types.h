//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <nn_runtime_types.h>
#include <nn_runtime_configs.h>
#include <nn_relocation_3720.h>
#include <nn_memory.h>
#include <nn_memory_alloc.h>
#include <array>
#include <algorithm>
#include <Dma.h>

#ifdef CONFIG_TARGET_SOC_3720
namespace nn {
namespace common_runtime {
namespace frontend {

struct ActKernelRangeWrapper {
    act_runtime::ActKernelRange kRange_;
    RelativeAddress kernelTextBuffer_;
    unsigned int kInvoCount_;

    ActKernelRangeWrapper()
        : kRange_()
        , kernelTextBuffer_()
        , kInvoCount_(0) {}
};

struct ActKernelInvocationWrapper {
    act_runtime::ActKernelInvocation kInvo_;
    RelativeAddress kernelDataBuffer_;
    RelativeAddress args_;
    unsigned int kRangeIndex_;
    unsigned int tile_;

    ActKernelInvocationWrapper()
        : kInvo_()
        , kernelDataBuffer_()
        , args_()
        , kRangeIndex_(0)
        , tile_(1024) {}
};

struct ActKernelRuntimeConfigsWrapper {
    act_runtime::ActKernelRuntimeConfigs asRtCfg_{};
    RelativeAddress stacks_[AS_TOTAL]{};
    RelativeAddress kernelDataBuffer_{};
};

struct DMATask {
    DmaDescriptor transaction_;
    dpu_runtime::BarrierUserConfig barriers_;

    DMATask()
        : transaction_()
        , barriers_() {}
};

struct DMAExtension {
    RelativeAddress src_;
    RelativeAddress dst_;

    DMAExtension()
        : src_()
        , dst_() {}
};
} // namespace frontend

namespace backend {
typedef frontend::ActKernelRangeWrapper ActKernelRangeWrapper;
typedef frontend::ActKernelInvocationWrapper ActKernelInvocationWrapper;
typedef frontend::ActKernelRuntimeConfigsWrapper ActKernelRuntimeConfigsWrapper;
typedef frontend::DMATask DMATask;
} // namespace backend

struct BarrierCfg {
    unsigned char real_id_;
    short next_same_id_;
    unsigned short producer_count_;
    unsigned short consumer_count_;

    BarrierCfg()
        : real_id_(255)
        , next_same_id_(-1)
        , producer_count_(0)
        , consumer_count_(0) {}
};

struct NN_CACHE_ALIGNED ResourceRequirements {
    unsigned char nn_slice_count_;
    unsigned char nn_barriers_;
    unsigned int nn_slice_length_;
    unsigned int ddr_scratch_length_;

    ResourceRequirements()
        : nn_slice_count_(0)
        , nn_barriers_(0)
        , nn_slice_length_(0)
        , ddr_scratch_length_(0) {}
};

struct NN_CACHE_ALIGNED Inference {
    ResourceRequirements resource_requirements_;
    frontend::ActKernelRuntimeConfigsWrapper actRtConfigs_;
    std::array<memory::cache_aligned_vector<frontend::DMATask>, MAX_DMA_ENGINES> dmaTasks_;
    std::array<memory::cache_aligned_vector<frontend::DMAExtension>, MAX_DMA_ENGINES> dmaExtensions_;
    memory::cache_aligned_vector<frontend::ActKernelRangeWrapper> actKRanges_;
    memory::cache_aligned_vector<frontend::ActKernelInvocationWrapper> actKInvocations_;
    memory::cache_aligned_vector<BarrierCfg> barrierConfigs_;
};

struct NN_CACHE_ALIGNED MappedInference {
    act_runtime::ActKernelRuntimeConfigs actRtConfigs_;
    std::array<unsigned int, MAX_DMA_ENGINES> leadingDmaTasks_;
    std::array<DmaDescriptor, NUM_METADATA_FEEDERS> feederDescriptors_;
    std::array<memory::FixedVector<backend::DMATask>, MAX_DMA_ENGINES> dmaTasks_;
    memory::FixedVector<backend::ActKernelRangeWrapper> actKRanges_;
    memory::FixedVector<backend::ActKernelInvocationWrapper> actKInvocations_;
    memory::FixedVector<BarrierCfg> barrierConfigs_;
};

struct NN_CACHE_ALIGNED ParsedInference {
    ResourceRequirements resource_requirements_;
    memory::FixedVector<MappedInference> mapped_;
};

struct NN_CACHE_ALIGNED InferenceRequest {
    enum Code {
        IDLE,
        PENDING,
        RUNNING,
        COMPLETE,
        OUT_OF_MEMORY,
        OUT_OF_RESOURCES,
        CONTEXT_VIOLATION,
        CONTEXT_VIOLATION_IR_HALTED,
        INTERNAL_ERROR,
        UNDEFINED
    };

    struct NN_CACHE_ALIGNED Request {
        unsigned int *barrier_lift_times_{nullptr};
        unsigned int *barrier_free_times_{nullptr};
    } request_;

    struct NN_CACHE_ALIGNED Response {
        Code code_{IDLE};
        unsigned int lnn_ticks_{0};
        const unsigned int *timestamp_{nullptr};
    } response_;

    InferenceRequest()
        : request_()
        , response_() {}
};

struct UserContextInfo {
    uint8_t ssid_;
    uint32_t aperture_offset_;

    UserContextInfo()
        : ssid_(0)
        , aperture_offset_(0) {}

    UserContextInfo(uint8_t ssid, uint32_t aperture_offset)
        : ssid_(ssid)
        , aperture_offset_(aperture_offset) {}
};

struct TileContextState {
    uint8_t ssid;
    uint32_t host_ssid;

    TileContextState(uint8_t ssid_ = 0xC0, uint32_t host_ssid_ = 0xDEAD)
        : ssid(ssid_)
        , host_ssid(host_ssid_) {}

    inline bool operator!=(const TileContextState &rhs) { return !(ssid == rhs.ssid && host_ssid == rhs.host_ssid); }
};

enum TileMask : uint8_t {
    TILE_UNALLOCATED = 0x0,
    TILE_0 = 0x1,
    TILE_1 = 0x2,
    TILE_BOTH = 0x3,
};

struct alignas(64) WorkRequest {
    enum Phase {
        PREFETCH,
        PREPARE,
        RUN,
    };

    void *user_data_;
    Phase phase_;
    UserContextInfo context_info_;
    TileMask tile_mask_;
    bool flush_required_;          // implies this WR is bound to a new PASID not seen before
    const uint64_t created_ticks_; // Global ticks when the WR was created

    union {
        struct {
            const ResourceRequirements &resources_;
            const MappedInference *mapped_;
            InferenceRequest *inference_request_;
        };
        struct {
            unsigned int count_;
            const ConstBuffer *from_;
            const ConstBuffer *to_;
        };
    };

    WorkRequest(const ResourceRequirements &res, const MappedInference *mapped, InferenceRequest *ir,
                UserContextInfo context_info, uint64_t ticks = 0, TileMask tile = TileMask::TILE_UNALLOCATED,
                bool flush_required = true)
        : user_data_(nullptr)
        , phase_(ir ? RUN : PREPARE)
        , context_info_(context_info)
        , tile_mask_(tile)
        , flush_required_(flush_required)
        , created_ticks_(ticks)
        , resources_(res)
        , mapped_(mapped)
        , inference_request_(ir){};

    WorkRequest(const ResourceRequirements &res)
        : user_data_(nullptr)
        , phase_(PREFETCH)
        , context_info_()
        , tile_mask_(TILE_UNALLOCATED)
        , flush_required_(false)
        , created_ticks_(0)
        , resources_(res)
        , mapped_(nullptr)
        , inference_request_(nullptr){};

    WorkRequest(unsigned int count, const ConstBuffer *from, const ConstBuffer *to)
        : user_data_(nullptr)
        , phase_(PREFETCH)
        , context_info_()
        , tile_mask_(TILE_UNALLOCATED)
        , flush_required_(false)
        , created_ticks_(0)
        , count_(count)
        , from_(from)
        , to_(to){};
};
} // namespace common_runtime
} // namespace nn
#endif
