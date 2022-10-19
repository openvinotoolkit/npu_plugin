//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <stdint.h>

#include "nce_2p7_hw.h"

namespace host_parsing {

struct BarrierConfig {
    uint8_t group;
    uint8_t mask;
    uint64_t consumer_mask;
    uint64_t producer_mask;
};

struct DPUInvariant {
    DPUInvariantRegisters registers;
    BarrierConfig barriers;
    uint32_t channel_major_stride;
    uint32_t output_sparsity_offset;
};

struct DPUVariant {
    uint32_t invariant_addr;
    DPUVariantRegisters registers;
    uint32_t output_sparsity_offset;
    uint32_t weight_table_offset;
};

template <typename T>
struct TaskReference {
    uint64_t address;
    uint32_t count;
    uint32_t reserved;

    T* data() const { return reinterpret_cast<T*>(address); }
    uint32_t size() const { return count; };
    T& operator[] (uint32_t index) { return data()[index]; }
};

struct DmaWrapper {
    host_parsing::DmaDescriptor transaction;
    uint16_t start_after;
};

struct DPUInvariantWrapper {
    DPUInvariant invariant;
    uint16_t variant_count;
    uint16_t start_after;
    uint16_t clean_after;
    uint8_t cluster;
    uint8_t padding;
};

struct DPUVariantWrapper {
    DPUVariant variant;
    uint32_t invariant_index;
};

struct BarrierWrapper {
    int32_t  next_same_id;
    uint16_t producer_count;
    uint16_t consumer_count;
    uint8_t  real_id;
};

// ActKernel structs
extern "C" struct ActKernelRange {
    ActWLType type_; 
    actKernelEntry kernelEntry_; 
    actKernelTextBuffer textWindowBase_; 

    uint32_t codeSize_;
    uint32_t dataSecSize_;
};

extern "C" struct ActKernelInvocation {
    uint32_t range_; 
    actKernelArgs kernelArgs_;
    actKernelDataBuffer dataWindowBase_;

    BarrierConfig barriers_;
};
extern "C" struct ActKernelRuntimeConfigs {
    uint32_t stackFrames_[4]; // 4 = AS_TOTAL;
    uint32_t stackSize_;
    bool useScheduleEmbeddedRt_;

    // when useScheduleEmbeddedRt = true
    actRuntimeEntry runtimeEntry_;

    // when useScheduleEmbeddedRt = false; FW copies ActRt to this buffer
    // when useScheduleEmbeddedRt = true; buffer already contains the ActRt
    uint32_t actRtWindowBase_;
    uint32_t codeWindowBufferSize_;
};

struct ActKernelRuntimeConfigsWrapper {
    ActKernelRuntimeConfigs asRtCfg_;
};

struct ActKernelRangeWrapper {
    ActKernelRange kRange_;
    uint32_t kInvoCount_;
};

struct ActKernelInvocationWrapper {
    ActKernelInvocation kInvo_;
    uint32_t kRangeIndex_;
    uint32_t tile_;
    uint16_t start_after_;
    uint16_t clean_after_;
};

struct MappedInference {
    TaskReference<DmaWrapper> dmaTasks[2];
    uint32_t leadingDmaCount[2];
    TaskReference<DPUInvariantWrapper> invariants;
    TaskReference<DPUVariantWrapper> variants;
    TaskReference<BarrierWrapper> barrierConfigs;
    DmaDescriptor feederDescriptors[5];

    // TBD - one of these lists may go away depending on final SW layer design
    TaskReference<ActKernelRangeWrapper> actKRanges;
    TaskReference<ActKernelInvocationWrapper> actKInvocations;
    ActKernelRuntimeConfigs actRtConfigs;

};

struct ResourceRequirements {
    uint8_t  slice_count;
    uint16_t barrier_count;
};

struct HostParsedInference {
    uint32_t magic;
    ResourceRequirements resource_requirements;
    TaskReference<MappedInference> mapped;
};

}
