//
// Copyright (C) 2022 Intel Corporation.
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

#include <stdint.h>

#include "nce_2p7_hw.h"

namespace host_parsing {

// Data structures used by LNN/SNN for actual execution
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

// Data structures with fields used by LNN for managing execution flow
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

struct ActKernelRuntimeConfigs {};
struct ActKernelRangeWrapper {};
struct ActKernelInvocationWrapper {};

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
