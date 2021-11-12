// {% copyright %}

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

extern "C" struct ActKernelRange {
    ActWLType type_{ActWLType::WL_UNKNOWN};
    actKernelEntry kernelEntry_{nullptr}; // entry point to main func (hswish etc)
    actKernelTextBuffer textWindowBase_{nullptr}; // pointer to real address of text window (not 1d)

    uint32_t codeSize_{0};
    uint32_t dataSecSize_{0};
};

extern "C" struct ActKernelInvocation {
    ActKernelRange *range_{nullptr};
    act_kernel_args *kernelArgs_{nullptr}; //
    actKernelDataBuffer dataWindowBase_{nullptr}; //pointer to real address of data window (not 1e)

    BarrierConfig barriers_{};
    // BarrierGpioConfig barriers_gpio_{}; NOT PRESENT IN DPU INVARIANT AS WELL
    unsigned int invo_index_{0};
};
extern "C" struct ActKernelRuntimeConfigs_backend {
    unsigned int stackFrames_[4]{0}; // AS_TOTAL = AS_PER_TILE * MAX_TILES = 4 * 4
    unsigned int stackSize_{0};
    bool useScheduleEmbeddedRt_{false};

    // when useScheduleEmbeddedRt = true
    // this is a windowed address
    // idk what dis is
    actRuntimeEntry runtimeEntry_{nullptr};

    // when useScheduleEmbeddedRt = false; FW copies ActRt to this buffer
    // when useScheduleEmbeddedRt = true; buffer already contains the ActRt
    unsigned char *actRtWindowBase_{nullptr};
    unsigned int codeWindowBufferSize_{0};
};

struct ActKernelRuntimeConfigs {
    ActKernelRuntimeConfigs_backend asRtCfg_{};
    // RelativeAddress stacks_[AS_TOTAL]{};
    // RelativeAddress kernelDataBuffer_{};
};

struct ActKernelRangeWrapper {
    ActKernelRange kRange_;
    // RelativeAddress kernelTextBuffer_;
    unsigned int kInvoCount_;
};

struct ActKernelInvocationWrapper {
    ActKernelInvocation kInvo_;
    // RelativeAddress kernelDataBuffer_;
    // RelativeAddress args_;
    unsigned int kRangeIndex_;
    unsigned int tile_;
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
