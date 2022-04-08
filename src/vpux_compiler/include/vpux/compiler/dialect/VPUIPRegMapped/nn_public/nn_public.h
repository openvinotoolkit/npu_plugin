//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <stdint.h>
#include <array>

#include "dma_2p7_hw.h"
#include "nce_2p7_hw.h"

/*
 * When a change is made to nn_public.h that breaks backwards compatibility
 * VPU_NN_PUBLIC_VER_MAJOR must be incremented.
 *
 * If a change preserves backwards compatibility then VPU_NN_PUBLIC_VER_MINOR
 * should be incremented. It resets to 0 when the major version is incremented.
 *
 * If nn_public.h is modified (field names, documentation, formatting) but the API
 * itself is not changed, then VPU_NN_PUBLIC_VER_PATCH should be incremented.
 *
 * When the compiler creates a MappedInference in an ELF blob
 * MappedInference.nn_public_version is set to the version of nn_public used.
 * NNRuntime checks this version at inference time to ensure it is current and
 * returns an error if the major version does not match.
 * Note: VPU_NN_PUBLIC_VER_PATCH is not stored in the MappedInference as
 * compatibility is not affected if this changes.
 */
#define VPU_NN_PUBLIC_VER_MAJOR 3
#define VPU_NN_PUBLIC_VER_MINOR 0
#define VPU_NN_PUBLIC_VER_PATCH 0
#define VPU_NN_PUBLIC_VER ((VPU_NN_PUBLIC_VER_MAJOR << 16) | VPU_NN_PUBLIC_VER_MINOR)

/*
 * When a change is made to the Activation Shave Runtime / Mangement kernel
 * (nnActEntry.cpp), that breaks backwards compatibility (e.g. changing the
 * nnActEntry function parameters) VPU_ACT_RT_VER_MAJOR must be incremented.
 *
 * If a change preserves backwards compatibility then VPU_ACT_RT_VER_MINOR
 * should be incremented. It resets to 0 when the major version is incremented.
 */
#define VPU_ACT_RT_VER_MAJOR 1
#define VPU_ACT_RT_VER_MINOR 0
#define VPU_ACT_RT_VER_PATCH 0
#define VPU_ACT_RT_VER ((VPU_ACT_RT_VER_MAJOR << 16) | VPU_ACT_RT_VER_MINOR)

#define VPU_SCALABILITY_NUM_OF_FREQ 5
#define VPU_SCALABILITY_VALUES_PER_FREQ 5

namespace nn_public {

template <typename T>
struct vpu_ptr {
    uint32_t ptr;

    vpu_ptr(): ptr(0) {
    }

    vpu_ptr<T>& operator=(T* ptr) {
        this->ptr = reinterpret_cast<uint32_t>(ptr);
        return *this;
    }

    vpu_ptr<T>& operator=(uint32_t ptr) {
        this->ptr = ptr;
        return *this;
    }

    operator T*() const {
        return reinterpret_cast<T*>(ptr);
    }
    T* operator->() const {
        return reinterpret_cast<T*>(ptr);
    }
    explicit operator bool() const {
        return ptr;
    }
    explicit operator uintptr_t() const {
        return ptr;
    }
};

template <typename T>
struct TaskReference {
    // uint32_t address can point to memory in host user space within the PIOVA aperture.
    // In this case it needs converted to the bridge aperture to make it accessible from
    // LeonRT/LeonNN on MTL.
    //
    // Use the methods data(int64_t offset) and at(uint32_t index, int64_t offset) to
    // apply the aperture offset to convert address to the bridge aperture.
    uint64_t address;
    uint64_t count;

    T* data() {
        return reinterpret_cast<T*>(address);
    }
    const T* data() const {
        return reinterpret_cast<T*>(address);
    }

    T* data(int64_t offset) {
        return reinterpret_cast<T*>(address + offset);
    }
    const T* data(int64_t offset) const {
        return reinterpret_cast<T*>(address + offset);
    }

    uint32_t size() const {
        return count;
    }

    T& at(uint32_t index, int64_t offset = 0) {
        return (reinterpret_cast<T*>(address + offset))[index];
    }
    const T& at(uint32_t index, int64_t offset = 0) const {
        return (reinterpret_cast<T*>(address + offset))[index];
    }

    template <class TD>
    TaskReference& operator=(TD fixedVector) {
        // If fixedVector has a bridge aperture offset remove it to store the address in the
        // PIOVA aperture as the offset can change.
        address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fixedVector.data())) - fixedVector.apertureOffset();
        count = static_cast<uint64_t>(fixedVector.size());
        return *this;
    };
};

typedef void(actKernelEntryFunction)(void*);

struct TaskSchedulingBarrierConfig {
    uint16_t start_after_;
    uint16_t clean_after_;
};

static_assert(sizeof(TaskSchedulingBarrierConfig) == 4, "TaskSchedulingBarrierConfig size != 4");

struct TaskBarrierDependecy {
    uint64_t wait_mask_;
    uint64_t post_mask_;
    uint8_t group_;
    uint8_t mask_;
};

struct BarrierCountConfig {
    int16_t next_same_id_;
    uint16_t producer_count_;
    uint16_t consumer_count_;
    uint8_t real_id_;
};

struct DPUInvariant {
    DPUInvariantRegisters registers_;
    uint32_t hwp_cmx_base_offset_;
    TaskBarrierDependecy barriers_;
    TaskSchedulingBarrierConfig barriers_sched_;
    uint32_t output_sparsity_offset_;
    uint16_t variant_count_;
    uint8_t cluster_;
    uint8_t is_cont_conv_;
};

struct DPUVariant {
    DPUVariantRegisters registers_;
    vpu_ptr<DPUInvariant> invariant_;
    uint32_t invariant_index_;
    uint32_t output_sparsity_offset_;
    uint32_t weight_table_offset_;
    int32_t wload_id_;
    uint8_t cluster_;
};

struct ResourceRequirements {
    uint32_t nn_slice_length_;
    uint32_t ddr_scratch_length_;
    uint16_t nn_barrier_count_;
    uint8_t nn_slice_count_;
    uint8_t nn_barriers_;
};

struct NNShaveRuntimeConfigs {
    uint32_t runtime_entry;  // when useScheduleEmbeddedRt = true this is a windowed address
    uint32_t act_rt_window_base;
    uint32_t stack_frames[AS_TOTAL];  // this is aligned to 64 bits due to AS_TOTAL
    uint32_t stack_size;
    uint32_t code_window_buffer_size;
    uint32_t perf_metrics_mask;
    uint32_t runtime_version;
    uint8_t use_schedule_embedded_rt;  // when useScheduleEmbeddedRt = false; FW copies ActRt to this buffer
                                       // when useScheduleEmbeddedRt = true; buffer already contains the ActRt
    HWPStatMode dpu_perf_mode;
};

// Forcing struct padding so we have same sizeof() of the structure both on x86 compilation and Sparc
// compilation.
struct DMATask {
    vpu_dma_descriptor_t transaction_;
    TaskSchedulingBarrierConfig barriers_sched_;
    uint8_t pad_[128 - (sizeof(vpu_dma_descriptor_t) + sizeof(TaskSchedulingBarrierConfig))];
};

static_assert(sizeof(DMATask) == 128, "DMATask size != 128");

struct ActKernelRange {
    ActWLType type;
    vpu_ptr<actKernelEntryFunction> kernel_entry;
    vpu_ptr<void> text_window_base;
    uint32_t code_size;
    uint32_t data_sec_size;
    uint32_t kernel_invo_count;
};

struct ActKernelInvocation {
    vpu_ptr<ActKernelRange> range;
    vpu_ptr<void> kernel_args;
    vpu_ptr<void> data_window_base;
    vpu_ptr<void> perf_packet_out;
    TaskBarrierDependecy barriers;
    TaskSchedulingBarrierConfig barriers_sched;
    // The schedule compiler can infer an index if it's needed pre/post inference
    // Update: we can/will use the index to virtualize a WI FIFO state in a preemption payload
    uint32_t invo_index;
    uint32_t invo_tile;
    uint32_t kernel_range_index;
};

// Plain wrapper struct for vpu_dma_descriptor_t. Defining an array of a structure where the size of the struct is not
// multiple of its alignment is undefined behavior via the C standard. Up until GCC 11 it has been silently handled. In
// vpuip gcc case it actually ingonred alignment for N+1th element and generated code with each vpu_dma_descriptor_t in
// the array 80-bytes distanced. (See
// https://gcc.gnu.org/git/?p=gcc.git;a=commit;h=50bc94898fac1bd9cc1dabf227208fb5d369c4c4) Explicit padding required to
// force size of structure to 128.

struct DescriptorWrapper {
    vpu_dma_descriptor_t descriptor_;
    uint8_t pad_[128 - sizeof(vpu_dma_descriptor_t)];
};

static_assert(sizeof(DescriptorWrapper) == 128, "DMA descriptor wrapper size != 128");

struct MappedInference {
    uint32_t nn_public_version;
    TaskReference<DMATask> dma_tasks[MAX_DMA_ENGINES];
    TaskReference<DPUInvariant> invariants;
    TaskReference<DPUVariant> variants;
    TaskReference<BarrierCountConfig> barrier_configs;
    TaskReference<ActKernelRange> act_kernel_ranges;
    TaskReference<ActKernelInvocation> act_kernel_invocations;
    DescriptorWrapper feeder_descriptors[NUM_METADATA_FEEDERS];
    uint32_t leading_dma_tasks[MAX_DMA_ENGINES];
    NNShaveRuntimeConfigs shv_rt_configs;
};

static_assert(sizeof(MappedInference::feeder_descriptors) == (NUM_METADATA_FEEDERS * sizeof(DescriptorWrapper)),
              "Sizeof feeder_descriptors != NUM_METADATA_FEEDERS * DescriptorWrapper");
static_assert(sizeof(MappedInference) == 960, "MappedInference size != 960");

struct PerformanceMetrics {
    uint32_t freq_base;  ///< Base of frequency values used in tables (in MHz).
    uint32_t freq_step;  ///< Step of frequency for each entry in tables (in MHz).
    uint32_t bw_base;    ///< Base of bandwidth values used in tables (in MB/s).
    uint32_t bw_step;    ///< Step of bandwidth values used in tables (in MB/s).

    /// Inner arrays are for different bandwidth values.
    /// Outer arrays are for different frequency values.
    float scalability[VPU_SCALABILITY_NUM_OF_FREQ][VPU_SCALABILITY_VALUES_PER_FREQ];  ///< Table of scalability values.
    uint64_t ticks[VPU_SCALABILITY_NUM_OF_FREQ][VPU_SCALABILITY_VALUES_PER_FREQ];     ///< Table of inference timings.

    float activity_factor;  ///< Compiler estimated activity factor for the inference.
};

struct alignas(64) HostParsedInference {
    ResourceRequirements resource_requirements_;
    PerformanceMetrics performance_metrics_;
    TaskReference<MappedInference> mapped_;
};

// Segment sizes in CMX
constexpr uint32_t SNN_DATA_SIZE = 1024;
constexpr uint32_t SNN_STACK_SIZE = 1024;
constexpr uint32_t ACTSHV_SCRATCH_SIZE = 1024;
constexpr uint32_t METADATA_SIZE = 45 * 1024;
constexpr uint32_t WORKSPACE_SIZE = 1965 * 1024;
constexpr uint32_t DMA_STORAGE_SIZE = 35 * 1024;
constexpr uint32_t DMA_STORAGE_PER_ENGINE = DMA_STORAGE_SIZE / MAX_DMA_ENGINES;

constexpr uint32_t INVARIANT_COUNT = 32;
constexpr uint32_t VARIANT_COUNT = 256;
constexpr uint32_t KERNAL_RANGE_COUNT = 32;
constexpr uint32_t KERNAL_INVO_COUNT = 64;

}  // namespace nn_public
