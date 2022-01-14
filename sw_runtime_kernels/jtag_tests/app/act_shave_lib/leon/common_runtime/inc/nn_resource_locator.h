//
// Copyright Intel Corporation.
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

#include <nn_inference_runtime_types.h>
#include <nn_cmx_memory_map.h>
#include <nn_relocation_3720.h>
#include <nn_hw_resources.h>
#include <nn_math.h>
#include <array>

namespace nn {
namespace common_runtime {
template <typename Task>
class TaskLocator {
public:
    TaskLocator()
        : tasks_(nullptr)
        , count_(0) {}

    TaskLocator(uint32_t addr, unsigned int size)
        : TaskLocator(reinterpret_cast<void *>(addr), size) {}

    TaskLocator(void *addr, unsigned int size)
        : tasks_(reinterpret_cast<Task *>(math::ptr_align_up<alignof(Task)>(addr)))
        , count_(round_down_to_power_of_2(
              (size - static_cast<unsigned int>(reinterpret_cast<char *>(tasks_) - reinterpret_cast<char *>(addr))) /
              sizeof(Task))) {
        nnLog(MVLOG_DEBUG, "%p, %u -> %p, %u", addr, size, tasks_, count_);
    }

    inline Task &task(unsigned int i) const { return tasks_[i & (count_ - 1)]; }
    inline Task *tasks() const { return tasks_; }
    inline unsigned int count() const { return count_; }

private:
    Task *tasks_;
    unsigned int count_;

    static inline unsigned int round_down_to_power_of_2(unsigned int x) {
        if (x == 0)
            return 0;

        return 1u << math::lastBitIndex(x);
    }
};

typedef TaskLocator<backend::DMATask> DMALocator;
typedef TaskLocator<backend::ActKernelRangeWrapper> AKRangeLocator;
typedef TaskLocator<backend::ActKernelInvocationWrapper> AKInvocationLocator;

struct StaticMapping {
    StaticMapping() = default;
    StaticMapping(NNCmxMemoryMap *cmx);

    std::array<Buffer, MAX_CLUSTERS> globalData_;
    std::array<Buffer, MAX_CLUSTERS> snnStack_;
    std::array<Buffer, MAX_CLUSTERS> workareas_;
    std::array<Buffer, MAX_CLUSTERS> actShvStack_;
    std::array<Buffer, MAX_CLUSTERS> metadataStorage_;
    std::array<Buffer, MAX_DMA_ENGINES> dmaStorage_;
};

struct RuntimeMapping {
    RuntimeMapping();
    RuntimeMapping(const StaticMapping &global, ClusterMapper::Config config);

    std::array<DMALocator, MAX_DMA_ENGINES> dma_;
    AKRangeLocator akr_;
    AKInvocationLocator aki_;

    ClusterMapper::Config config_;
    std::array<unsigned char, MAX_CLUSTERS> fifos_;

private:
    enum {
        INVARIANT_COUNT = 63,
        VARIANT_COUNT = 512,
        KERNAL_RANGE_COUNT = 0,
        KERNAL_INVO_COUNT = 128,
    };
};
} // namespace common_runtime
} // namespace nn
