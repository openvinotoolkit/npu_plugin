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

#ifndef PROFILING_PARSER_HPP
#define PROFILING_PARSER_HPP

#include <string>
#include <vector>

namespace vpux {
enum ProfilingTaskType {
    ALL,     //< Report all tasks for profiling
    DPU_SW,  //< Only execution tasks profiling
    DMA,     //< Only DMA tasks profiling
};

struct ProfilingLayerInfo {
    char name[256];
    char layer_type[50];
    enum layer_status_t { NOT_RUN, OPTIMIZED_OUT, EXECUTED };
    layer_status_t status;
    uint64_t start_time_ns;   //< Absolute start time
    uint64_t duration_ns;     //< Total duration (from start time until last compute task completed)
    uint32_t layer_id;        //< Not used
    uint64_t fused_layer_id;  //< Not used

    // Aggregate compute time  (aka. "CPU" time, will include DPU, SW, DMA)
    uint64_t dpu_ns;
    uint64_t sw_ns;
    uint64_t dma_ns;
};

struct ProfilingTaskInfo {
    char name[256];
    char layer_type[50];
    enum exec_type_t {
        NONE,
        DPU,
        SW,
        DMA,
    };
    exec_type_t exec_type;
    uint64_t start_time_ns;
    uint64_t duration_ns;
    uint32_t active_cycles;
    uint32_t stall_cycles;
    uint32_t task_id;
    uint32_t parent_layer_id;  //< Not used
};

void getTaskProfilingInfo(const void* data, size_t data_len,      // Pointer to the graph blob
                          const void* output, size_t output_len,  // Pointer to full profilngBuffer tensor
                          std::vector<ProfilingTaskInfo>& profInfo, ProfilingTaskType type);
void getLayerProfilingInfo(const void* data, size_t data_len,      // Pointer to the graph blob
                           const void* output, size_t output_len,  // Pointer to full profilngBuffer tensor
                           std::vector<ProfilingLayerInfo>& profInfo);
}  // namespace vpux

#endif
