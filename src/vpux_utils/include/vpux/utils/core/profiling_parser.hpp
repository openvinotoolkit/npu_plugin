#ifndef PROFILING_PARCER_HPP
#define PROFILING_PARCER_HPP

#include <string>
#include <vector>

namespace vpux {
enum profiling_task_type {
    ALL,     //< Report all tasks for profiling
    DPU_SW,  //< Only execution tasks profiling
    DMA,     //< Only DMA tasks profiling
};

struct profiling_layer_info {
    char name[256];
    char layer_type[50];
    enum layer_status_t { NOT_RUN, OPTIMIZED_OUT, EXECUTED };
    layer_status_t status;
    uint64_t start_time_ns;   //< Absolute start time
    uint64_t duration_ns;     //< Total duration (from start time unitl last compute task completed)
    uint32_t layer_id;        //< Not used
    uint64_t fused_layer_id;  //< Not used

    // Aggregate compute time  (aka. "CPU" time, will include DPU, SW, DMA)
    uint64_t dpu_ns;
    uint64_t sw_ns;
    uint64_t dma_ns;
};

struct profiling_task_info {
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
                          std::vector<profiling_task_info>& profInfo, profiling_task_type type);
void getLayerProfilingInfo(const void* data, size_t data_len,      // Pointer to the graph blob
                           const void* output, size_t output_len,  // Pointer to full profilngBuffer tensor
                           std::vector<profiling_layer_info>& profInfo);
}  // namespace vpux

#endif
