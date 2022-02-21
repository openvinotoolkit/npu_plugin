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
namespace profiling {

/**
 * @enum TaskType
 * @brief Declares which task types are required in profiling output.
 */
enum TaskType {
    ALL,     ///< Report all tasks for profiling
    DPU_SW,  ///< Only execution tasks profiling
    DMA,     ///< Only DMA tasks profiling
};

struct LayerInfo {
    char name[256];
    char layer_type[50];
    enum layer_status_t { NOT_RUN, OPTIMIZED_OUT, EXECUTED };
    layer_status_t status;
    uint64_t start_time_ns;   ///< Absolute start time
    uint64_t duration_ns;     ///< Total duration (from start time until last compute task completed)
    uint32_t layer_id;        ///< Not used
    uint64_t fused_layer_id;  ///< Not used

    // Aggregate compute time  (aka. "CPU" time, will include DPU, SW, DMA)
    uint64_t dpu_ns;
    uint64_t sw_ns;
    uint64_t dma_ns;
};

struct TaskInfo {
    char name[256];
    char layer_type[50];
    enum class ExecType {
        NONE,
        DPU,
        SW,
        DMA,
    };
    ExecType exec_type;
    uint64_t start_time_ns;
    uint64_t duration_ns;
    uint32_t active_cycles;
    uint32_t stall_cycles;
    uint32_t task_id;
    uint32_t parent_layer_id;  ///< Not used
};

/**
 * @fn getTaskInfo
 * @brief Parse raw profiling output to get per-tasks info.
 * @param blob_data pointer to the buffer with blob binary
 * @param blob_size blob size in bytes
 * @param prof_data pointer to the buffer with raw profiling data
 * @param prof_size raw profiling data size
 * @param type type of tasks to be profiled
 * @see TaskType
 * @return std::vector of TaskInfo structures
 */
std::vector<TaskInfo> getTaskInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                  TaskType type);

/**
 * @fn getLayerInfo
 * @brief Parse raw profiling output to get per-layer info.
 * @param blob_data pointer to the buffer with blob binary
 * @param blob_size blob size in bytes
 * @param prof_data pointer to the buffer with raw profiling data
 * @param prof_size raw profiling data size
 * @return std::vector of LayerInfo structures
 */
std::vector<LayerInfo> getLayerInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize);

/**
 * @fn getLayerInfo
 * @brief Parse raw profiling output to get per-layer info. Reuses precomputed info about tasks.
 * @param taskInfo output from \b getTaskInfo function.
 * @return std::vector of LayerInfo structures
 * @see getTaskInfo
 */
std::vector<LayerInfo> getLayerInfo(const std::vector<TaskInfo>& taskInfo);

}  // namespace profiling
}  // namespace vpux

#endif
