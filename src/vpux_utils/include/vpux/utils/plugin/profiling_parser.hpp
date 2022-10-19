//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef PROFILING_PARSER_HPP
#define PROFILING_PARSER_HPP

#include <string>
#include <vector>

namespace vpux {
namespace profiling {

// Suffix used to create cluster name from task name
const std::string CLUSTER_LEVEL_PROFILING_SUFFIX = "/cluster_";
// Suffix used to create variant name from cluster name
const std::string VARIANT_LEVEL_PROFILING_SUFFIX = "/variant_";

/**
 * @enum TaskType
 * @brief Declares which task types are required in profiling output.
 */
enum TaskType {
    ALL,     ///< Report all tasks for profiling
    DPU_SW,  ///< Only execution tasks profiling
    DMA,     ///< Only DMA tasks profiling
};

/**
 * @enum VerbosityLevel
 * @brief Declares verbosity level of printing information
 */
enum VerbosityLevel {
    LOW = 0,     ///< Default, only DMA/SW/Aggregated DPU info
    MEDIUM = 1,  ///< Extend by cluster level information
    HIGH = 5,    ///< Full information including individual variants timings
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
 * @see VerbosityLevel
 * @return std::vector of TaskInfo structures
 */
std::vector<TaskInfo> getTaskInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                  TaskType type, VerbosityLevel verbosity);

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
