/*
 * {% copyright %}
 */

#pragma once
namespace nn {
namespace common_runtime {

enum : unsigned int {
    // For Std FW, only use 1 thread for 1/2 tile execution.
    // For standalone tests, use 2 threads to support parallel tile 0/1 execution

#ifndef CONFIG_VALIDATION_APP_ENABLED
    INF_THREADS_COUNT = 1,
#else
    INF_THREADS_COUNT = 2,
#endif
    IR_WORKER_COUNT = INF_THREADS_COUNT,

    // TODO: Change this once multi-substreamID architecture is defined
    // For now just use one worker for simplicity

    // variant, invariant, act invo
    NUM_COMPONENT_FEEDERS = 3,

    // Two DMA + components
    NUM_METADATA_FEEDERS = MAX_DMA_ENGINES + NUM_COMPONENT_FEEDERS,

    // Dual tile blob can use all 64 barriers. Single tile blob can only use 32
    // (since it may be running in parallel)
    MAX_BARRIERS_PER_INFERENCE = 64,

    // See docs/ReadMe.md : Windowing and Memory Layout
    ACT_CMX_WINDOW = 0x1F000000,
    ACT_RT_CODE_WINDOW = 0x1C000000,
    ACT_KERNEL_CODE_WINDOW = 0x1D000000,
    ACT_KERNEL_DATA_WINDOW = 0x1E000000
};
} // namespace common_runtime
} // namespace nn
