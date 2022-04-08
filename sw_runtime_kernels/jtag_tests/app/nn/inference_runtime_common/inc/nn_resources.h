/*
* {% copyright %}
*/
#ifndef NN_RESOURCES_H_
#define NN_RESOURCES_H_

namespace nn
{
    namespace inference_runtime
    {
        enum
        {
#if defined(CONFIG_TARGET_SOC_3600) || defined(CONFIG_TARGET_SOC_3710)
            // TODO: Change this once multi-substreamID architecture is defined
            // For now just use one worker for simplicity
            IR_WORKER_COUNT = 1,

            MAX_SLICES = 2,
            MAX_CLUSTERS = 2,
            MAX_DPUS = 4,
            MAX_DPUS_FIFOS = 4,

            SLICE_LENGTH = 2 * 1024 * 1024,
#elif defined(CONFIG_TARGET_SOC_3720)
            // TODO: Change this once multi-substreamID architecture is defined
            // For now just use one worker for simplicity
            IR_WORKER_COUNT = 1,

            MAX_SLICES = 2,
            MAX_CLUSTERS = 2,
            MAX_DPUS = 2,
            MAX_DPUS_FIFOS = 2,

            SLICE_LENGTH = 2 * 1024 * 1024,
#else
            IR_WORKER_COUNT = 4,

            MAX_SLICES = 4,
            MAX_CLUSTERS = 4,
            MAX_DPUS = 20,
            MAX_DPUS_FIFOS = 16,

            SLICE_LENGTH = 1024 * 1024,
#endif

#if defined(CONFIG_TARGET_SOC_3710) || defined(CONFIG_TARGET_SOC_3720)
            MAX_TILES = MAX_CLUSTERS,
            ACT_SHAVES_IN_TILE = 2,
            ACT_SHAVE_0_INDEX = 4,
            WORK_FIFO_COUNT = 16,

#endif
// TODO: Enable second DMA engine on VPUX37XX (D#3752)
#if defined(CONFIG_TARGET_SOC_MA2490) || defined(CONFIG_TARGET_SOC_3600) || defined(CONFIG_TARGET_SOC_3710) || defined(CONFIG_TARGET_SOC_3720)
            DMA_ENGINES = 1,
#else
            DMA_ENGINES = 2,
#endif

            MAX_DMA_ENGINES = 2,

#if defined(CONFIG_TARGET_SOC_3710) || defined(CONFIG_TARGET_SOC_3720)
            // Two DMA, variant & invariant
            NUM_METADATA_FEEDERS = MAX_DMA_ENGINES + 2,
#else
            // Two DMA, variant, invariant & softlayer
            NUM_METADATA_FEEDERS = MAX_DMA_ENGINES + 3,
#endif

            BARRIERS_PER_GROUP = 8,
            DPUS_IN_CLUSTER = MAX_DPUS / MAX_CLUSTERS,

            TOTAL_PHYSICAL_BARRIERS = 64,
            TOTAL_USED_BARRIERS = BARRIERS_PER_GROUP * MAX_CLUSTERS,

            FIFO_COUNT = MAX_CLUSTERS,
            FIFO_LENGTH = 1024 / FIFO_COUNT,

            // minimum is 1
            MAX_CONCURRENT_SOFT_LAYERS_PER_WORKER = 2,
        };
    }
}

#endif // NN_RESOURCES_H_
